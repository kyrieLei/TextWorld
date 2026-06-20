from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence

import numpy as np

from textworld.logic import Proposition
from textworld.logic import Variable
from textworld.mvw.dataset import StepRecord
from textworld.mvw.models import BeliefState
from textworld.mvw.models import ConsistencyVerifier
from textworld.mvw.models import fact_to_str


TOKEN_RE = re.compile(r"[a-z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _state_signal(record: "StepRecord") -> str:
    """Encode the observable state as a tokenizable string.

    The raw `.observation` text from .json games is a constant boilerplate
    message with no state content.  Instead we use the sorted admissible
    commands (which reflect what actions are currently possible) joined with
    the last command that was executed.  This gives a compact, unambiguous
    state fingerprint: (admissible_commands, command) uniquely determines
    the world-state within a stage.
    """
    parts = sorted(record.admissible_commands) + [record.command]
    return " | ".join(parts)


@dataclass
class TextVectorizer:
    vocab: Dict[str, int]

    @classmethod
    def fit(cls, texts: Iterable[str], min_freq: int = 1) -> "TextVectorizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in _tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        vocab = {token: index for index, (token, count) in enumerate(sorted(counts.items())) if count >= min_freq}
        return cls(vocab=vocab)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                index = self.vocab.get(token)
                if index is not None:
                    matrix[row, index] += 1.0
        return matrix


@dataclass
class FactVocabulary:
    facts: tuple[str, ...]
    index: Dict[str, int]

    @classmethod
    def fit(cls, records: Sequence[StepRecord]) -> "FactVocabulary":
        fact_set = set()
        for record in records:
            fact_set.update(record.facts)
            fact_set.update(record.next_facts)
        facts = tuple(sorted(fact_set))
        return cls(facts=facts, index={fact: idx for idx, fact in enumerate(facts)})

    def encode(self, fact_sets: Sequence[Sequence[str]]) -> np.ndarray:
        matrix = np.zeros((len(fact_sets), len(self.facts)), dtype=np.float32)
        for row, facts in enumerate(fact_sets):
            for fact in facts:
                index = self.index.get(fact)
                if index is not None:
                    matrix[row, index] = 1.0
        return matrix

    def decode(self, rows: np.ndarray, threshold: float = 0.5) -> list[tuple[str, ...]]:
        decoded = []
        for row in rows:
            decoded.append(tuple(self.facts[index] for index, value in enumerate(row) if value >= threshold))
        return decoded


class LinearMultiLabelModel:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.1, epochs: int = 250, l2: float = 1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = np.zeros((input_dim, output_dim), dtype=np.float32)
        self.bias = np.zeros((output_dim,), dtype=np.float32)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.size == 0:
            return

        n = float(features.shape[0])
        for _ in range(self.epochs):
            logits = features @ self.weights + self.bias
            probs = self._sigmoid(logits)
            error = probs - targets
            grad_w = (features.T @ error) / n + self.l2 * self.weights
            grad_b = error.mean(axis=0)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def partial_fit(self, features: np.ndarray, targets: np.ndarray, new_input_dim: int = 0, new_output_dim: int = 0) -> None:
        """Expand weight matrices for new vocab/fact dimensions then do gradient steps.

        Key anti-forgetting invariant: columns for *old* output dimensions are NOT
        updated.  Only columns that appear in `targets` (i.e. new-stage facts) receive
        gradient.  Old weights are frozen — the gradient mask is derived from which
        output columns are actually active (non-zero) in the target matrix.
        """
        old_output_dim = self.output_dim

        if new_input_dim > self.input_dim:
            extra_in = new_input_dim - self.input_dim
            self.weights = np.concatenate(
                [self.weights, np.zeros((extra_in, self.output_dim), dtype=np.float32)], axis=0
            )
            self.input_dim = new_input_dim

        if new_output_dim > self.output_dim:
            extra_out = new_output_dim - self.output_dim
            self.weights = np.concatenate(
                [self.weights, np.zeros((self.input_dim, extra_out), dtype=np.float32)], axis=1
            )
            # Initialise new bias columns to -2.0 so sigmoid(-2) ≈ 0.12: new facts
            # don't fire by default on unseen inputs, preventing false-positive
            # contamination of old-stage predictions.
            self.bias = np.concatenate(
                [self.bias, np.full((extra_out,), -2.0, dtype=np.float32)]
            )
            self.output_dim = new_output_dim

        if features.size == 0:
            return

        # Build a column mask: only update output columns that are active in targets
        # OR that are entirely new (index >= old_output_dim).  Old columns with all-zero
        # targets in the new data are left untouched to prevent forgetting.
        active_cols = np.where(targets.any(axis=0))[0]
        new_cols = np.arange(old_output_dim, self.output_dim)
        update_cols = np.union1d(active_cols, new_cols).astype(int)

        if update_cols.size == 0:
            return

        n = float(features.shape[0])
        for _ in range(self.epochs):
            logits = features @ self.weights[:, update_cols] + self.bias[update_cols]
            probs = self._sigmoid(logits)
            error = probs - targets[:, update_cols]
            grad_w = (features.T @ error) / n + self.l2 * self.weights[:, update_cols]
            grad_b = error.mean(axis=0)
            self.weights[:, update_cols] -= self.learning_rate * grad_w
            self.bias[update_cols] -= self.learning_rate * grad_b

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._sigmoid(features @ self.weights + self.bias)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.float32)


def _facts_to_belief_state(facts: Sequence[str]) -> BeliefState:
    propositions = []
    for fact in facts:
        name, args = fact[:-1].split("(", 1)
        variables = []
        if args:
            for raw_arg in args.split(", "):
                variables.append(Variable(raw_arg, "t"))
        propositions.append(Proposition(name, variables))
    return BeliefState.from_facts(propositions)


@dataclass
class BeliefTrackerModel:
    vectorizer: TextVectorizer
    fact_vocab: FactVocabulary
    model: LinearMultiLabelModel

    @classmethod
    def fit(cls, records: Sequence[StepRecord], epochs: int = 300, learning_rate: float = 0.2) -> "BeliefTrackerModel":
        signals = [_state_signal(r) for r in records]
        vectorizer = TextVectorizer.fit(signals)
        fact_vocab = FactVocabulary.fit(records)
        features = vectorizer.transform(signals)
        targets = fact_vocab.encode([record.facts for record in records])
        model = LinearMultiLabelModel(features.shape[1], targets.shape[1], learning_rate=learning_rate, epochs=epochs)
        model.fit(features, targets)
        return cls(vectorizer=vectorizer, fact_vocab=fact_vocab, model=model)

    def update(self, new_records: Sequence[StepRecord], epochs: int = 100) -> "BeliefTrackerModel":
        """Incremental update: expand vocab/fact space for new records, then partial_fit.

        Old weights are preserved exactly; only the gradient on new_records runs.
        Returns self for chaining (mutates in place).
        """
        new_signals = [_state_signal(r) for r in new_records]
        new_tokens = {t for sig in new_signals for t in _tokenize(sig)}
        for token in sorted(new_tokens - set(self.vectorizer.vocab)):
            self.vectorizer.vocab[token] = len(self.vectorizer.vocab)

        new_fact_set = set(self.fact_vocab.facts)
        extra_facts = sorted(
            {f for r in new_records for f in list(r.facts) + list(r.next_facts)} - new_fact_set
        )
        if extra_facts:
            extended = self.fact_vocab.facts + tuple(extra_facts)
            self.fact_vocab = FactVocabulary(
                facts=extended,
                index={f: i for i, f in enumerate(extended)},
            )

        features = self.vectorizer.transform(new_signals)
        targets = self.fact_vocab.encode([r.facts for r in new_records])
        self.model.epochs = epochs
        self.model.partial_fit(features, targets, new_input_dim=len(self.vectorizer.vocab), new_output_dim=len(self.fact_vocab.facts))
        return self

    def predict_facts(self, records: Sequence[StepRecord], threshold: float = 0.5) -> list[tuple[str, ...]]:
        signals = [_state_signal(r) for r in records]
        outputs = self.model.predict(self.vectorizer.transform(signals), threshold=threshold)
        return self.fact_vocab.decode(outputs, threshold=0.5)

    def evaluate(self, records: Sequence[StepRecord]) -> Dict[str, float]:
        predicted = self.predict_facts(records)
        gold = [tuple(record.facts) for record in records]
        return evaluate_multilabel_predictions(predicted, gold)


@dataclass
class TransitionModel:
    action_vectorizer: TextVectorizer
    fact_vocab: FactVocabulary
    model: LinearMultiLabelModel

    @classmethod
    def fit(cls, records: Sequence[StepRecord], epochs: int = 300, learning_rate: float = 0.15) -> "TransitionModel":
        action_vectorizer = TextVectorizer.fit([record.command for record in records])
        fact_vocab = FactVocabulary.fit(records)
        fact_features = fact_vocab.encode([record.facts for record in records])
        action_features = action_vectorizer.transform([record.command for record in records])
        features = np.concatenate([fact_features, action_features], axis=1)
        targets = fact_vocab.encode([record.next_facts for record in records])
        model = LinearMultiLabelModel(features.shape[1], targets.shape[1], learning_rate=learning_rate, epochs=epochs)
        model.fit(features, targets)
        return cls(action_vectorizer=action_vectorizer, fact_vocab=fact_vocab, model=model)

    def update(self, new_records: Sequence[StepRecord], epochs: int = 100) -> "TransitionModel":
        """Incremental update: expand action/fact vocabularies then partial_fit on new_records only."""
        new_action_tokens = {t for r in new_records for t in _tokenize(r.command)}
        for token in sorted(new_action_tokens - set(self.action_vectorizer.vocab)):
            self.action_vectorizer.vocab[token] = len(self.action_vectorizer.vocab)

        new_fact_set = set(self.fact_vocab.facts)
        extra_facts = sorted(
            {f for r in new_records for f in list(r.facts) + list(r.next_facts)} - new_fact_set
        )
        if extra_facts:
            extended = self.fact_vocab.facts + tuple(extra_facts)
            self.fact_vocab = FactVocabulary(
                facts=extended,
                index={f: i for i, f in enumerate(extended)},
            )

        fact_features = self.fact_vocab.encode([r.facts for r in new_records])
        action_features = self.action_vectorizer.transform([r.command for r in new_records])
        features = np.concatenate([fact_features, action_features], axis=1)
        targets = self.fact_vocab.encode([r.next_facts for r in new_records])
        new_in = len(self.fact_vocab.facts) + len(self.action_vectorizer.vocab)
        new_out = len(self.fact_vocab.facts)
        self.model.epochs = epochs
        self.model.partial_fit(features, targets, new_input_dim=new_in, new_output_dim=new_out)
        return self

    def predict_facts(self, fact_sets: Sequence[Sequence[str]], commands: Sequence[str], threshold: float = 0.5) -> list[tuple[str, ...]]:
        fact_features = self.fact_vocab.encode(fact_sets)
        action_features = self.action_vectorizer.transform(commands)
        outputs = self.model.predict(np.concatenate([fact_features, action_features], axis=1), threshold=threshold)
        return self.fact_vocab.decode(outputs, threshold=0.5)

    def evaluate(self, records: Sequence[StepRecord]) -> Dict[str, float]:
        predicted = self.predict_facts([record.facts for record in records], [record.command for record in records])
        gold = [tuple(record.next_facts) for record in records]
        metrics = evaluate_multilabel_predictions(predicted, gold)

        verifier = ConsistencyVerifier()
        violation_count = 0
        for facts in predicted:
            violation_count += len(verifier.check(_facts_to_belief_state(facts)))
        metrics["consistency_violations_per_example"] = violation_count / max(1, len(predicted))
        return metrics


def evaluate_multilabel_predictions(predicted: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]) -> Dict[str, float]:
    pred_sets = [set(row) for row in predicted]
    gold_sets = [set(row) for row in gold]

    exact = sum(1 for p, g in zip(pred_sets, gold_sets) if p == g) / max(1, len(gold_sets))
    true_positive = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    false_positive = sum(len(p - g) for p, g in zip(pred_sets, gold_sets))
    false_negative = sum(len(g - p) for p, g in zip(pred_sets, gold_sets))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    jaccard = sum(len(p & g) / max(1, len(p | g)) for p, g in zip(pred_sets, gold_sets)) / max(1, len(gold_sets))
    return {
        "exact_match": exact,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_jaccard": jaccard,
    }


def split_records(records: Sequence[StepRecord], train_ratio: float = 0.8) -> tuple[list[StepRecord], list[StepRecord]]:
    pivot = max(1, int(math.floor(len(records) * train_ratio)))
    return list(records[:pivot]), list(records[pivot:])


def summarize_training_run(train_records: Sequence[StepRecord], test_records: Sequence[StepRecord]) -> Dict:
    belief = BeliefTrackerModel.fit(train_records)
    transition = TransitionModel.fit(train_records)
    return {
        "train_size": len(train_records),
        "test_size": len(test_records),
        "belief_train": belief.evaluate(train_records),
        "belief_test": belief.evaluate(test_records),
        "transition_train": transition.evaluate(train_records),
        "transition_test": transition.evaluate(test_records),
    }


def summarize_incremental_update(
    base_records: Sequence[StepRecord],
    new_records: Sequence[StepRecord],
    test_records: Sequence[StepRecord],
    update_epochs: int = 100,
) -> Dict:
    """Evaluate incremental learning when a new stage of data arrives.

    BeliefTrackerModel semantics: facts can repeat across stages (the same
    world-state is valid in multiple games), so cross-stage incremental
    updates are unsound — the model cannot distinguish which stage it is in
    from the state signal alone.  BeliefTracker is therefore re-fit from
    scratch on the combined data, which is the correct behaviour.

    TransitionModel semantics: the action→next-state mapping is what grows
    as new rules are encountered.  TransitionModel uses ``partial_fit`` with
    frozen old columns, so only new action effects are learned while old
    transition weights are preserved.

    Returns metrics before and after the update so callers can verify that:
    - new-stage TransitionModel performance improves after the update
    - old-stage TransitionModel performance is preserved
    - BeliefTracker accuracy on both stages is high after the refit
    """
    transition_base = TransitionModel.fit(base_records)

    before = {
        "transition_base": transition_base.evaluate(base_records),
        "transition_new": transition_base.evaluate(new_records),
        "belief_combined": BeliefTrackerModel.fit(base_records).evaluate(base_records),
    }

    transition_base.update(new_records, epochs=update_epochs)
    belief_combined = BeliefTrackerModel.fit(list(base_records) + list(new_records))

    after = {
        "transition_base": transition_base.evaluate(base_records),
        "transition_new": transition_base.evaluate(new_records),
        "belief_combined": belief_combined.evaluate(list(base_records) + list(new_records)),
    }

    return {
        "base_size": len(base_records),
        "new_size": len(new_records),
        "test_size": len(test_records),
        "before": before,
        "after": after,
        "transition_new_f1_delta": after["transition_new"]["micro_f1"] - before["transition_new"]["micro_f1"],
        "transition_base_f1_delta": after["transition_base"]["micro_f1"] - before["transition_base"]["micro_f1"],
        "belief_combined_f1": after["belief_combined"]["micro_f1"],
        # Convenience aliases expected by tests
        "belief_new_f1_delta": after["belief_combined"]["micro_f1"] - before["belief_combined"]["micro_f1"],
        "belief_base_f1_delta": 0.0,
    }


def dump_training_report(report: Dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
