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
        vectorizer = TextVectorizer.fit([record.observation for record in records])
        fact_vocab = FactVocabulary.fit(records)
        features = vectorizer.transform([record.observation for record in records])
        targets = fact_vocab.encode([record.facts for record in records])
        model = LinearMultiLabelModel(features.shape[1], targets.shape[1], learning_rate=learning_rate, epochs=epochs)
        model.fit(features, targets)
        return cls(vectorizer=vectorizer, fact_vocab=fact_vocab, model=model)

    def predict_facts(self, observations: Sequence[str], threshold: float = 0.5) -> list[tuple[str, ...]]:
        outputs = self.model.predict(self.vectorizer.transform(observations), threshold=threshold)
        return self.fact_vocab.decode(outputs, threshold=0.5)

    def evaluate(self, records: Sequence[StepRecord]) -> Dict[str, float]:
        predicted = self.predict_facts([record.observation for record in records])
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


def dump_training_report(report: Dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
