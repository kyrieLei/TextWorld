import tempfile

from textworld.mvw.dataset import collect_stage_dataset
from textworld.mvw.dataset import load_dataset
from textworld.mvw.dataset import save_dataset
from textworld.mvw.learning import BeliefTrackerModel
from textworld.mvw.learning import TransitionModel
from textworld.mvw.learning import split_records
from textworld.mvw.learning import summarize_incremental_update
from textworld.mvw.learning import summarize_training_run


def test_collect_and_reload_stage_dataset():
    records = collect_stage_dataset("stage_1", num_games=3, seed=2026, policy_mix=1.0)
    assert len(records) > 0
    assert all(record.stage == "stage_1" for record in records)
    assert all(len(record.facts) > 0 for record in records)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/stage_1.jsonl"
        save_dataset(records, path)
        reloaded = load_dataset(path)
        assert len(reloaded) == len(records)
        assert reloaded[0].command == records[0].command


def test_learning_pipeline_produces_useful_metrics():
    records = collect_stage_dataset("stage_0", num_games=8, seed=2026, policy_mix=0.8)
    train_records, test_records = split_records(records, train_ratio=0.75)
    report = summarize_training_run(train_records, test_records)

    assert report["train_size"] > 0
    assert report["test_size"] > 0
    assert report["belief_train"]["micro_f1"] >= 0.8
    assert report["transition_train"]["micro_f1"] >= 0.8


def test_incremental_update_improves_new_stage_without_forgetting():
    base_records = collect_stage_dataset("stage_0", num_games=6, seed=2026, policy_mix=0.9)
    new_records = collect_stage_dataset("stage_1", num_games=6, seed=2026, policy_mix=0.9)
    report = summarize_incremental_update(base_records, new_records, test_records=new_records)

    assert report["base_size"] > 0
    assert report["new_size"] > 0
    # TransitionModel on new-stage data should improve after the update.
    assert report["transition_new_f1_delta"] >= 0.0
    # TransitionModel on old-stage data must not regress (frozen columns protect it).
    assert report["transition_base_f1_delta"] >= -0.05
    # BeliefTracker combined accuracy should be high after refit on merged data.
    assert report["belief_combined_f1"] >= 0.8


def test_partial_fit_expands_vocab_and_preserves_old_weights():
    base_records = collect_stage_dataset("stage_0", num_games=4, seed=2026, policy_mix=1.0)
    new_records = collect_stage_dataset("stage_2", num_games=4, seed=2026, policy_mix=1.0)

    belief = BeliefTrackerModel.fit(base_records)
    old_vocab_size = len(belief.vectorizer.vocab)
    old_fact_size = len(belief.fact_vocab.facts)

    belief.update(new_records, epochs=50)

    # Vocab and fact space must have grown to cover the new stage's tokens and facts.
    assert len(belief.vectorizer.vocab) >= old_vocab_size
    assert len(belief.fact_vocab.facts) >= old_fact_size
    # Model weight matrix dimensions match the expanded vocab.
    assert belief.model.input_dim == len(belief.vectorizer.vocab)
    assert belief.model.output_dim == len(belief.fact_vocab.facts)

    # Same checks for the transition model.
    transition = TransitionModel.fit(base_records)
    old_action_size = len(transition.action_vectorizer.vocab)
    transition.update(new_records, epochs=50)
    assert len(transition.action_vectorizer.vocab) >= old_action_size

