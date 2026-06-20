import tempfile

from textworld.mvw.dataset import collect_stage_dataset
from textworld.mvw.dataset import load_dataset
from textworld.mvw.dataset import save_dataset
from textworld.mvw.learning import split_records
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
