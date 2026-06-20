from textworld.mvw.eval import evaluate_benchmark
from textworld.mvw.eval import evaluate_counterfactuals
from textworld.mvw.eval import evaluate_novelty_accommodation
from textworld.mvw.eval import evaluate_patch_transfer
from textworld.mvw.eval import evaluate_planning_improvement
from textworld.mvw.curriculum import build_stage_game


def test_counterfactual_probe_catches_locked_door_rule():
    report = evaluate_counterfactuals("stage_3", known_stage="stage_3", seed=2026)
    assert report["counterfactual_accuracy"] == 1.0
    assert any(probe["command"] == "open wooden door" and probe["correct"] for probe in report["probes"])


def test_novelty_accommodation_and_planning_improve_with_expansion():
    accommodation = evaluate_novelty_accommodation(stage="stage_5", base_known_stage="stage_4", seed=2026)
    planning = evaluate_planning_improvement(stage="stage_5", base_known_stage="stage_4", seed=2026)

    assert accommodation["novelty_accommodation"] > 0.0
    assert planning["planning_improvement"] > 0.0


def test_benchmark_exposes_idea_metrics():
    report = evaluate_benchmark("stage_4", novelty_stage="stage_5", seed=2026)
    assert report["old_world_retention"] is True
    assert report["novelty_accommodation"] > 0.0
    assert report["counterfactual_accuracy"] == 1.0
    assert report["planning_improvement"] > 0.0


def test_magic_box_benchmark_exposes_idea_metrics():
    report = evaluate_benchmark("stage_4", novelty_stage="stage_5", novelty_scenario="magic_box", seed=2026)
    assert report["old_world_retention"] is True
    assert report["novelty_accommodation"] > 0.0
    assert report["counterfactual_accuracy"] == 1.0
    assert report["planning_improvement"] > 0.0
    assert report["details"]["transfer"]["transfer_improvement"] > 0.0


def test_magic_box_patch_transfers_across_seeded_objects():
    source_game = build_stage_game("stage_5", seed=2026, novelty_scenario="magic_box")
    target_game = build_stage_game("stage_5", seed=2027, novelty_scenario="magic_box")
    assert source_game.metadata["target_object"] != target_game.metadata["target_object"]

    report = evaluate_patch_transfer(
        stage="stage_5",
        base_known_stage="stage_4",
        discovery_seed=2026,
        eval_seed=2027,
        novelty_scenario="magic_box",
    )

    assert report["transfer_success_before"] == 0.0
    assert report["transfer_success_after"] == 1.0
    assert report["after"]["won"] is True
