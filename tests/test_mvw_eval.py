from textworld.mvw.eval import evaluate_benchmark
from textworld.mvw.eval import evaluate_counterfactuals
from textworld.mvw.eval import evaluate_novelty_accommodation
from textworld.mvw.eval import evaluate_novelty_suite
from textworld.mvw.eval import evaluate_patch_transfer
from textworld.mvw.eval import evaluate_planning_improvement
from textworld.mvw.models import DataDrivenExpansionPlanner
from textworld.mvw.models import SearchExpansionPlanner
from textworld.mvw.report import generate_ablation_report
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


def test_bridge_button_benchmark_and_transfer():
    source_game = build_stage_game("stage_5", seed=2026, novelty_scenario="bridge_button")
    target_game = build_stage_game("stage_5", seed=2027, novelty_scenario="bridge_button")
    assert source_game.metadata["trigger_name"] != target_game.metadata["trigger_name"]

    report = evaluate_benchmark("stage_4", novelty_stage="stage_5", novelty_scenario="bridge_button", seed=2026)
    assert report["old_world_retention"] is True
    assert report["novelty_accommodation"] > 0.0
    assert report["planning_improvement"] > 0.0

    transfer = evaluate_patch_transfer(
        stage="stage_5",
        base_known_stage="stage_4",
        discovery_seed=2026,
        eval_seed=2027,
        novelty_scenario="bridge_button",
    )
    assert transfer["transfer_success_before"] == 0.0
    assert transfer["transfer_success_after"] == 1.0


def test_data_driven_planner_recovers_portal_novelty():
    """DataDrivenExpansionPlanner must induce a portal_transition patch from signal alone."""
    planner = DataDrivenExpansionPlanner()
    report = evaluate_benchmark(
        "stage_4",
        novelty_stage="stage_5",
        novelty_scenario="portal",
        seed=2026,
        planner=planner,
    )
    assert report["old_world_retention"] is True
    assert report["novelty_accommodation"] > 0.0
    assert report["planning_improvement"] > 0.0
    assert report["details"]["accommodation"]["after"]["trace"][0]["patch"] == "portal_transition"


def test_data_driven_planner_recovers_magic_box_novelty():
    """DataDrivenExpansionPlanner must induce a transform_on_open patch from signal alone."""
    planner = DataDrivenExpansionPlanner()
    report = evaluate_benchmark(
        "stage_4",
        novelty_stage="stage_5",
        novelty_scenario="magic_box",
        seed=2026,
        planner=planner,
    )
    assert report["old_world_retention"] is True
    assert report["novelty_accommodation"] > 0.0
    assert report["planning_improvement"] > 0.0
    assert report["details"]["transfer"]["transfer_improvement"] > 0.0
    assert report["details"]["accommodation"]["after"]["trace"][0]["patch"] == "transform_on_open"
    assert report["rule_minimality"] == 4


def test_search_planner_supports_full_novelty_suite():
    planner = SearchExpansionPlanner()
    suite = evaluate_novelty_suite(
        known_stage="stage_4",
        novelty_stage="stage_5",
        seed=2026,
        planners={"search": planner},
    )
    assert set(suite["results"]) == {"portal", "magic_box", "bridge_button"}
    assert suite["results"]["bridge_button"]["search"]["planning_improvement"] > 0.0


def test_ablation_report_covers_requested_matrix():
    report = generate_ablation_report(
        known_stage="stage_4",
        novelty_stage="stage_5",
        seed=2026,
        novelty_scenarios=("portal", "bridge_button"),
        planners=("rule_based", "search"),
    )
    assert report["scenarios"] == ["portal", "bridge_button"]
    assert report["planners"] == ["rule_based", "search"]
    assert report["results"]["bridge_button"]["search"]["novelty_accommodation"] > 0.0
