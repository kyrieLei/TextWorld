from __future__ import annotations

from typing import Dict
from typing import Iterable
from typing import Sequence

from textworld.mvw.eval import evaluate_novelty_suite
from textworld.mvw.models import DataDrivenExpansionPlanner
from textworld.mvw.models import RuleBasedExpansionPlanner
from textworld.mvw.models import SearchExpansionPlanner
from textworld.mvw.scenarios import NOVELTY_SCENARIOS


def planner_registry() -> Dict[str, object]:
    return {
        "rule_based": RuleBasedExpansionPlanner(),
        "data_driven": DataDrivenExpansionPlanner(),
        "search": SearchExpansionPlanner(),
    }


def generate_ablation_report(
    known_stage: str = "stage_4",
    novelty_stage: str = "stage_5",
    seed: int = 1234,
    novelty_scenarios: Sequence[str] = NOVELTY_SCENARIOS,
    planners: Iterable[str] = ("rule_based", "data_driven", "search"),
) -> Dict:
    novelty_scenarios = tuple(novelty_scenarios or NOVELTY_SCENARIOS)
    registry = planner_registry()
    selected = {name: registry[name] for name in planners}
    return evaluate_novelty_suite(
        known_stage=known_stage,
        novelty_stage=novelty_stage,
        seed=seed,
        novelty_scenarios=novelty_scenarios,
        planners=selected,
    )


def format_ablation_markdown(report: Dict) -> str:
    lines = [
        "# MVW Ablation Report",
        "",
        "| scenario | planner | retention | accommodation | consistency | minimality | counterfactual | planning | transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        for planner in report["planners"]:
            metrics = report["results"][scenario][planner]
            transfer = metrics["details"]["transfer"]["transfer_improvement"]
            lines.append(
                "| {scenario} | {planner} | {retention} | {accommodation:.1f} | {consistency:.1f} | {minimality} | {counterfactual:.1f} | {planning:.1f} | {transfer:.1f} |".format(
                    scenario=scenario,
                    planner=planner,
                    retention=str(metrics["old_world_retention"]),
                    accommodation=metrics["novelty_accommodation"],
                    consistency=metrics["consistency_violation_rate"],
                    minimality=metrics["rule_minimality"],
                    counterfactual=metrics["counterfactual_accuracy"],
                    planning=metrics["planning_improvement"],
                    transfer=transfer,
                )
            )

    return "\n".join(lines) + "\n"
