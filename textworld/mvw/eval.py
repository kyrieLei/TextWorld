from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

import tempfile

import textworld
from textworld.core import EnvInfos
from textworld.mvw.curriculum import STAGE_ORDER
from textworld.mvw.curriculum import build_stage_game
from textworld.mvw.curriculum import normalize_stage
from textworld.mvw.models import BeliefState
from textworld.mvw.models import NoveltyDetector
from textworld.mvw.models import OracleStateTracker
from textworld.mvw.models import RuleBasedExpansionPlanner
from textworld.mvw.models import SymbolicTransitionModel
from textworld.mvw.models import WorldContext
from textworld.mvw.models import WorldPatch
from textworld.mvw.models import fact_to_str
from textworld.mvw.runner import evaluate_game


@dataclass(frozen=True)
class CounterfactualProbe:
    stage: str
    command: str
    expect_state_change: bool
    description: str


COUNTERFACTUAL_PROBES: Dict[str, tuple[CounterfactualProbe, ...]] = {
    "stage_2": (
        CounterfactualProbe("stage_2", "take apple from fridge", False, "Cannot take from a closed fridge."),
        CounterfactualProbe("stage_2", "open fridge", True, "Opening the fridge changes visibility and access."),
    ),
    "stage_3": (
        CounterfactualProbe("stage_3", "open wooden door", False, "Locked doors should not open directly."),
        CounterfactualProbe("stage_3", "take old key from nightstand", True, "The key can be moved into inventory."),
    ),
    "stage_5": (
        CounterfactualProbe("stage_5", "use blue portal", True, "Using the portal should teleport the player."),
    ),
}


def _request_infos() -> EnvInfos:
    return EnvInfos(facts=True, admissible_commands=True, policy_commands=True, feedback=True)


def _save_temp_game(game) -> str:
    tmpdir = tempfile.mkdtemp(prefix="tw-mvw-eval-")
    path = pjoin(tmpdir, "{}.json".format(game.metadata["stage"]))
    game.save(path)
    return path


def _facts_to_strings(facts) -> tuple[str, ...]:
    return tuple(sorted(fact_to_str(fact) for fact in facts))


def _state_changed(before: Sequence[str], after: Sequence[str]) -> bool:
    return tuple(before) != tuple(after)


def _make_model(stage: str, known_stage: str, patches: Sequence[WorldPatch] = ()) -> tuple[SymbolicTransitionModel, WorldContext]:
    game = build_stage_game(stage)
    context = WorldContext(game)
    return SymbolicTransitionModel(context, int(known_stage.split("_")[-1]), patches), context


def evaluate_counterfactuals(stage: Union[int, str], known_stage: Optional[Union[int, str]] = None, expand: bool = False, seed: int = 1234) -> Dict:
    stage_id = normalize_stage(stage)
    known_stage_id = normalize_stage(known_stage) if known_stage is not None else stage_id
    probes = COUNTERFACTUAL_PROBES.get(stage_id, ())
    if not probes:
        return {"stage": stage_id, "counterfactual_accuracy": 1.0, "probes": []}

    game = build_stage_game(stage_id, seed=seed)
    env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
    state = env.reset()
    tracker = OracleStateTracker()
    detector = NoveltyDetector()
    planner = RuleBasedExpansionPlanner()
    context = WorldContext(game)
    model = SymbolicTransitionModel(context, int(known_stage_id.split("_")[-1]))
    belief = tracker.observe(state.facts)

    # Optionally preload the expansion patch using the first detected novelty on the walkthrough.
    if expand:
        for command in game.metadata.get("walkthrough", []):
            predicted, supported = model.predict(belief, command)
            next_state, _, _ = env.step(command)
            observed = tracker.observe(next_state.facts)
            signal = detector.detect(command, predicted, observed, supported, ())
            patch = planner.propose(signal)
            if patch is not None:
                model = model.with_patch(patch)
                break
            belief = observed

        env.close()
        env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
        state = env.reset()
        belief = tracker.observe(state.facts)

    results = []
    correct = 0
    for probe in probes:
        predicted_state, _ = model.predict(belief, probe.command)
        predicted_change = _state_changed(_facts_to_strings(state.facts), tuple(sorted(fact_to_str(fact) for fact in predicted_state.facts)))

        next_state, _, _ = env.step(probe.command)
        observed_change = _state_changed(_facts_to_strings(state.facts), _facts_to_strings(next_state.facts))
        is_correct = predicted_change == observed_change == probe.expect_state_change
        correct += int(is_correct)
        results.append(
            {
                "command": probe.command,
                "description": probe.description,
                "expect_state_change": probe.expect_state_change,
                "predicted_state_change": predicted_change,
                "observed_state_change": observed_change,
                "correct": is_correct,
            }
        )
        env.close()
        env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
        state = env.reset()
        belief = tracker.observe(state.facts)

    env.close()
    return {
        "stage": stage_id,
        "counterfactual_accuracy": correct / max(1, len(results)),
        "probes": results,
    }


def evaluate_novelty_accommodation(stage: Union[int, str] = "stage_5", base_known_stage: Union[int, str] = "stage_4", seed: int = 1234) -> Dict:
    before = evaluate_game(stage, known_stage=base_known_stage, seed=seed, expand=False)
    after = evaluate_game(stage, known_stage=base_known_stage, seed=seed, expand=True)

    def _performance(report: Dict) -> float:
        if not report["won"]:
            return 0.0
        return 1.0 if report["unresolved_novelty_steps"] == 0 else 0.0

    before_score = _performance(before)
    after_score = _performance(after)
    return {
        "stage": normalize_stage(stage),
        "known_stage": normalize_stage(base_known_stage),
        "performance_before": before_score,
        "performance_after": after_score,
        "novelty_accommodation": after_score - before_score,
        "before": before,
        "after": after,
    }


def evaluate_rule_minimality(stage: Union[int, str] = "stage_5", base_known_stage: Union[int, str] = "stage_4", seed: int = 1234) -> Dict:
    stage_id = normalize_stage(stage)
    game = build_stage_game(stage_id, seed=seed)
    env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
    tracker = OracleStateTracker()
    detector = NoveltyDetector()
    planner = RuleBasedExpansionPlanner()
    context = WorldContext(game)
    model = SymbolicTransitionModel(context, int(normalize_stage(base_known_stage).split("_")[-1]))
    state = env.reset()
    belief = tracker.observe(state.facts)
    patches = []

    for command in game.metadata.get("walkthrough", []):
        predicted, supported = model.predict(belief, command)
        next_state, _, _ = env.step(command)
        observed = tracker.observe(next_state.facts)
        signal = detector.detect(command, predicted, observed, supported, ())
        patch = planner.propose(signal)
        if patch is not None:
            patches.append(patch)
            model = model.with_patch(patch)
        belief = observed

    env.close()
    return {
        "stage": stage_id,
        "num_patches": len(patches),
        "rule_minimality": sum(patch.complexity for patch in patches),
        "patches": [
            {
                "name": patch.name,
                "kind": patch.kind,
                "complexity": patch.complexity,
                "new_entities": list(patch.new_entities),
                "new_properties": list(patch.new_properties),
                "rules": list(patch.rules),
            }
            for patch in patches
        ],
    }


def _canonical_fact_string(fact, context: WorldContext) -> str:
    args = []
    for arg in fact.arguments:
        args.append(context.id_to_name.get(arg.name, arg.name))
    return "{}({})".format(fact.name, ", ".join(args)) if args else "{}()".format(fact.name)


def _goal_reached(game, belief_state: BeliefState, context: WorldContext) -> bool:
    goal_fact_strings = set()
    for quest in game.quests:
        for event in quest.win_events:
            goal_fact_strings |= {_canonical_fact_string(fact, context) for fact in event.condition.preconditions}
    belief_fact_strings = {fact_to_str(fact) for fact in belief_state.facts}
    return goal_fact_strings.issubset(belief_fact_strings)


def _candidate_commands(game, state) -> list[str]:
    walkthrough = list(game.metadata.get("walkthrough", []))
    admissible = list(state.admissible_commands or [])
    commands = []
    seen = set()
    for command in walkthrough + admissible:
        if command not in seen:
            commands.append(command)
            seen.add(command)
    return commands


def plan_with_model(stage: Union[int, str], known_stage: Optional[Union[int, str]] = None, expand: bool = False, seed: int = 1234, max_depth: int = 4) -> Dict:
    stage_id = normalize_stage(stage)
    known_stage_id = normalize_stage(known_stage) if known_stage is not None else stage_id
    game = build_stage_game(stage_id, seed=seed)
    env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
    initial_state = env.reset()
    tracker = OracleStateTracker()
    detector = NoveltyDetector()
    planner = RuleBasedExpansionPlanner()
    context = WorldContext(game)
    model = SymbolicTransitionModel(context, int(known_stage_id.split("_")[-1]))
    initial_belief = tracker.observe(initial_state.facts)

    if expand:
        for command in game.metadata.get("walkthrough", []):
            predicted, supported = model.predict(initial_belief, command)
            next_state, _, _ = env.step(command)
            observed = tracker.observe(next_state.facts)
            patch = planner.propose(detector.detect(command, predicted, observed, supported, ()))
            if patch is not None:
                model = model.with_patch(patch)
                break
            initial_belief = observed

        env.close()
        env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
        initial_state = env.reset()
        initial_belief = tracker.observe(initial_state.facts)

    command_pool = _candidate_commands(game, initial_state)
    max_depth = max(max_depth, len(game.metadata.get("walkthrough", [])))
    queue = deque([(initial_belief, [])])
    visited = {tuple(sorted(fact_to_str(fact) for fact in initial_belief.facts))}
    plan = None

    while queue:
        belief, actions = queue.popleft()
        if _goal_reached(game, belief, context):
            plan = actions
            break

        if len(actions) >= max_depth:
            continue

        for command in command_pool:
            next_belief, supported = model.predict(belief, command)
            if not supported:
                continue
            signature = tuple(sorted(fact_to_str(fact) for fact in next_belief.facts))
            if signature in visited:
                continue
            visited.add(signature)
            queue.append((next_belief, actions + [command]))

    env.close()
    return {
        "stage": stage_id,
        "known_stage": known_stage_id,
        "expand": expand,
        "plan_success": plan is not None,
        "plan": plan or [],
    }


def evaluate_planning_improvement(stage: Union[int, str] = "stage_5", base_known_stage: Union[int, str] = "stage_4", seed: int = 1234) -> Dict:
    before = plan_with_model(stage, known_stage=base_known_stage, expand=False, seed=seed)
    after = plan_with_model(stage, known_stage=base_known_stage, expand=True, seed=seed)
    return {
        "stage": normalize_stage(stage),
        "known_stage": normalize_stage(base_known_stage),
        "performance_before": float(before["plan_success"]),
        "performance_after": float(after["plan_success"]),
        "planning_improvement": float(after["plan_success"]) - float(before["plan_success"]),
        "before": before,
        "after": after,
    }


def evaluate_benchmark(known_stage: Union[int, str], novelty_stage: Union[int, str] = "stage_5", seed: int = 1234) -> Dict:
    known_stage_id = normalize_stage(known_stage)
    retention = {
        "known_stage": known_stage_id,
        "old_world_retention": all(
            evaluate_game(stage, known_stage=known_stage_id, seed=seed + idx)["won"]
            for idx, stage in enumerate(STAGE_ORDER[: int(known_stage_id.split("_")[-1]) + 1])
        ),
    }
    accommodation = evaluate_novelty_accommodation(stage=novelty_stage, base_known_stage=known_stage_id, seed=seed)
    counterfactual = evaluate_counterfactuals(novelty_stage, known_stage=known_stage_id, expand=True, seed=seed)
    rule_minimality = evaluate_rule_minimality(stage=novelty_stage, base_known_stage=known_stage_id, seed=seed)
    planning = evaluate_planning_improvement(stage=novelty_stage, base_known_stage=known_stage_id, seed=seed)
    consistency = accommodation["after"]["observed_violation_count"] / max(1, accommodation["after"]["steps"])
    return {
        "known_stage": known_stage_id,
        "novelty_stage": normalize_stage(novelty_stage),
        "old_world_retention": retention["old_world_retention"],
        "novelty_accommodation": accommodation["novelty_accommodation"],
        "consistency_violation_rate": consistency,
        "rule_minimality": rule_minimality["rule_minimality"],
        "counterfactual_accuracy": counterfactual["counterfactual_accuracy"],
        "planning_improvement": planning["planning_improvement"],
        "details": {
            "retention": retention,
            "accommodation": accommodation,
            "counterfactual": counterfactual,
            "rule_minimality": rule_minimality,
            "planning": planning,
        },
    }
