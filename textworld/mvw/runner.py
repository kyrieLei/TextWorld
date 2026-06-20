from __future__ import annotations

import json
import tempfile
from os.path import join as pjoin
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import textworld
from textworld.core import EnvInfos
from textworld.mvw.curriculum import STAGE_ORDER
from textworld.mvw.curriculum import build_stage_game
from textworld.mvw.curriculum import normalize_stage
from textworld.mvw.models import ConsistencyVerifier
from textworld.mvw.models import NoveltyDetector
from textworld.mvw.models import OracleStateTracker
from textworld.mvw.models import RuleBasedExpansionPlanner
from textworld.mvw.models import SymbolicTransitionModel
from textworld.mvw.models import WorldContext
from textworld.mvw.models import fact_to_str
from textworld.mvw.scenarios import apply_custom_goal
from textworld.mvw.scenarios import apply_novelty_runtime


def _request_infos() -> EnvInfos:
    return EnvInfos(facts=True, policy_commands=True, admissible_commands=True, intermediate_reward=True)


def build_curriculum(output_dir: str, stages: Optional[Iterable[Union[int, str]]] = None, seed: int = 1234) -> list[str]:
    stages = stages or STAGE_ORDER
    saved_paths: list[str] = []
    for index, stage in enumerate(stages):
        stage_id = normalize_stage(stage)
        game = build_stage_game(stage_id, seed=seed + index)
        path = pjoin(output_dir, "{}.json".format(stage_id))
        game.save(path)
        saved_paths.append(path)

    return saved_paths


def _save_temp_game(game) -> str:
    tmpdir = tempfile.mkdtemp(prefix="tw-mvw-")
    path = pjoin(tmpdir, "{}.json".format(game.metadata["stage"]))
    game.save(path)
    return path


def evaluate_game(
    stage: Union[int, str],
    known_stage: Optional[Union[int, str]] = None,
    seed: int = 1234,
    expand: bool = False,
    llm_proposer=None,
    novelty_scenario: str = None,
) -> Dict:
    stage_id = normalize_stage(stage)
    game = build_stage_game(stage_id, seed=seed, novelty_scenario=novelty_scenario)
    game_path = _save_temp_game(game)
    env = textworld.start(game_path, request_infos=_request_infos())
    tracker = OracleStateTracker()
    verifier = ConsistencyVerifier()
    detector = NoveltyDetector()
    context = WorldContext(game)
    model_stage = normalize_stage(known_stage) if known_stage is not None else stage_id
    model = SymbolicTransitionModel(context, int(model_stage.split("_")[-1]))
    planner = RuleBasedExpansionPlanner()

    current_state = env.reset()
    belief = tracker.observe(current_state.facts)
    commands = list(game.metadata.get("walkthrough", current_state.policy_commands or []))
    traces = []

    for command in commands:
        predicted, supported = model.predict(belief, command)
        predicted_violations = verifier.check(predicted)

        next_state, reward, done = env.step(command)
        next_state = apply_novelty_runtime(next_state, command, game.metadata.get("novelty_scenario"))
        next_state = apply_custom_goal(next_state, stage_id, game.metadata.get("novelty_scenario"))
        done = bool(next_state.done)
        observed = tracker.observe(next_state.facts)
        signal = detector.detect(command, predicted, observed, supported, predicted_violations)

        patch = None
        llm_hypothesis = None
        if expand and signal.is_novel:
            if llm_proposer is not None:
                try:
                    llm_hypothesis = llm_proposer.propose(signal)
                except Exception as exc:
                    llm_hypothesis = "LLM proposer failed: {}".format(exc)
            patch = planner.propose(signal)
            if patch is not None:
                model = model.with_patch(patch)
                predicted, supported = model.predict(belief, command)
                predicted_violations = verifier.check(predicted)
                signal = detector.detect(command, predicted, observed, supported, predicted_violations)

        traces.append(
            {
                "command": command,
                "reward": reward,
                "done": done,
                "supported": supported,
                "novel": signal.is_novel,
                "missing_facts": list(signal.missing_facts),
                "unexpected_facts": list(signal.unexpected_facts),
                "predicted_violations": list(signal.predicted_violations),
                "observed_violations": [violation.code for violation in verifier.check(observed)],
                "patch": patch.kind if patch is not None else None,
                "llm_hypothesis": llm_hypothesis,
                "player_room": next((fact_to_str(fact) for fact in observed.facts if fact.name == "at" and fact.arguments[0].name == "P"), None),
            }
        )
        belief = observed
        current_state = next_state

    env.close()
    novelty_steps = sum(1 for trace in traces if trace["novel"])
    unresolved_novelty_steps = sum(1 for trace in traces if trace["novel"] and not trace["patch"])
    observed_violations = sum(len(trace["observed_violations"]) for trace in traces)
    task_won = bool(current_state.won) and unresolved_novelty_steps == 0
    return {
        "stage": stage_id,
        "novelty_scenario": game.metadata.get("novelty_scenario"),
        "known_stage": model_stage,
        "expand": expand,
        "final_score": current_state.max_score if task_won else 0,
        "max_score": current_state.max_score,
        "won": task_won,
        "steps": len(traces),
        "novelty_steps": novelty_steps,
        "unresolved_novelty_steps": unresolved_novelty_steps,
        "observed_violation_count": observed_violations,
        "trace": traces,
    }


def evaluate_retention(up_to_stage: Union[int, str], seed: int = 1234) -> Dict:
    known_stage = normalize_stage(up_to_stage)
    max_index = int(known_stage.split("_")[-1])
    reports = []
    for stage_name in STAGE_ORDER[: max_index + 1]:
        report = evaluate_game(stage_name, known_stage=known_stage, seed=seed + max_index)
        reports.append(report)

    return {
        "known_stage": known_stage,
        "old_world_retention": all(report["won"] and report["unresolved_novelty_steps"] == 0 for report in reports),
        "reports": reports,
    }


def report_to_json(report: Dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
