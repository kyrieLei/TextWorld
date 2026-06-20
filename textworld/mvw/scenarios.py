from __future__ import annotations

from typing import Iterable
from typing import Optional

from textworld.core import GameState
from textworld.logic import Proposition
from textworld.logic import Variable
from textworld.mvw.models import fact_to_str


NOVELTY_SCENARIOS = ("portal", "magic_box")


def normalize_novelty_scenario(scenario: Optional[str]) -> str:
    scenario = (scenario or "portal").strip().lower()
    if scenario not in NOVELTY_SCENARIOS:
        raise KeyError("Unknown novelty scenario: {}".format(scenario))
    return scenario


def _has_fact(facts, name: str, *args: str) -> bool:
    return any(
        fact.name == name and tuple(arg.name for arg in fact.arguments) == tuple(args)
        for fact in facts
    )


def _remove_fact(facts: list[Proposition], name: str, *args: str) -> None:
    for fact in list(facts):
        if fact.name == name and tuple(arg.name for arg in fact.arguments) == tuple(args):
            facts.remove(fact)


def _add_fact(facts: list[Proposition], name: str, *args: tuple[str, str]) -> None:
    variables = [Variable(arg_name, arg_type) for arg_name, arg_type in args]
    proposition = Proposition(name, variables)
    if proposition not in facts:
        facts.append(proposition)


def apply_novelty_runtime(state: GameState, command: str, novelty_scenario: Optional[str]) -> GameState:
    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario != "magic_box":
        return state

    command = command.strip().lower()
    if command != "open magic box":
        return state

    facts = list(state["facts"] or [])
    if _has_fact(facts, "open", "magic box") and _has_fact(facts, "in", "apple", "magic box"):
        _add_fact(facts, "golden", ("apple", "f"))
        _add_fact(facts, "transformed", ("apple", "f"))
        state["facts"] = facts

    return state


def novelty_goal_facts(stage: str, novelty_scenario: Optional[str]) -> tuple[str, ...]:
    if stage != "stage_5":
        return ()

    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario == "portal":
        return ("at(P, garden)", "in(apple, I)")

    return ("golden(apple)", "transformed(apple)")


def apply_custom_goal(state: GameState, stage: str, novelty_scenario: Optional[str]) -> GameState:
    goal_facts = novelty_goal_facts(stage, novelty_scenario)
    if not goal_facts:
        return state

    fact_strings = {fact_to_str(fact) for fact in state["facts"] or []}
    custom_won = all(goal in fact_strings for goal in goal_facts)
    state["won"] = bool(custom_won)
    state["done"] = bool(state.get("done", False) or custom_won)
    if "max_score" in state:
        state["score"] = state["max_score"] if custom_won else 0
    return state


def novelty_metadata(novelty_scenario: Optional[str]) -> dict:
    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario == "portal":
        return {
            "kind": "portal_transition",
            "description": "Using the portal teleports the player to a non-adjacent room.",
        }

    return {
        "kind": "magic_box_transform",
        "description": "Opening the magic box transforms its apple into a golden apple.",
    }


def novelty_rules_summary(novelty_scenario: Optional[str]) -> tuple[str, ...]:
    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario == "portal":
        return ("use portal => move player to linked room",)

    return ("open magic box with apple inside => add golden(apple) and transformed(apple)",)
