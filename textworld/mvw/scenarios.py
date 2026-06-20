from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textworld.core import GameState
from textworld.mvw.models import fact_to_str


NOVELTY_SCENARIOS = ("portal", "magic_box", "bridge_button")
MAGIC_BOX_OBJECTS = ("apple", "carrot", "potato")
BRIDGE_BUTTON_NAMES = ("silver button", "amber button", "jade button")
BRIDGE_DOOR_NAMES = ("bridge door", "canal gate", "vault hatch")
BRIDGE_OBJECTS = ("apple", "coin", "orb")


@dataclass(frozen=True)
class ScenarioBindings:
    target_object: Optional[str] = None
    container_name: Optional[str] = None
    trigger_name: Optional[str] = None
    target_room: Optional[str] = None
    target_door: Optional[str] = None


def normalize_novelty_scenario(scenario: Optional[str]) -> str:
    scenario = (scenario or "portal").strip().lower()
    if scenario not in NOVELTY_SCENARIOS:
        raise KeyError("Unknown novelty scenario: {}".format(scenario))
    return scenario


def scenario_bindings(metadata: Optional[dict]) -> ScenarioBindings:
    metadata = metadata or {}
    return ScenarioBindings(
        target_object=metadata.get("target_object"),
        container_name=metadata.get("container_name"),
        trigger_name=metadata.get("trigger_name"),
        target_room=metadata.get("target_room"),
        target_door=metadata.get("target_door"),
    )


def pick_magic_box_object(seed: int) -> str:
    return MAGIC_BOX_OBJECTS[seed % len(MAGIC_BOX_OBJECTS)]


def pick_bridge_button_name(seed: int) -> str:
    return BRIDGE_BUTTON_NAMES[seed % len(BRIDGE_BUTTON_NAMES)]


def pick_bridge_door_name(seed: int) -> str:
    return BRIDGE_DOOR_NAMES[seed % len(BRIDGE_DOOR_NAMES)]


def pick_bridge_object(seed: int) -> str:
    return BRIDGE_OBJECTS[seed % len(BRIDGE_OBJECTS)]


def apply_novelty_runtime(state: GameState, command: str, novelty_scenario: Optional[str], metadata: Optional[dict] = None) -> GameState:
    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario not in {"magic_box", "bridge_button"}:
        return state

    # magic_box and bridge_button are both produced natively by stage-specific KB rules.
    return state


def novelty_goal_facts(stage: str, novelty_scenario: Optional[str], metadata: Optional[dict] = None) -> tuple[str, ...]:
    if stage != "stage_5":
        return ()

    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario == "portal":
        return ("at(P, garden)", "in(apple, I)")
    if scenario == "bridge_button":
        bindings = scenario_bindings(metadata)
        return (
            "at(P, {})".format(bindings.target_room or "vault"),
            "in({}, I)".format(bindings.target_object or "apple"),
        )

    bindings = scenario_bindings(metadata)
    target_object = bindings.target_object or "apple"
    return (
        "golden({})".format(target_object),
        "transformed({})".format(target_object),
    )


def apply_custom_goal(state: GameState, stage: str, novelty_scenario: Optional[str], metadata: Optional[dict] = None) -> GameState:
    metadata = metadata or {}
    goal_facts = tuple(metadata.get("custom_goal_facts", ())) or novelty_goal_facts(stage, novelty_scenario, metadata)
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
    if scenario == "bridge_button":
        return {
            "kind": "switch_open_path",
            "description": "Pushing the button opens a linked door and frees a blocked path.",
        }

    return {
        "kind": "magic_box_transform",
        "description": "Opening the magic box transforms its apple into a golden apple.",
    }


def novelty_rules_summary(novelty_scenario: Optional[str]) -> tuple[str, ...]:
    scenario = normalize_novelty_scenario(novelty_scenario)
    if scenario == "portal":
        return ("use portal => move player to linked room",)
    if scenario == "bridge_button":
        return ("push button => open linked door and free path",)

    return ("open magic box with object inside => add golden(x) and transformed(x)",)
