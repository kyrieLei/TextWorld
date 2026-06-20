from dataclasses import asdict
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Dict
from typing import Iterable
from typing import Union

import textworld

from textworld.challenges.tw_cooking import cooking
from textworld.generator.data import KnowledgeBase
from textworld.generator.game import Game
from textworld.generator.game import GameOptions
from textworld.generator.game import Event
from textworld.generator.game import Quest
from textworld.mvw.kb import load_bridge_button_kb
from textworld.mvw.kb import load_magic_box_kb
from textworld.mvw.kb import load_portal_kb
from textworld.mvw.scenarios import pick_bridge_button_name
from textworld.mvw.scenarios import pick_bridge_door_name
from textworld.mvw.scenarios import pick_bridge_object
from textworld.mvw.scenarios import normalize_novelty_scenario
from textworld.mvw.scenarios import novelty_goal_facts
from textworld.mvw.scenarios import novelty_metadata
from textworld.mvw.scenarios import pick_magic_box_object


@dataclass(frozen=True)
class StageSpec:
    id: str
    index: int
    title: str
    summary: str
    new_entities: tuple[str, ...]
    new_relations: tuple[str, ...]
    new_actions: tuple[str, ...]
    new_constraints: tuple[str, ...]
    novelty: bool = False


STAGE_ORDER = tuple(f"stage_{idx}" for idx in range(6))

STAGE_SPECS: Dict[str, StageSpec] = {
    "stage_0": StageSpec(
        id="stage_0",
        index=0,
        title="Rooms And Navigation",
        summary="A minimal closed world with rooms, map edges, and player position tracking.",
        new_entities=("room", "player"),
        new_relations=("at", "east_of", "west_of", "north_of", "south_of", "free"),
        new_actions=("go north", "go south", "go east", "go west"),
        new_constraints=("player can only be in one room", "navigation must follow free map edges"),
    ),
    "stage_1": StageSpec(
        id="stage_1",
        index=1,
        title="Portable Objects",
        summary="Adds inventory and object relocation without breaking navigation.",
        new_entities=("object", "inventory"),
        new_relations=("in",),
        new_actions=("take", "drop"),
        new_constraints=("objects cannot be both in inventory and in a room",),
    ),
    "stage_2": StageSpec(
        id="stage_2",
        index=2,
        title="Containers",
        summary="Introduces hidden state through containers and open/close preconditions.",
        new_entities=("container",),
        new_relations=("in",),
        new_actions=("open", "close", "take from container", "insert into container"),
        new_constraints=("containers cannot be open and closed simultaneously", "objects have a single parent location"),
    ),
    "stage_3": StageSpec(
        id="stage_3",
        index=3,
        title="Locked Doors And Keys",
        summary="Adds affordances, access control, and short multi-step dependency chains.",
        new_entities=("door", "key"),
        new_relations=("link", "match"),
        new_actions=("unlock",),
        new_constraints=("locked doors are not free", "unlock requires the matching key"),
    ),
    "stage_4": StageSpec(
        id="stage_4",
        index=4,
        title="Cooking And Cutting",
        summary="Adds compositional state transformations for food processing.",
        new_entities=("food", "stove", "tool"),
        new_relations=("fried", "sliced", "raw", "uncut"),
        new_actions=("slice", "dice", "chop", "cook"),
        new_constraints=("food can only have one cutting state", "raw food cannot stay raw after cooking"),
    ),
    "stage_5": StageSpec(
        id="stage_5",
        index=5,
        title="Novelty Scenarios",
        summary="Introduces exception cases such as portals, transforming containers, and control actions that require a minimal rule patch.",
        new_entities=("portal", "magic_box", "button"),
        new_relations=("portal_link", "golden", "transformed", "bridge_target"),
        new_actions=("use portal", "open magic box", "push button"),
        new_constraints=("novel transitions or transforms must be encoded explicitly",),
        novelty=True,
    ),
}


def normalize_stage(stage: Union[int, str]) -> str:
    if isinstance(stage, int):
        key = f"stage_{stage}"
    else:
        key = stage.lower().strip().replace("-", "_")

    if key not in STAGE_SPECS:
        raise KeyError("Unknown stage: {}".format(stage))

    return key


def _options(seed: int, kb: KnowledgeBase = None) -> GameOptions:
    options = textworld.GameOptions()
    options.seeds = seed
    if kb is not None:
        options.kb = kb
    return options


def _finalize_stage(game: Game, stage_id: str, walkthrough: Iterable[str]) -> Game:
    spec = STAGE_SPECS[stage_id]
    game.metadata["stage"] = stage_id
    game.metadata["stage_spec"] = asdict(spec)
    game.metadata["walkthrough"] = list(walkthrough)
    return game


def _build_stage_0(seed: int) -> Game:
    options = _options(seed)
    M = textworld.GameMaker(options)
    hall = M.new_room("hall")
    kitchen = M.new_room("kitchen")
    bedroom = M.new_room("bedroom")
    M.connect(hall.east, kitchen.west)
    M.connect(hall.north, bedroom.south)
    M.set_player(hall)
    walkthrough = ["go east"]
    M.quests = [M.new_quest_using_commands(walkthrough)]
    M.set_walkthrough(walkthrough)
    return _finalize_stage(M.build(), "stage_0", walkthrough)


def _build_stage_1(seed: int) -> Game:
    options = _options(seed)
    M = textworld.GameMaker(options)
    pantry = M.new_room("pantry")
    M.set_player(pantry)
    apple = M.new(type="f", name="apple")
    pantry.add(apple)
    walkthrough = ["take apple"]
    M.quests = [M.new_quest_using_commands(walkthrough)]
    M.set_walkthrough(walkthrough)
    return _finalize_stage(M.build(), "stage_1", walkthrough)


def _build_stage_2(seed: int) -> Game:
    options = _options(seed)
    M = textworld.GameMaker(options)
    kitchen = M.new_room("kitchen")
    M.set_player(kitchen)
    fridge = M.new(type="c", name="fridge")
    fridge.add_property("closed")
    apple = M.new(type="f", name="apple")
    fridge.add(apple)
    kitchen.add(fridge)
    walkthrough = ["open fridge", "take apple from fridge"]
    M.quests = [M.new_quest_using_commands(walkthrough)]
    M.set_walkthrough(walkthrough)
    return _finalize_stage(M.build(), "stage_2", walkthrough)


def _build_stage_3(seed: int) -> Game:
    options = _options(seed)
    M = textworld.GameMaker(options)
    bedroom = M.new_room("bedroom")
    kitchen = M.new_room("kitchen")
    path = M.connect(bedroom.east, kitchen.west)
    path.door = M.new(type="d", name="wooden door")
    path.door.add_property("locked")
    M.set_player(bedroom)

    nightstand = M.new(type="s", name="nightstand")
    bedroom.add(nightstand)
    key = M.new(type="k", name="old key")
    nightstand.add(key)
    M.add_fact("match", key, path.door)

    apple = M.new(type="f", name="apple")
    kitchen.add(apple)

    walkthrough = [
        "take old key from nightstand",
        "unlock wooden door with old key",
        "open wooden door",
        "go east",
        "take apple",
    ]
    M.quests = [M.new_quest_using_commands(walkthrough)]
    M.set_walkthrough(walkthrough)
    return _finalize_stage(M.build(), "stage_3", walkthrough)


def _build_stage_4(seed: int) -> Game:
    options = _options(seed)
    game = cooking.make(
        {
            "recipe": 1,
            "take": 1,
            "go": 1,
            "open": False,
            "cook": True,
            "cut": True,
            "drop": False,
            "recipe_seed": seed,
            "split": "train",
        },
        options,
    )
    walkthrough = game.metadata["walkthrough"]
    return _finalize_stage(game, "stage_4", walkthrough)


def _build_stage_5_portal(seed: int) -> Game:
    options = _options(seed, load_portal_kb())
    M = textworld.GameMaker(options)
    lab = M.new_room("lab")
    garden = M.new_room("garden")
    pantry = M.new_room("pantry")
    M.connect(lab.north, pantry.south)
    M.set_player(lab)

    portal = M.new(type="portal", name="blue portal")
    lab.add(portal)
    apple = M.new(type="f", name="apple")
    garden.add(apple)
    M.add_fact("portal_link", lab, portal, garden)

    walkthrough = ["use blue portal", "take apple"]
    win_event = Event(
        conditions={
            M.new_fact("at", M.player, garden),
            M.new_fact("in", apple, M.inventory),
        }
    )
    M.quests = [Quest(win_events=[win_event], commands=walkthrough)]
    game = _finalize_stage(M.build(), "stage_5", walkthrough)
    game.metadata["novelty"] = novelty_metadata("portal")
    game.metadata["novelty_scenario"] = "portal"
    game.metadata["custom_goal_facts"] = list(novelty_goal_facts("stage_5", "portal"))
    return game


def _build_stage_5_magic_box(seed: int) -> Game:
    options = _options(seed, load_magic_box_kb())
    M = textworld.GameMaker(options)
    lab = M.new_room("lab")
    M.set_player(lab)

    magic_box = M.new(type="magic_box", name="magic box")
    magic_box.add_property("closed")
    lab.add(magic_box)

    target_object_name = pick_magic_box_object(seed)
    target_object = M.new(type="f", name=target_object_name)
    magic_box.add(target_object)

    walkthrough = ["open magic box"]
    win_event = Event(conditions={M.new_fact("open", magic_box)})
    M.quests = [Quest(win_events=[win_event], commands=walkthrough)]
    game = _finalize_stage(M.build(), "stage_5", walkthrough)
    game.metadata["novelty"] = novelty_metadata("magic_box")
    game.metadata["novelty_scenario"] = "magic_box"
    game.metadata["container_name"] = "magic box"
    game.metadata["target_object"] = target_object_name
    game.metadata["custom_goal_facts"] = list(novelty_goal_facts("stage_5", "magic_box", game.metadata))
    return game


def _build_stage_5_bridge_button(seed: int) -> Game:
    options = _options(seed, load_bridge_button_kb())
    M = textworld.GameMaker(options)
    lab = M.new_room("lab")
    vault = M.new_room("vault")
    M.set_player(lab)
    M.add_fact("east_of", vault, lab)
    M.add_fact("west_of", lab, vault)

    button_name = pick_bridge_button_name(seed)
    button = M.new(type="button", name=button_name)
    lab.add(button)
    M.add_fact("bridge_target", button, lab, vault)

    target_object_name = pick_bridge_object(seed)
    target_object = M.new(type="o", name=target_object_name)
    vault.add(target_object)

    walkthrough = ["push {}".format(button_name), "go east", "take {}".format(target_object_name)]
    win_event = Event(
        conditions={
            M.new_fact("at", M.player, vault),
            M.new_fact("in", target_object, M.inventory),
        }
    )
    M.quests = [Quest(win_events=[win_event], commands=walkthrough)]
    game = _finalize_stage(M.build(), "stage_5", walkthrough)
    game.metadata["novelty"] = novelty_metadata("bridge_button")
    game.metadata["novelty_scenario"] = "bridge_button"
    game.metadata["trigger_name"] = button_name
    game.metadata["target_room"] = "vault"
    game.metadata["target_door"] = pick_bridge_door_name(seed)
    game.metadata["target_object"] = target_object_name
    game.metadata["custom_goal_facts"] = list(novelty_goal_facts("stage_5", "bridge_button", game.metadata))
    return game


def build_stage_game(stage: Union[int, str], seed: int = 1234, novelty_scenario: str = None) -> Game:
    stage_id = normalize_stage(stage)
    if stage_id == "stage_5":
        scenario = normalize_novelty_scenario(novelty_scenario)
        if scenario == "magic_box":
            return _build_stage_5_magic_box(seed)
        if scenario == "bridge_button":
            return _build_stage_5_bridge_button(seed)
        return _build_stage_5_portal(seed)

    builders = {
        "stage_0": _build_stage_0,
        "stage_1": _build_stage_1,
        "stage_2": _build_stage_2,
        "stage_3": _build_stage_3,
        "stage_4": _build_stage_4,
    }
    return builders[stage_id](seed)
