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
from textworld.mvw.kb import load_portal_kb


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
        title="Portal Novelty",
        summary="Introduces a new transition primitive that must be explained by a minimal rule patch.",
        new_entities=("portal",),
        new_relations=("portal_link",),
        new_actions=("use portal",),
        new_constraints=("portals create explicit non-local transitions",),
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


def _build_stage_5(seed: int) -> Game:
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
    game.metadata["novelty"] = {
        "kind": "portal_transition",
        "description": "Entering the portal teleports the player to a non-adjacent room.",
    }
    return game


def build_stage_game(stage: Union[int, str], seed: int = 1234) -> Game:
    stage_id = normalize_stage(stage)
    builders = {
        "stage_0": _build_stage_0,
        "stage_1": _build_stage_1,
        "stage_2": _build_stage_2,
        "stage_3": _build_stage_3,
        "stage_4": _build_stage_4,
        "stage_5": _build_stage_5,
    }
    return builders[stage_id](seed)
