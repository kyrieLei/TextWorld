import glob
from os.path import dirname
from os.path import join as pjoin
from typing import Iterable

from textworld.logic import GameLogic
from textworld.generator.data import KnowledgeBase
from textworld.generator.data import LOGIC_DATA_PATH
from textworld.generator.data import TEXT_GRAMMARS_PATH


def _load_logic(paths: Iterable[str]) -> GameLogic:
    return GameLogic.load(sorted(paths))


def load_augmented_kb(extra_logic_files: Iterable[str], grammar_path: str = TEXT_GRAMMARS_PATH) -> KnowledgeBase:
    builtin_logic = glob.glob(pjoin(LOGIC_DATA_PATH, "*.twl"))
    logic = _load_logic(list(builtin_logic) + list(extra_logic_files))
    kb = KnowledgeBase(logic, grammar_path)
    kb.logic_path = "augmented"
    return kb


def load_portal_kb() -> KnowledgeBase:
    extra_logic = pjoin(dirname(__file__), "data", "logic", "portal.twl")
    return load_augmented_kb([extra_logic])
