from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

from textworld.generator.game import Game
from textworld.logic import Proposition
from textworld.logic import Variable


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().strip().split())


def fact_to_str(fact: Proposition) -> str:
    args = ", ".join(arg.name for arg in fact.arguments)
    return "{}({})".format(fact.name, args) if args else "{}()".format(fact.name)


def _sorted_facts(facts: Iterable[Proposition]) -> tuple[Proposition, ...]:
    return tuple(sorted(set(facts), key=fact_to_str))


@dataclass(frozen=True)
class BeliefState:
    facts: tuple[Proposition, ...]

    @classmethod
    def from_facts(cls, facts: Iterable[Proposition]) -> "BeliefState":
        return cls(_sorted_facts(facts))

    def as_set(self) -> set[Proposition]:
        return set(self.facts)


@dataclass(frozen=True)
class ConsistencyViolation:
    code: str
    message: str


@dataclass(frozen=True)
class NoveltySignal:
    command: str
    unsupported_action: bool
    missing_facts: tuple[str, ...]
    unexpected_facts: tuple[str, ...]
    predicted_violations: tuple[str, ...]

    @property
    def is_novel(self) -> bool:
        return self.unsupported_action or bool(self.missing_facts) or bool(self.unexpected_facts) or bool(self.predicted_violations)


@dataclass(frozen=True)
class WorldPatch:
    name: str
    kind: str
    description: str
    rules: tuple[str, ...]
    new_entities: tuple[str, ...] = ()
    new_properties: tuple[str, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.new_entities) + len(self.new_properties) + len(self.rules)


class WorldContext:
    def __init__(self, game: Game):
        self.game = game
        self.variables: Dict[str, Variable] = {}
        self.name_to_ids: Dict[str, List[str]] = defaultdict(list)
        self.id_to_name: Dict[str, str] = {}
        self.id_to_type: Dict[str, str] = {}

        for fact in game.world.facts:
            for arg in fact.arguments:
                self.variables[arg.name] = arg

        for entity_id, infos in game.infos.items():
            name = infos.name or entity_id
            self.id_to_name[entity_id] = name
            self.id_to_type[entity_id] = infos.type
            if name not in self.variables:
                self.variables[name] = Variable(name, infos.type)
                self.id_to_type[name] = infos.type

            self.name_to_ids[_normalize_name(name)].append(name)
            self.name_to_ids[_normalize_name(name)].append(entity_id)
            self.name_to_ids[_normalize_name(entity_id)].append(entity_id)
            self.name_to_ids[_normalize_name(entity_id)].append(name)

    def resolve(self, name: str) -> Optional[str]:
        ids = self.name_to_ids.get(_normalize_name(name), [])
        return ids[0] if ids else None

    def variable(self, entity_id: str) -> Variable:
        if entity_id not in self.variables:
            entity_type = self.id_to_type.get(entity_id, entity_id if entity_id in {"P", "I"} else "t")
            self.variables[entity_id] = Variable(entity_id, entity_type)
        return self.variables[entity_id]


class OracleStateTracker:
    def observe(self, facts: Iterable[Proposition]) -> BeliefState:
        return BeliefState.from_facts(facts)


class ConsistencyVerifier:
    _location_predicates = frozenset({"at", "in", "on"})

    def check(self, state: BeliefState) -> list[ConsistencyViolation]:
        violations: list[ConsistencyViolation] = []
        by_entity: Dict[str, list[Proposition]] = defaultdict(list)
        properties: Dict[str, set[str]] = defaultdict(set)

        for fact in state.facts:
            if fact.name in self._location_predicates and fact.arguments:
                by_entity[fact.arguments[0].name].append(fact)

            if len(fact.arguments) == 1:
                properties[fact.arguments[0].name].add(fact.name)

        player_rooms = [fact for fact in by_entity.get("P", []) if fact.name == "at"]
        if len(player_rooms) != 1:
            violations.append(ConsistencyViolation("player_location", "player must be in exactly one room"))

        for entity, facts in by_entity.items():
            if len(facts) > 1:
                violations.append(ConsistencyViolation("multi_location", "{} has multiple parent locations".format(entity)))

        for entity, props in properties.items():
            if len({"open", "closed", "locked"} & props) > 1:
                violations.append(ConsistencyViolation("door_or_container_state", "{} has incompatible open/closed/locked states".format(entity)))

            if "raw" in props and {"fried", "roasted", "grilled", "cooked"} & props:
                violations.append(ConsistencyViolation("raw_vs_cooked", "{} cannot be raw and cooked".format(entity)))

            if len({"uncut", "chopped", "sliced", "diced"} & props) > 1:
                violations.append(ConsistencyViolation("cut_state", "{} has incompatible cutting states".format(entity)))

        links: Dict[str, set[str]] = defaultdict(set)
        blocked_doors: set[str] = set()
        for fact in state.facts:
            if fact.name == "link":
                links[fact.arguments[1].name].add((fact.arguments[0].name, fact.arguments[2].name))
            elif fact.name in {"closed", "locked"} and fact.arguments:
                blocked_doors.add(fact.arguments[0].name)

        for fact in state.facts:
            if fact.name == "free":
                room_pair = (fact.arguments[0].name, fact.arguments[1].name)
                for door, door_links in links.items():
                    if room_pair in door_links and door in blocked_doors:
                        violations.append(ConsistencyViolation("free_blocked_door", "{} is blocked but rooms remain free".format(door)))

        return violations


class SymbolicTransitionModel:
    def __init__(self, context: WorldContext, known_stage: int, patches: Sequence[WorldPatch] = ()) -> None:
        self.context = context
        self.known_stage = known_stage
        self.patches = tuple(patches)

    def with_patch(self, patch: WorldPatch) -> "SymbolicTransitionModel":
        return SymbolicTransitionModel(self.context, self.known_stage, self.patches + (patch,))

    def predict(self, state: BeliefState, command: str) -> tuple[BeliefState, bool]:
        supported = True
        cmd = _normalize_name(command)
        facts = state.as_set()

        if cmd in {"look", "inventory"} or cmd.startswith("examine "):
            return BeliefState.from_facts(facts), supported

        if cmd.startswith("go "):
            if self.known_stage < 0:
                supported = False
            else:
                self._apply_go(facts, cmd.split(" ", 1)[1])

        elif cmd.startswith("take "):
            if self.known_stage < 1:
                supported = False
            else:
                self._apply_take(facts, cmd)

        elif cmd.startswith("drop "):
            if self.known_stage < 1:
                supported = False
            else:
                self._apply_drop(facts, cmd.split(" ", 1)[1])

        elif cmd.startswith("open "):
            if self.known_stage < 2:
                supported = False
            else:
                self._apply_open(facts, cmd.split(" ", 1)[1])
                self._apply_transform_on_open(facts, cmd.split(" ", 1)[1])

        elif cmd.startswith("close "):
            if self.known_stage < 2:
                supported = False
            else:
                self._apply_close(facts, cmd.split(" ", 1)[1])

        elif cmd.startswith("insert "):
            if self.known_stage < 2:
                supported = False
            else:
                self._apply_insert(facts, cmd)

        elif cmd.startswith("unlock "):
            if self.known_stage < 3:
                supported = False
            else:
                self._apply_unlock(facts, cmd)

        elif cmd == "prepare meal":
            if self.known_stage < 4:
                supported = False
            else:
                self._apply_prepare_meal(facts)

        elif cmd == "eat meal":
            if self.known_stage < 4:
                supported = False
            else:
                self._apply_eat_meal(facts)

        elif cmd.startswith(("slice ", "dice ", "chop ", "cook ")):
            if self.known_stage < 4:
                supported = False
            else:
                self._apply_food_transform(facts, cmd)

        elif cmd.startswith("use "):
            if any(patch.kind == "portal_transition" for patch in self.patches):
                self._apply_use_portal(facts, cmd.split(" ", 1)[1])
            else:
                supported = False
        else:
            supported = False

        return BeliefState.from_facts(facts), supported

    def _current_room(self, facts: set[Proposition]) -> Optional[str]:
        for fact in facts:
            if fact.name == "at" and fact.arguments[0].name == "P":
                return fact.arguments[1].name

        return None

    def _has_fact(self, facts: set[Proposition], name: str, *args: str) -> bool:
        return Proposition(name, [self.context.variable(arg) for arg in args]) in facts

    def _remove_fact(self, facts: set[Proposition], name: str, *args: str) -> None:
        facts.discard(Proposition(name, [self.context.variable(arg) for arg in args]))

    def _add_fact(self, facts: set[Proposition], name: str, *args: str) -> None:
        facts.add(Proposition(name, [self.context.variable(arg) for arg in args]))

    def _replace_player_room(self, facts: set[Proposition], room_id: str) -> None:
        for fact in list(facts):
            if fact.name == "at" and fact.arguments[0].name == "P":
                facts.remove(fact)
        self._add_fact(facts, "at", "P", room_id)

    def _remove_object_location(self, facts: set[Proposition], entity_id: str) -> None:
        for fact in list(facts):
            if fact.name in {"at", "in", "on"} and fact.arguments and fact.arguments[0].name == entity_id:
                facts.remove(fact)

    def _apply_go(self, facts: set[Proposition], direction: str) -> None:
        room_id = self._current_room(facts)
        if room_id is None:
            return

        target = None
        predicate = "{}_of".format(direction)
        for fact in facts:
            if fact.name == predicate and fact.arguments[1].name == room_id:
                candidate = fact.arguments[0].name
                if self._has_fact(facts, "free", room_id, candidate) and self._has_fact(facts, "free", candidate, room_id):
                    target = candidate
                    break

        if target is not None:
            self._replace_player_room(facts, target)

    def _apply_take(self, facts: set[Proposition], command: str) -> None:
        room_id = self._current_room(facts)
        if room_id is None:
            return

        match = re.fullmatch(r"take (.+?) from (.+)", command)
        if match:
            obj_name, holder_name = match.groups()
            holder_id = self.context.resolve(holder_name)
        else:
            obj_name = command.split(" ", 1)[1]
            holder_id = None

        obj_id = self.context.resolve(obj_name)
        if obj_id is None:
            return

        for fact in list(facts):
            if fact.name not in {"at", "in", "on"} or fact.arguments[0].name != obj_id:
                continue

            parent = fact.arguments[1].name
            if holder_id is not None and parent != holder_id:
                continue

            if fact.name == "at" and parent != room_id:
                continue

            if fact.name in {"in", "on"} and not self._container_or_supporter_accessible(facts, parent, room_id):
                continue

            facts.remove(fact)
            self._add_fact(facts, "in", obj_id, "I")
            break

    def _container_or_supporter_accessible(self, facts: set[Proposition], holder_id: str, room_id: str) -> bool:
        if self._has_fact(facts, "at", holder_id, room_id):
            if self._has_fact(facts, "locked", holder_id) or self._has_fact(facts, "closed", holder_id):
                return self._has_fact(facts, "open", holder_id)
            return True

        return False

    def _apply_drop(self, facts: set[Proposition], object_name: str) -> None:
        obj_id = self.context.resolve(object_name)
        room_id = self._current_room(facts)
        if obj_id is None or room_id is None:
            return

        if self._has_fact(facts, "in", obj_id, "I"):
            self._remove_fact(facts, "in", obj_id, "I")
            self._add_fact(facts, "at", obj_id, room_id)

    def _apply_open(self, facts: set[Proposition], entity_name: str) -> None:
        entity_id = self.context.resolve(entity_name)
        if entity_id is None or self._has_fact(facts, "locked", entity_id):
            return

        if self._has_fact(facts, "closed", entity_id):
            self._remove_fact(facts, "closed", entity_id)
            self._add_fact(facts, "open", entity_id)
            self._set_free_if_door(facts, entity_id, free=True)

    def _apply_close(self, facts: set[Proposition], entity_name: str) -> None:
        entity_id = self.context.resolve(entity_name)
        if entity_id is None:
            return

        if self._has_fact(facts, "open", entity_id):
            self._remove_fact(facts, "open", entity_id)
            self._add_fact(facts, "closed", entity_id)
            self._set_free_if_door(facts, entity_id, free=False)

    def _set_free_if_door(self, facts: set[Proposition], door_id: str, free: bool) -> None:
        links = [fact for fact in facts if fact.name == "link" and fact.arguments[1].name == door_id]
        for link in links:
            src = link.arguments[0].name
            dst = link.arguments[2].name
            if free:
                self._add_fact(facts, "free", src, dst)
            else:
                self._remove_fact(facts, "free", src, dst)

    def _apply_transform_on_open(self, facts: set[Proposition], entity_name: str) -> None:
        if not any(patch.kind == "transform_on_open" for patch in self.patches):
            return

        entity_id = self.context.resolve(entity_name)
        if entity_id is None or not self._has_fact(facts, "open", entity_id):
            return

        if entity_id != "magic box":
            return

        for fact in list(facts):
            if fact.name == "in" and fact.arguments[1].name == entity_id:
                obj_id = fact.arguments[0].name
                self._add_fact(facts, "golden", obj_id)
                self._add_fact(facts, "transformed", obj_id)

    def _apply_insert(self, facts: set[Proposition], command: str) -> None:
        match = re.fullmatch(r"insert (.+?) into (.+)", command)
        if not match:
            return

        obj_name, container_name = match.groups()
        obj_id = self.context.resolve(obj_name)
        container_id = self.context.resolve(container_name)
        if obj_id is None or container_id is None:
            return

        if self._has_fact(facts, "in", obj_id, "I") and self._has_fact(facts, "open", container_id):
            self._remove_fact(facts, "in", obj_id, "I")
            self._add_fact(facts, "in", obj_id, container_id)

    def _apply_unlock(self, facts: set[Proposition], command: str) -> None:
        match = re.fullmatch(r"unlock (.+?) with (.+)", command)
        if not match:
            return

        door_name, key_name = match.groups()
        door_id = self.context.resolve(door_name)
        key_id = self.context.resolve(key_name)
        if door_id is None or key_id is None:
            return

        if self._has_fact(facts, "locked", door_id) and self._has_fact(facts, "in", key_id, "I") and self._has_fact(facts, "match", key_id, door_id):
            self._remove_fact(facts, "locked", door_id)
            self._add_fact(facts, "closed", door_id)

    def _apply_food_transform(self, facts: set[Proposition], command: str) -> None:
        slice_match = re.fullmatch(r"(slice|dice|chop) (.+?) with (.+)", command)
        if slice_match:
            action, food_name, tool_name = slice_match.groups()
            food_id = self.context.resolve(food_name)
            tool_id = self.context.resolve(tool_name)
            if food_id and tool_id and self._has_fact(facts, "in", food_id, "I") and self._has_fact(facts, "in", tool_id, "I") and self._has_fact(facts, "sharp", tool_id):
                self._remove_fact(facts, "uncut", food_id)
                self._add_fact(facts, {"slice": "sliced", "dice": "diced", "chop": "chopped"}[action], food_id)
            return

        cook_match = re.fullmatch(r"cook (.+?) with (.+)", command)
        if not cook_match:
            return

        food_name, stove_name = cook_match.groups()
        food_id = self.context.resolve(food_name)
        stove_id = self.context.resolve(stove_name)
        room_id = self._current_room(facts)
        if food_id is None or stove_id is None or room_id is None:
            return

        if self._has_fact(facts, "in", food_id, "I") and self._has_fact(facts, "at", stove_id, room_id):
            self._remove_fact(facts, "raw", food_id)
            self._remove_fact(facts, "needs_cooking", food_id)
            self._remove_fact(facts, "inedible", food_id)
            for cooking_state in ("fried", "roasted", "grilled"):
                self._remove_fact(facts, cooking_state, food_id)
            self._add_fact(facts, "cooked", food_id)
            self._add_fact(facts, "edible", food_id)
            appliance_type = self.context.variable(stove_id).type
            if appliance_type == "oven":
                self._add_fact(facts, "roasted", food_id)
            elif appliance_type == "toaster":
                self._add_fact(facts, "grilled", food_id)
            else:
                self._add_fact(facts, "fried", food_id)

    def _apply_prepare_meal(self, facts: set[Proposition]) -> None:
        meal_id = self.context.resolve("meal")
        if meal_id is None:
            return

        carried_foods = []
        for fact in list(facts):
            if fact.name == "in" and fact.arguments[1].name == "I":
                entity_id = fact.arguments[0].name
                if self.context.variable(entity_id).type == "f" and entity_id != meal_id:
                    carried_foods.append(entity_id)
                    facts.remove(fact)

        if not carried_foods:
            return

        for food_id in carried_foods:
            self._add_fact(facts, "used", food_id)

        self._add_fact(facts, "in", meal_id, "I")
        self._add_fact(facts, "raw", meal_id)
        self._add_fact(facts, "edible", meal_id)

    def _apply_eat_meal(self, facts: set[Proposition]) -> None:
        meal_id = self.context.resolve("meal")
        if meal_id is None or not self._has_fact(facts, "in", meal_id, "I"):
            return

        self._remove_fact(facts, "in", meal_id, "I")
        self._remove_fact(facts, "edible", meal_id)
        self._add_fact(facts, "consumed", meal_id)

    def _apply_use_portal(self, facts: set[Proposition], portal_name: str) -> None:
        portal_id = self.context.resolve(portal_name)
        room_id = self._current_room(facts)
        if portal_id is None or room_id is None:
            return

        for fact in facts:
            if fact.name == "portal_link" and fact.arguments[0].name == room_id and fact.arguments[1].name == portal_id:
                self._replace_player_room(facts, fact.arguments[2].name)
                break


class NoveltyDetector:
    def detect(
        self,
        command: str,
        predicted: BeliefState,
        observed: BeliefState,
        supported: bool,
        predicted_violations: Sequence[ConsistencyViolation],
    ) -> NoveltySignal:
        predicted_set = {fact_to_str(fact) for fact in predicted.facts}
        observed_set = {fact_to_str(fact) for fact in observed.facts}
        missing = tuple(sorted(observed_set - predicted_set))
        unexpected = tuple(sorted(predicted_set - observed_set))
        return NoveltySignal(
            command=command,
            unsupported_action=not supported,
            missing_facts=missing,
            unexpected_facts=unexpected,
            predicted_violations=tuple(violation.code for violation in predicted_violations),
        )


class RuleBasedExpansionPlanner:
    def propose(self, signal: NoveltySignal) -> Optional[WorldPatch]:
        command = _normalize_name(signal.command)
        if command.startswith("use ") and "portal" in command:
            return WorldPatch(
                name="portal-transition",
                kind="portal_transition",
                description="Using a portal moves the player along a portal_link edge.",
                rules=("use portal => move player to linked room",),
                new_entities=("portal",),
            )

        if command.startswith("open ") and any("gold" in fact for fact in signal.missing_facts):
            return WorldPatch(
                name="transforming-container",
                kind="transform_on_open",
                description="Opening this container triggers an object transformation.",
                rules=("open container => transform contained object",),
                new_properties=("transform_on_open",),
            )

        return None
