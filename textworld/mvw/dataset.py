from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import textworld
from textworld.core import EnvInfos
from textworld.mvw.curriculum import STAGE_ORDER
from textworld.mvw.curriculum import build_stage_game
from textworld.mvw.curriculum import normalize_stage
from textworld.mvw.models import fact_to_str


@dataclass(frozen=True)
class StepRecord:
    stage: str
    game_seed: int
    step_index: int
    observation: str
    command: str
    next_observation: str
    facts: tuple[str, ...]
    next_facts: tuple[str, ...]
    admissible_commands: tuple[str, ...]
    policy_commands: tuple[str, ...]
    won: bool
    done: bool

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "game_seed": self.game_seed,
            "step_index": self.step_index,
            "observation": self.observation,
            "command": self.command,
            "next_observation": self.next_observation,
            "facts": list(self.facts),
            "next_facts": list(self.next_facts),
            "admissible_commands": list(self.admissible_commands),
            "policy_commands": list(self.policy_commands),
            "won": self.won,
            "done": self.done,
        }


def _request_infos() -> EnvInfos:
    return EnvInfos(facts=True, admissible_commands=True, policy_commands=True, intermediate_reward=True, feedback=True)


def _save_temp_game(game) -> str:
    tmpdir = tempfile.mkdtemp(prefix="tw-mvw-data-")
    path = pjoin(tmpdir, "{}.json".format(game.metadata["stage"]))
    game.save(path)
    return path


def _serialize_facts(facts) -> tuple[str, ...]:
    return tuple(sorted(fact_to_str(fact) for fact in facts))


def collect_stage_dataset(
    stage: Union[int, str],
    num_games: int = 16,
    seed: int = 1234,
    policy_mix: float = 0.7,
    max_steps: Optional[int] = None,
) -> list[StepRecord]:
    stage_id = normalize_stage(stage)
    rng = random.Random(seed)
    dataset: list[StepRecord] = []

    for game_index in range(num_games):
        game_seed = seed + game_index
        game = build_stage_game(stage_id, seed=game_seed)
        env = textworld.start(_save_temp_game(game), request_infos=_request_infos())
        state = env.reset()
        steps_budget = max_steps or max(8, 3 * max(1, len(game.metadata.get("walkthrough", []))))

        for step_index in range(steps_budget):
            admissible = list(state.admissible_commands or [])
            if not admissible:
                break

            policy = list(state.policy_commands or [])
            if policy and rng.random() < policy_mix:
                command = policy[0]
            else:
                command = rng.choice(admissible)

            next_state, _, done = env.step(command)
            dataset.append(
                StepRecord(
                    stage=stage_id,
                    game_seed=game_seed,
                    step_index=step_index,
                    observation=state.feedback or "",
                    command=command,
                    next_observation=next_state.feedback or "",
                    facts=_serialize_facts(state.facts),
                    next_facts=_serialize_facts(next_state.facts),
                    admissible_commands=tuple(admissible),
                    policy_commands=tuple(policy),
                    won=bool(next_state.won),
                    done=bool(done),
                )
            )
            state = next_state
            if done:
                break

        env.close()

    return dataset


def collect_curriculum_dataset(
    stages: Optional[Sequence[Union[int, str]]] = None,
    num_games_per_stage: int = 16,
    seed: int = 1234,
    policy_mix: float = 0.7,
) -> list[StepRecord]:
    stages = stages or STAGE_ORDER
    dataset: list[StepRecord] = []
    for index, stage in enumerate(stages):
        dataset.extend(
            collect_stage_dataset(
                stage=stage,
                num_games=num_games_per_stage,
                seed=seed + 100 * index,
                policy_mix=policy_mix,
            )
        )

    return dataset


def save_dataset(records: Iterable[StepRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def load_dataset(path: str) -> list[StepRecord]:
    records: list[StepRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            records.append(
                StepRecord(
                    stage=data["stage"],
                    game_seed=data["game_seed"],
                    step_index=data["step_index"],
                    observation=data["observation"],
                    command=data["command"],
                    next_observation=data["next_observation"],
                    facts=tuple(data["facts"]),
                    next_facts=tuple(data["next_facts"]),
                    admissible_commands=tuple(data["admissible_commands"]),
                    policy_commands=tuple(data["policy_commands"]),
                    won=bool(data["won"]),
                    done=bool(data["done"]),
                )
            )
    return records
