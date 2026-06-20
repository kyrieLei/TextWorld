import tempfile
from os.path import join as pjoin

import textworld

from textworld.core import EnvInfos
from textworld.mvw.curriculum import STAGE_ORDER
from textworld.mvw.curriculum import build_stage_game
from textworld.mvw.runner import build_curriculum
from textworld.mvw.runner import evaluate_game


def test_curriculum_builds_and_saves():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = build_curriculum(tmpdir, seed=2026)
        assert len(paths) == len(STAGE_ORDER)
        for path in paths:
            env = textworld.start(path, request_infos=EnvInfos(facts=True, policy_commands=True))
            state = env.reset()
            assert state.facts
            env.close()


def test_stage3_walkthrough_has_no_unresolved_novelty():
    report = evaluate_game("stage_3", known_stage="stage_3", seed=2026)
    assert report["won"]
    assert report["novelty_steps"] == 0
    assert report["unresolved_novelty_steps"] == 0


def test_stage5_portal_needs_expansion_then_recovers():
    without_patch = evaluate_game("stage_5", known_stage="stage_4", seed=2026, expand=False)
    with_patch = evaluate_game("stage_5", known_stage="stage_4", seed=2026, expand=True)

    assert without_patch["novelty_steps"] > 0
    assert without_patch["unresolved_novelty_steps"] > 0
    assert with_patch["won"]
    assert with_patch["unresolved_novelty_steps"] == 0


def test_stage5_magic_box_needs_expansion_then_recovers():
    without_patch = evaluate_game("stage_5", known_stage="stage_4", seed=2026, expand=False, novelty_scenario="magic_box")
    with_patch = evaluate_game("stage_5", known_stage="stage_4", seed=2026, expand=True, novelty_scenario="magic_box")

    assert without_patch["novelty_steps"] > 0
    assert without_patch["won"] is False
    assert with_patch["won"]
    assert with_patch["unresolved_novelty_steps"] == 0
