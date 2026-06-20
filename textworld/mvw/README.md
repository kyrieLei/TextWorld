# Minimum Viable World

This package implements a staged TextWorld benchmark for conflict-driven world-model expansion.

Stages:

1. `stage_0`: rooms and navigation
2. `stage_1`: portable objects and inventory
3. `stage_2`: containers and visibility/preconditions
4. `stage_3`: locked doors and keys
5. `stage_4`: cooking and cutting
6. `stage_5`: novelty scenarios such as `portal` and `magic_box`

Main entrypoints:

```bash
./.venv/bin/python scripts/tw-mvw build --output-dir /tmp/tw-mvw
./.venv/bin/python scripts/tw-mvw collect --output /tmp/tw-mvw-stage0.jsonl --stage stage_0 --num-games 8
./.venv/bin/python scripts/tw-mvw train --dataset /tmp/tw-mvw-stage0.jsonl
./.venv/bin/python scripts/tw-mvw evaluate --stage stage_4 --known-stage stage_4
./.venv/bin/python scripts/tw-mvw evaluate --stage stage_5 --known-stage stage_4 --expand
./.venv/bin/python scripts/tw-mvw counterfactual --stage stage_5 --known-stage stage_4 --expand
./.venv/bin/python scripts/tw-mvw benchmark --known-stage stage_4
./.venv/bin/python scripts/tw-mvw benchmark --known-stage stage_4 --novelty-scenario magic_box
./.venv/bin/python scripts/tw-mvw transfer --stage stage_5 --known-stage stage_4 --novelty-scenario magic_box --discovery-seed 2026 --eval-seed 2027
./.venv/bin/python scripts/tw-mvw retention --known-stage stage_4
```

Learning pipeline:

- `collect`: rollout data with `observation / command / facts / next_facts`
- `train`: train a learned belief tracker and a learned transition model with NumPy-only multilabel logistic baselines
- `evaluate`: symbolic baseline plus optional novelty expansion
- `counterfactual`: explicit counterfactual probes such as "open locked door" or "use portal"
- `benchmark`: the full metric suite from `idea.md`
- `transfer`: discover a novelty patch on one game seed and reuse it on a held-out seed
- `retention`: old-world retention audit across stages

`benchmark` reports:

- old-world retention
- novelty accommodation
- consistency violation rate
- rule minimality
- counterfactual accuracy
- planning improvement

For `stage_5`, you can switch novelty families with `--novelty-scenario`:

- `portal`: non-local transition edge
- `magic_box`: opening the box adds `golden(x)` and `transformed(x)` for the seeded object inside the box

`textworld.mvw.llm.OpenAICompatibleHypothesisProposer` is optional. It can be pointed at any OpenAI-compatible serving stack, including a locally served Qwen3 Instruct model.

Environment variables for the optional proposer:

```bash
export TW_MVW_LLM_BASE_URL=http://127.0.0.1:8000/v1
export TW_MVW_LLM_API_KEY=...
```

Or directly:

```bash
./.venv/bin/python scripts/tw-mvw evaluate \
  --stage stage_5 \
  --known-stage stage_4 \
  --expand \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen3-8B-Instruct
```
