# Minimum Viable World

This package implements a staged TextWorld benchmark for conflict-driven world-model expansion.

Stages:

1. `stage_0`: rooms and navigation
2. `stage_1`: portable objects and inventory
3. `stage_2`: containers and visibility/preconditions
4. `stage_3`: locked doors and keys
5. `stage_4`: cooking and cutting
6. `stage_5`: portal novelty that requires a new transition rule

Main entrypoints:

```bash
./.venv/bin/python scripts/tw-mvw build --output-dir /tmp/tw-mvw
./.venv/bin/python scripts/tw-mvw evaluate --stage stage_4 --known-stage stage_4
./.venv/bin/python scripts/tw-mvw evaluate --stage stage_5 --known-stage stage_4 --expand
./.venv/bin/python scripts/tw-mvw retention --known-stage stage_4
```

`textworld.mvw.llm.OpenAICompatibleHypothesisProposer` is optional. It can be pointed at any OpenAI-compatible serving stack, including a locally served Qwen3 Instruct model.

Environment variables for the optional proposer:

```bash
export TW_MVW_LLM_BASE_URL=http://127.0.0.1:8000/v1
export TW_MVW_LLM_API_KEY=...
```
