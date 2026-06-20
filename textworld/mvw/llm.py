from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

from textworld.mvw.models import NoveltySignal


@dataclass
class OpenAICompatibleHypothesisProposer:
    model: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 60

    @classmethod
    def from_env(cls, model: str = "Qwen/Qwen3-0.6B") -> "OpenAICompatibleHypothesisProposer":
        base_url = os.environ["TW_MVW_LLM_BASE_URL"]
        api_key = os.environ.get("TW_MVW_LLM_API_KEY")
        return cls(model=model, base_url=base_url, api_key=api_key)

    @classmethod
    def from_config(cls, base_url: str, model: str = "Qwen/Qwen3-0.6B", api_key: Optional[str] = None) -> "OpenAICompatibleHypothesisProposer":
        return cls(model=model, base_url=base_url, api_key=api_key)

    def propose(self, signal: NoveltySignal) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {}".format(self.api_key)

        prompt = (
            "You are proposing the smallest world-model patch that explains a novelty.\n"
            "Command: {command}\n"
            "Unsupported action: {unsupported}\n"
            "Missing facts: {missing}\n"
            "Unexpected facts: {unexpected}\n"
            "Return one concise rule patch."
        ).format(
            command=signal.command,
            unsupported=signal.unsupported_action,
            missing=", ".join(signal.missing_facts) or "none",
            unexpected=", ".join(signal.unexpected_facts) or "none",
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Propose minimal symbolic world-model patches."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
