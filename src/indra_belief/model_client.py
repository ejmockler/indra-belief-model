"""Unified model-transport client.

Supports:
- OpenAI-compatible APIs (LiteLLM → Ollama serving Gemma, Qwen, etc.)
- Anthropic API (Claude)
- Local models that emit reasoning in content (Qwen CRACK variants)
- Local models with separate reasoning_content (Gemma-4, Qwen3-thinking)

Design principles:
1. Single ModelClient interface; backend detail hidden.
2. Plain chat only. Tool-use is implemented by pre-computing the tool
   result and injecting it into the prompt (see
   `scorer._format_entity_lookups`), not by native tool-calling — the
   model ignored tool results after committing a verdict in pass one.
3. This module is pure transport — verdict parsing and score mapping
   live in `scorers/_prompts.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


# Model registry — name → (base_url, model_id, notes)
LOCAL_MODELS: dict[str, dict] = {
    "qwen-thinker": {
        "base_url": "http://localhost:8082/v1",
        "model_id": "dealignai/Qwen3.5-VL-122B-A10B-4bit-MLX-CRACK",
        "reasoning_in_content": True,  # CoT is emitted in content
        "typical_tokens": 2500,
        "max_tokens": 8000,
        "timeout": 180,
    },
    "gemma-moe": {
        "base_url": "http://localhost:8085/v1",
        "model_id": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "reasoning_in_content": False,  # separate reasoning_content field
        "typical_tokens": 400,
        "max_tokens": 1000,
        "timeout": 60,
    },
    "gemma-31b": {
        "base_url": "http://localhost:8084/v1",
        "model_id": "mlx-community/gemma-4-31b-it-8bit",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 1000,
        "timeout": 60,
    },
    "gemma-remote": {
        "base_url": "http://100.97.101.59:11434/v1",
        "model_id": "gemma-4-26b",
        "reasoning_in_content": False,
        "reasoning_effort": "medium",
        "typical_tokens": 400,
        "max_tokens": 12000,
        "num_ctx": 32768,
        "timeout": 600,
    },
}


@dataclass
class ModelResponse:
    """Response from a model call with unified fields."""
    content: str            # Final assistant message (may be empty if all reasoning)
    reasoning: str          # Chain-of-thought text (may be empty)
    tokens: int             # Total completion tokens
    raw_text: str           # Content + reasoning joined (for parsing)
    finish_reason: str      # "stop", "length", etc.


class ModelClient:
    """Unified client for calling LLMs across backends."""

    def __init__(self, model_name: str):
        if model_name in LOCAL_MODELS:
            self.config = LOCAL_MODELS[model_name]
            self.backend = "openai_compat"
            self._setup_openai_client()
        elif model_name.startswith("claude-"):
            self.config = {"model_id": model_name, "reasoning_in_content": False,
                           "max_tokens": 2000, "timeout": 120}
            self.backend = "anthropic"
            self._setup_anthropic_client()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.model_name = model_name

    def _setup_openai_client(self):
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.config["base_url"],
            api_key="not-needed",
        )

    def _setup_anthropic_client(self):
        import anthropic
        self._client = anthropic.Anthropic()

    def call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.1,
        retries: int = 3,
    ) -> ModelResponse:
        """Call the model with a system prompt and messages.

        Returns ModelResponse with unified fields regardless of backend.
        Retries on timeout errors.
        """
        import time as _time

        mt = max_tokens or self.config.get("max_tokens", 2000)
        timeout = self.config.get("timeout", 120)

        for attempt in range(retries):
            try:
                if self.backend == "openai_compat":
                    return self._call_openai_compat(system, messages, mt, temperature, timeout)
                elif self.backend == "anthropic":
                    return self._call_anthropic(system, messages, mt, temperature, timeout)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
            except Exception as e:
                if attempt < retries - 1 and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                    _time.sleep(2 ** attempt)
                    continue
                raise
        # Unreachable — the loop above either returns or raises.
        raise RuntimeError("call() exhausted retries without returning")

    def _call_openai_compat(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs = dict(
            model=self.config["model_id"],
            messages=full_messages,
            max_tokens=mt,
            temperature=temp,
            timeout=timeout,
        )
        # Pass Ollama-specific options via extra_body
        extra_body = {}
        if self.config.get("reasoning_effort"):
            extra_body["reasoning_effort"] = self.config["reasoning_effort"]
        if self.config.get("num_ctx"):
            extra_body["num_ctx"] = self.config["num_ctx"]
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # For models where reasoning is IN content, raw_text = content
        # For models with separate reasoning, raw_text = reasoning + content
        if self.config.get("reasoning_in_content"):
            raw_text = content
        else:
            raw_text = (reasoning + "\n" + content) if reasoning else content

        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=response.usage.completion_tokens,
            raw_text=raw_text,
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    def _call_anthropic(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
    ) -> ModelResponse:
        response = self._client.messages.create(
            model=self.config["model_id"],
            max_tokens=mt,
            system=system,
            messages=messages,
            temperature=temp,
        )
        content = response.content[0].text
        return ModelResponse(
            content=content,
            reasoning="",
            tokens=response.usage.output_tokens,
            raw_text=content,
            finish_reason=response.stop_reason or "stop",
        )


# Verdict parsing and score mapping live in scorers._prompts — this module
# is the model client, not an output parser. See _prompts.extract_verdict.
