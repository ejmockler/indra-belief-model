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
    # Minimax-M2.7 (JANGTQ-CRACK quant) served via vmlx-engine on a local
    # MLX backend. Same response shape as gemma — separate reasoning_content
    # field, JSON content. Faster than gemma-remote (~43 tok/sec vs 22).
    "minimax-local": {
        "base_url": "http://localhost:8086/v1",
        "model_id": "minimax-m2.7-jangtq-crack",
        "reasoning_in_content": False,
        "typical_tokens": 2500,
        # vmlx server is started with --max-tokens 32768 — match the cap.
        # Reasoning is always-on (--default-enable-thinking true) so 32K
        # gives the model plenty of room for CoT + structured output.
        "max_tokens": 32000,
        "timeout": 600,
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
        # O-phase circuit breaker (was 600s): healthy parse_evidence /
        # grounding calls finish in <30s on this endpoint. 90s is
        # ~3× the healthy median — long enough to catch a slow but
        # working response, short enough to fail fast under endpoint
        # degradation. Combined with O-phase retry removal in
        # parse_evidence, max per-record cost on degradation is
        # ~3 × 90s (parse + 2 grounding) = ~5 min, vs the pre-O 50 min
        # observed during the killed N9 holdout (2026-04-29).
        "timeout": 90,
    },
    # Google AI Studio (Gemma 4) — hosted Gemma via the Gemini API's
    # OpenAI-compatibility endpoint. Same weights as the local gemma-moe /
    # gemma-31b but routed through Google's infrastructure: no tailscale
    # latency, no LiteLLM proxy in the path (eliminates the channel-token
    # 500 class of failures), and significantly higher per-request
    # throughput. Auth: GEMINI_API_KEY env var.
    "gemma-google-moe": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_id": "gemma-4-26b-a4b-it",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 8192,
        "timeout": 120,
        # PaidTier3 quota: 16k input tokens/min for Gemma. With ~3k input
        # tokens per parse_evidence call, true sustainable throughput is
        # ~5 req/min — so concurrency above 2 just burns the budget on
        # bursts then waits 15s for the OpenAI client retry. Keep this
        # conservative; --workers can override per-run if quota changes.
        "concurrency_hint": 2,
    },
    "gemma-google-31b": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_id": "gemma-4-31b-it",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_in_content": False,
        "typical_tokens": 400,
        "max_tokens": 8192,
        "timeout": 120,
        "concurrency_hint": 2,
    },
}


_RETRY_DELAY_RE = __import__("re").compile(
    r"retry in (\d+(?:\.\d+)?)s|retryDelay['\"]:\s*['\"](\d+)s",
    flags=__import__("re").IGNORECASE,
)


def _parse_retry_delay(error_text: str, default: float = 30.0) -> float:
    """Pull the API's requested retry delay (seconds) out of a 429 error
    payload. Google formats it both as 'Please retry in 50.05s.' and as a
    structured RetryInfo block with retryDelay='50s'. Falls back to a
    reasonable default if neither is present."""
    m = _RETRY_DELAY_RE.search(error_text)
    if m:
        for g in m.groups():
            if g:
                try:
                    return float(g)
                except ValueError:
                    pass
    return default


def concurrency_hint(model_name: str) -> int:
    """Reasonable default max-concurrency for a model. 1 means serial.

    Local Ollama endpoints default to 1 (single-GPU serving); hosted
    backends like Google AI Studio can fan out. Callers may override."""
    cfg = LOCAL_MODELS.get(model_name)
    if cfg is None:
        return 1
    return int(cfg.get("concurrency_hint", 1))


@dataclass
class ModelResponse:
    """Response from a model call with unified fields."""
    content: str            # Final assistant message (may be empty if all reasoning)
    reasoning: str          # Chain-of-thought text (may be empty)
    tokens: int             # Total completion tokens
    raw_text: str           # Content + reasoning joined (for parsing)
    finish_reason: str      # "stop", "length", etc.
    prompt_tokens: int = -1  # Input tokens (-1 if backend doesn't report)


class ModelClient:
    """Unified client for calling LLMs across backends.

    Telemetry: every successful or failed call() appends one entry to a
    thread-local call log (`_tls.call_log`). Callers can snapshot and
    clear the log via `pop_call_log()`. The thread-local design works
    cleanly with ThreadPoolExecutor — each worker accumulates its own
    record's calls without cross-contamination.

    Wall-time guard (Q3): each call is dispatched on a class-level
    ThreadPoolExecutor and `result(timeout=N)` enforces a hard wall-time
    cap. The OpenAI SDK's `timeout` field is a per-chunk / connection
    timeout — for streaming generations it does not bound total wall
    time. The Q3 wrapper raises TimeoutError after `timeout` seconds
    regardless of underlying transport behavior. The in-flight urllib3
    request continues until the SDK's transport timeout reaps it (a
    transient thread leak; documented and accepted).
    """
    # Shared across instances; each call only consumes one slot for its
    # duration. 8 max workers covers single-threaded scoring + a few
    # concurrent ModelClient instances without bloat.
    import concurrent.futures as _cf
    _WALL_POOL = _cf.ThreadPoolExecutor(max_workers=8,
                                        thread_name_prefix="mc-wall")

    def __init__(self, model_name: str):
        # Set name first so setup helpers can use it in error messages.
        self.model_name = model_name
        # Thread-local call log; see `pop_call_log()`.
        import threading as _threading
        self._tls = _threading.local()
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

    def _get_call_log(self) -> list[dict]:
        if not hasattr(self._tls, "call_log"):
            self._tls.call_log = []
        return self._tls.call_log

    def pop_call_log(self) -> list[dict]:
        """Return the current thread's call log and clear it."""
        log = self._get_call_log()
        snapshot = list(log)
        log.clear()
        return snapshot

    def _invoke_with_wall_timeout(self, fn, timeout: int, *args, **kwargs):
        """Run `fn(*args, **kwargs)` with a hard wall-time cap.

        On timeout: raise TimeoutError immediately. The in-flight thread
        is abandoned (cannot cleanly cancel a running urllib3 request);
        urllib3's transport timeout will eventually reap it. The leak is
        bounded — at most one zombie thread per timeout incident, and
        the pool's max_workers=8 caps total concurrent leaks.

        Q3 fix (2026-05-01): the OpenAI SDK's `timeout` field is per-
        connection / per-chunk and does NOT bound total wall time on
        streaming generations. CXCL14 holdout record observed: 255s call
        wall time despite `timeout=90` in the model config. This wrapper
        is the actual circuit breaker.
        """
        future = self._WALL_POOL.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except self._cf.TimeoutError as e:
            raise TimeoutError(
                f"ModelClient.call exceeded {timeout}s wall-clock time"
            ) from e

    def _setup_openai_client(self):
        import os
        from openai import OpenAI
        # Hosted endpoints (Google, etc.) need a real key; local Ollama
        # endpoints don't care. `api_key_env` in the model config names the
        # env var to read; absence falls back to "not-needed" for local.
        api_key = "not-needed"
        env_var = self.config.get("api_key_env")
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"model {self.model_name!r} requires {env_var} in the "
                    f"environment (not set). Source it from your .env or "
                    f"export it before instantiating ModelClient."
                )
        self._client = OpenAI(
            base_url=self.config["base_url"],
            api_key=api_key,
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
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        kind: str = "unknown",
    ) -> ModelResponse:
        """Call the model with a system prompt and messages.

        Returns ModelResponse with unified fields regardless of backend.

        Retry doctrine (P-phase): the only retry-on-error is for 429 rate
        limits, which respect the server-requested delay. Timeouts and
        connection errors raise on the first occurrence — callers
        (parse_evidence, verify_grounding) abstain via their existing
        TimeoutError handlers. Pre-P holdouts saw retries amplify
        endpoint degradation: a slow record cost 3 × 600s = 30 min
        before failing.

        `response_format` constrains the output. Pass
        `{"type": "json_object"}` to force JSON-only output on backends that
        support it. Sub-calls that consume JSON (parse_evidence, grounding)
        opt in; backends that don't honor the constraint fall through to
        the previous behavior (caller still parses tolerantly).

        `reasoning_effort` (when set) overrides the per-model config's
        reasoning_effort. Pass "none" on sub-calls that are pure extraction
        (parse_evidence, verify_grounding) — with the config default of
        "medium", gemma-remote burns 12000+ tokens on reasoning_content
        before emitting JSON, causing silent truncation on the 3000-token
        per-call budget. "none" keeps the reasoning brief (~60 tokens on
        a simple test) and lets content populate.

        `kind` is a free-form telemetry label. Sub-callers pass
        "parse_evidence" / "verify_grounding" / "monolithic" so post-hoc
        analysis can stratify latency and truncation rates per call type.
        """
        import time as _time

        mt = max_tokens or self.config.get("max_tokens", 2000)
        timeout = self.config.get("timeout", 120)
        # Approximate prompt size up-front for telemetry (cheap when the
        # backend doesn't report prompt_tokens in the response).
        prompt_chars = len(system or "") + sum(
            len(m.get("content", "") or "") for m in messages
        )

        # 429 quota retries are bounded by this counter; everything else
        # raises on first occurrence (P-phase doctrine).
        rate_limit_retries = 5
        t_start = _time.time()

        try:
            while True:
                try:
                    if self.backend == "openai_compat":
                        response = self._invoke_with_wall_timeout(
                            self._call_openai_compat, timeout,
                            system, messages, mt, temperature, timeout,
                            response_format=response_format,
                            reasoning_effort=reasoning_effort,
                        )
                    elif self.backend == "anthropic":
                        response = self._invoke_with_wall_timeout(
                            self._call_anthropic, timeout,
                            system, messages, mt, temperature, timeout,
                        )
                    else:
                        raise ValueError(f"Unknown backend: {self.backend}")
                    self._get_call_log().append({
                        "kind": kind,
                        "duration_s": round(_time.time() - t_start, 3),
                        "prompt_chars": prompt_chars,
                        "prompt_tokens": response.prompt_tokens,
                        "out_tokens": response.tokens,
                        "finish_reason": response.finish_reason,
                        # Layer B capture — persist raw LLM I/O for tracing.
                        # Lets the viewer reconstruct what the model saw and
                        # what it said. Cost: ~1-10KB per call (typical) up
                        # to 30+KB when reasoning_content runs hot.
                        "system": system,
                        "messages": messages,
                        "model_id": self.config.get("model_id"),
                        "content": response.content,
                        "reasoning": response.reasoning,
                    })
                    return response
                except Exception as e:
                    msg = str(e).lower()
                    # 429 / rate-limit: respect the server's requested delay.
                    # This is the ONLY in-client retry.
                    if ("429" in msg or "rate limit" in msg
                            or "resource_exhausted" in msg) and rate_limit_retries > 0:
                        delay = _parse_retry_delay(str(e))
                        # tiny safety pad so the next request lands clean
                        _time.sleep(delay + 1)
                        rate_limit_retries -= 1
                        continue
                    raise
        except Exception as e:
            self._get_call_log().append({
                "kind": kind,
                "duration_s": round(_time.time() - t_start, 3),
                "prompt_chars": prompt_chars,
                "prompt_tokens": -1,
                "out_tokens": 0,
                "finish_reason": None,
                "error": type(e).__name__,
            })
            raise

    def _call_openai_compat(
        self, system: str, messages: list[dict], mt: int, temp: float, timeout: int,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system}] + messages
        kwargs = dict(
            model=self.config["model_id"],
            messages=full_messages,
            max_tokens=mt,
            temperature=temp,
            timeout=timeout,
        )
        if response_format is not None:
            # OpenAI / Google AI Studio honor `response_format` directly.
            # Ollama-backed endpoints expose JSON mode under their native
            # `format` field — set it via extra_body as a fallback for
            # backends where response_format alone isn't enough.
            kwargs["response_format"] = response_format
        # Pass backend-specific options via extra_body. Two reasoning-
        # control mechanisms are honored across backends:
        #   - `reasoning_effort` (low/medium/high): standard OpenAI extension
        #     used by Google AI Studio and some Ollama versions.
        #   - `chat_template_kwargs.enable_thinking` (bool): the actual
        #     mechanism Ollama-served Gemma honors. `reasoning_effort="none"`
        #     is silently dropped by Ollama, leaving thinking ON at the
        #     model's default. We send BOTH; whichever the backend understands
        #     wins. Q1 fix (2026-05-01) — earlier P-phase code only sent
        #     reasoning_effort, which is why parse_evidence calls were
        #     emitting 2400+ tokens of reasoning_content despite the
        #     "reasoning_effort=none" doctrine.
        extra_body = {}
        effort = reasoning_effort if reasoning_effort is not None \
                 else self.config.get("reasoning_effort")
        if effort:
            extra_body["reasoning_effort"] = effort
            # When the caller asks for "none", that's a request to disable
            # thinking entirely. Translate to the chat_template_kwargs
            # mechanism Ollama honors.
            if effort == "none":
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.config.get("num_ctx"):
            extra_body["num_ctx"] = self.config["num_ctx"]
        if response_format is not None:
            # Belt-and-suspenders: set Ollama's native `format` field too.
            # Backends that don't recognize it ignore unknown extra_body keys.
            extra_body["format"] = "json"
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
            prompt_tokens=getattr(response.usage, "prompt_tokens", -1),
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
        # Anthropic returns a list of content blocks. Extract:
        #   - `text` block → goes into `content`
        #   - `thinking` block (extended-thinking) → goes into `reasoning`
        # Tool-use blocks intentionally ignored — we don't use native tools
        # (see module docstring). Iterating preserves capture under future
        # API additions where order matters.
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_parts.append(getattr(block, "text", "") or "")
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
        content = "".join(content_parts)
        reasoning = "\n".join(thinking_parts) if thinking_parts else ""
        raw_text = (reasoning + "\n" + content) if reasoning else content
        return ModelResponse(
            content=content,
            reasoning=reasoning,
            tokens=response.usage.output_tokens,
            raw_text=raw_text,
            finish_reason=response.stop_reason or "stop",
            prompt_tokens=getattr(response.usage, "input_tokens", -1),
        )


# Verdict parsing and score mapping live in scorers._prompts — this module
# is the model client, not an output parser. See _prompts.extract_verdict.
