"""Unified model client and output parsing across backends.

Supports:
- OpenAI-compatible APIs (vllm-mlx serving Qwen, Gemma, etc.)
- Anthropic API (Claude)
- Local models that emit reasoning in content (Qwen CRACK variants)
- Local models with separate reasoning_content (Gemma-4, Qwen3-thinking)

Design principles:
1. Single ModelClient interface; backend detail hidden
2. Output extraction tries multiple strategies in order:
   (a) strict JSON from content field
   (b) JSON from markdown code blocks
   (c) JSON anywhere in content or reasoning
   (d) verdict phrases extracted from reasoning text
3. Failures return None + raw text for debugging; never crash the pipeline
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable


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
        "max_tokens": 4000,
        "timeout": 120,
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
    ) -> ModelResponse:
        """Call the model with a system prompt and messages.

        Returns ModelResponse with unified fields regardless of backend.
        """
        mt = max_tokens or self.config.get("max_tokens", 2000)
        timeout = self.config.get("timeout", 120)

        if self.backend == "openai_compat":
            return self._call_openai_compat(system, messages, mt, temperature, timeout)
        elif self.backend == "anthropic":
            return self._call_anthropic(system, messages, mt, temperature, timeout)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
        # Pass reasoning_effort for Ollama endpoints that support it
        if self.config.get("reasoning_effort"):
            kwargs["extra_body"] = {"reasoning_effort": self.config["reasoning_effort"]}
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

    # ------------------------------------------------------------------
    # Tool-calling support (Gemma 4 native format)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_gemma4_tool_calls(text: str) -> list[dict]:
        """Parse Gemma 4 native tool calls from model output.

        Gemma 4 format: <|tool_call>call:FUNCTION_NAME{args}<tool_call|>
        Args use <|"|> as string delimiter.

        Returns list of {name: str, arguments: dict}.
        """
        import re
        pattern = r'<\|tool_call>call:(\w+)\{([^}]*(?:\{[^}]*\}[^}]*)*)\}<tool_call\|>'
        # Also handle the variant without pipe in closing tag
        pattern2 = r'<\|tool_call>call:(\w+)\{([^}]*)\}<tool_call\|>'
        # And the simpler pattern we've seen in practice
        pattern3 = r'call:(\w+)\{([^}]+)\}'

        calls = []
        # First: try text-based TOOL_CALL format
        text_tool_pattern = r'TOOL_CALL:\s*(\w+)\(\s*["\']([^"\']*)["\']'
        for m in re.finditer(text_tool_pattern, text):
            func_name = m.group(1)
            arg_value = m.group(2)
            calls.append({"name": func_name, "arguments": {"entity_name": arg_value}})
        if calls:
            return calls

        # Then: try Gemma 4 native format
        for pat in [pattern, pattern2, pattern3]:
            for m in re.finditer(pat, text):
                func_name = m.group(1)
                args_str = m.group(2)
                args = {}
                args_str = args_str.replace('<|"|>', '"')
                for kv in re.finditer(r'(\w+)\s*[:=]\s*"([^"]*)"', args_str):
                    args[kv.group(1)] = kv.group(2)
                for kv in re.finditer(r"(\w+)\s*[:=]\s*'([^']*)'", args_str):
                    args[kv.group(1)] = kv.group(2)
                if func_name and args:
                    calls.append({"name": func_name, "arguments": args})
                    break
            if calls:
                break
        return calls

    @staticmethod
    def format_gemma4_tool_response(func_name: str, result: dict | str) -> str:
        """Format a tool response in Gemma 4 native format."""
        if isinstance(result, str):
            return f'<|tool_response>response:{func_name}{{result:<|"|>{result}<|"|>}}<tool_response|>'
        # Dict result
        parts = []
        for k, v in result.items():
            if isinstance(v, str):
                parts.append(f'{k}:<|"|>{v}<|"|>')
            else:
                parts.append(f'{k}:{v}')
        return f'<|tool_response>response:{func_name}{{{",".join(parts)}}}<tool_response|>'

    def call_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: dict[str, callable],
        tool_declarations: str = "",
        max_tokens: int = 4000,
        max_tool_rounds: int = 5,
        temperature: float = 0.1,
    ) -> ModelResponse:
        """Call model with tool-use support via Gemma 4 native format.

        The tool_declarations string is appended to the system prompt.
        When the model emits <|tool_call>..., we parse it, execute the
        function, format the response, and continue generation.

        Args:
            system: Base system prompt
            messages: Conversation messages
            tools: Dict mapping function_name → callable(arguments_dict) → result
            tool_declarations: Gemma 4 formatted tool declarations to append to system
            max_tokens: Per-generation token limit
            max_tool_rounds: Max tool call iterations before forcing stop
            temperature: Sampling temperature

        Returns:
            ModelResponse with all reasoning + tool interactions in raw_text
        """
        full_system = system
        if tool_declarations:
            full_system = system + "\n" + tool_declarations

        current_messages = list(messages)
        all_text = []
        total_tokens = 0

        for round_idx in range(max_tool_rounds):
            response = self.call(
                system=full_system,
                messages=current_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            total_tokens += response.tokens
            full_output = response.raw_text
            all_text.append(full_output)

            # Check for tool calls
            tool_calls = self.parse_gemma4_tool_calls(full_output)
            if not tool_calls:
                # No tool call — final response
                break

            # Execute each tool call
            for tc in tool_calls:
                func_name = tc["name"]
                func_args = tc["arguments"]

                if func_name in tools:
                    try:
                        result = tools[func_name](func_args)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Unknown tool: {func_name}"

                # Format response in Gemma 4 format
                tool_resp = self.format_gemma4_tool_response(func_name, result)
                all_text.append(f"[Tool: {func_name}({func_args})]")
                all_text.append(str(result))

                # Append tool call + response as assistant turn, then continue
                assistant_content = full_output + "\n" + tool_resp
                current_messages.append({"role": "assistant", "content": assistant_content})
                current_messages.append({
                    "role": "user",
                    "content": "Continue your analysis based on the tool result above.",
                })

        combined_text = "\n".join(all_text)
        return ModelResponse(
            content=all_text[-1] if all_text else "",
            reasoning="",
            tokens=total_tokens,
            raw_text=combined_text,
            finish_reason="stop",
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


# ============================================================
# Output parsing — multi-strategy extraction
# ============================================================

# Exact JSON pattern: {"verdict": "X", "confidence": "Y"}
# Allows extra fields, any order, single quotes, whitespace.
JSON_VERDICT_PATTERN = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)

# Alternative order: confidence first
JSON_VERDICT_PATTERN_ALT = re.compile(
    r'\{[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?\}',
    re.IGNORECASE,
)

# Fallback: verdict phrase in reasoning text
VERDICT_PHRASE_PATTERNS = [
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.IGNORECASE),
    re.compile(r'(?:final\s+)?(?:verdict|decision|conclusion)[^a-z]*?:[^a-z]*?(?:["\'\*]*)(correct|incorrect)', re.IGNORECASE),
    re.compile(r'\b(?:verdict|decision|answer)\s+(?:is|should be|would be|=)\s*[:"\'\*]*\s*(correct|incorrect)', re.IGNORECASE),
]

CONFIDENCE_PHRASE_PATTERNS = [
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.IGNORECASE),
    re.compile(r'confidence[^a-z]*?:[^a-z]*?(?:["\'\*]*)(high|medium|low)', re.IGNORECASE),
    re.compile(r'confidence\s+(?:is|level)?[^a-z]*?(high|medium|low)', re.IGNORECASE),
    re.compile(r'with\s+(high|medium|low)\s+confidence', re.IGNORECASE),
]


def extract_verdict(text: str) -> tuple[str | None, str | None, str]:
    """Extract (verdict, confidence, strategy) from model output.

    Returns:
        (verdict, confidence, strategy_used)
        strategy_used: "json_exact", "json_alt", "phrase_fallback", or "none"
    """
    if not text:
        return None, None, "none"

    # Strategy 1: exact JSON with verdict first
    matches = JSON_VERDICT_PATTERN.findall(text)
    if matches:
        v, c = matches[-1]
        return v.lower(), c.lower(), "json_exact"

    # Strategy 2: exact JSON with confidence first (alt order)
    matches = JSON_VERDICT_PATTERN_ALT.findall(text)
    if matches:
        c, v = matches[-1]
        return v.lower(), c.lower(), "json_alt"

    # Strategy 3: extract verdict and confidence separately from text
    verdict = None
    for pat in VERDICT_PHRASE_PATTERNS:
        m = pat.findall(text)
        if m:
            verdict = m[-1].lower()
            break

    if not verdict:
        return None, None, "none"

    confidence = "medium"  # default
    for pat in CONFIDENCE_PHRASE_PATTERNS:
        m = pat.findall(text)
        if m:
            confidence = m[-1].lower()
            break

    return verdict, confidence, "phrase_fallback"


def verdict_to_score(verdict: str | None, confidence: str | None) -> float:
    """Convert (verdict, confidence) to P(correct) scalar."""
    if verdict is None:
        return 0.5
    grid = {
        ("correct", "high"): 0.95,
        ("correct", "medium"): 0.80,
        ("correct", "low"): 0.65,
        ("incorrect", "low"): 0.35,
        ("incorrect", "medium"): 0.20,
        ("incorrect", "high"): 0.05,
    }
    return grid.get((verdict, confidence or "medium"), 0.50)


# ============================================================
# Markdown / code block handling
# ============================================================

def strip_markdown_code_block(text: str) -> str:
    """Remove ```json ... ``` wrappers and similar."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])
        return "\n".join(lines[1:])
    return text
