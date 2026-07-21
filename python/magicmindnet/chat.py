"""Chat message formatting (Ollama / OpenAI ChatML) + tool-calling helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from magicmindnet._native import Chatbot as _Chatbot
from magicmindnet._native import format_chat_messages

__all__ = [
    "format_chat_messages",
    "format_tools_prompt",
    "parse_tool_calls",
]


def format_tools_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
    """ChatML prompt that instructs the model to emit OpenAI-style tool_calls JSON."""
    tool_lines: list[str] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            fn = t if isinstance(t, dict) else {}
        name = str(fn.get("name", "unknown"))
        desc = str(fn.get("description", ""))
        params = fn.get("parameters", {})
        tool_lines.append(
            f"- {name}: {desc}\n  parameters: {json.dumps(params, separators=(',', ':'))}"
        )
    system = (
        "You are a function-calling assistant. When a tool is needed, reply with ONLY "
        'JSON of the form {"tool_calls":[{"name":"...","arguments":{...}}]}. '
        "Available tools:\n"
        + "\n".join(tool_lines)
    )
    msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
    msgs.extend(messages)
    return format_chat_messages(msgs, add_generation_prompt=add_generation_prompt)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract `tool_calls` from model text (JSON object or fenced blob)."""
    if not text:
        return []
    candidates: list[str] = [text.strip()]
    # Fenced ```json ... ```
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1))
    # First {...} spanning tool_calls
    for m in re.finditer(r"\{[^{}]*\"tool_calls\".*\}", text, flags=re.DOTALL):
        candidates.append(m.group(0))
    # Brute: find outermost JSON object containing tool_calls
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "tool_calls" in data:
            calls = data["tool_calls"]
            if isinstance(calls, list):
                out: list[dict[str, Any]] = []
                for c in calls:
                    if not isinstance(c, dict):
                        continue
                    name = c.get("name") or (c.get("function") or {}).get("name")
                    args = c.get("arguments")
                    if args is None and isinstance(c.get("function"), dict):
                        args = c["function"].get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    if not isinstance(args, dict):
                        args = {}
                    if name:
                        out.append({"name": str(name), "arguments": args})
                return out
        if isinstance(data, list):
            # Bare list of calls
            out = []
            for c in data:
                if isinstance(c, dict) and "name" in c:
                    args = c.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    out.append({"name": str(c["name"]), "arguments": args})
            if out:
                return out
    return []


def chat_with_tools(
    self: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate with a tools schema; return `{content, tool_calls}` (OpenAI-shaped)."""
    prompt = format_tools_prompt(messages, tools=tools)
    kwargs.setdefault("json_mode", True)
    content = self.generate(prompt, **kwargs)
    return {"content": content, "tool_calls": parse_tool_calls(content)}


# Attach OpenAI-style tools API onto the native Chatbot class.
_Chatbot.chat_with_tools = chat_with_tools  # type: ignore[method-assign, attr-defined]
