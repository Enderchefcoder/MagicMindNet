"""Wave 3: OpenAI-style tools / function-calling helper."""

from __future__ import annotations

import magicmindnet as ai
from magicmindnet.chat import format_tools_prompt, parse_tool_calls


def test_format_tools_prompt_includes_schemas():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    text = format_tools_prompt(
        [{"role": "user", "content": "Weather in Paris?"}],
        tools=tools,
    )
    assert "get_weather" in text
    assert "Paris" in text
    assert "tool_call" in text.lower() or "function" in text.lower()


def test_parse_tool_calls_from_json_blob():
    raw = 'Sure.\n{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}'
    calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"]["city"] == "Paris"


def test_chatbot_chat_with_tools_json_mode():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=14)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two ints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    # Tiny models won't reliably emit tool JSON; API must still return structured result.
    result = bot.chat_with_tools(
        [{"role": "user", "content": "add 1 and 2"}],
        tools=tools,
        max_new_tokens=48,
        temperature=0.0,
        json_mode=True,
    )
    assert isinstance(result, dict)
    assert "content" in result
    assert "tool_calls" in result
    assert isinstance(result["tool_calls"], list)


def test_tools_exports():
    assert callable(ai.format_tools_prompt)
    assert callable(ai.parse_tool_calls)
