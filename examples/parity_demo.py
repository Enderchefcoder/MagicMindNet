#!/usr/bin/env python3
"""Feature-parity smoke: stream, embed, typical_p, chat_messages, optional tools."""

from __future__ import annotations

import magicmindnet as ai


def main() -> None:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=11)
    chunks = list(bot.generate_stream("hi", max_new_tokens=4, temperature=0.0))
    vec = bot.embed("hello")
    typical = bot.generate("hi", max_new_tokens=4, temperature=0.8, typical_p=0.9)
    msgs = bot.chat_messages(
        [{"role": "user", "content": "hi"}],
        max_new_tokens=4,
        temperature=0.0,
    )
    print(f"stream_chunks={len(chunks)} embed_dim={len(vec)}")
    print(f"typical_p_sample={typical!r}")
    print(f"chat_messages={msgs!r}")

    if hasattr(bot, "chat_with_tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Return the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        out = bot.chat_with_tools(
            [{"role": "user", "content": "what time is it?"}],
            tools=tools,
            max_new_tokens=16,
            temperature=0.0,
        )
        print(f"tools_keys={sorted(out.keys())}")

    print("parity_demo ok")


if __name__ == "__main__":
    main()
