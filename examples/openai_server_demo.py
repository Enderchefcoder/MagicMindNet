#!/usr/bin/env python3
"""Train a tiny bot, start OpenAI-compatible server, hit /v1/chat/completions."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import magicmindnet as ai
from magicmindnet.serve import OpenAIServer


def main() -> None:
    data = ai.DatasetQA(
        data=[
            {"input": "hi", "output": "hello there!"},
            {"input": "ping", "output": "pong"},
        ]
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=11)
    bot.train(data, epochs=4, verbose=False)
    path = "_openai_server_demo.mmn"
    bot.save(path)

    server = OpenAIServer(model_path=path, host="127.0.0.1", port=0)
    base = server.start()
    try:
        health = urllib.request.urlopen(f"{base}/health", timeout=5).read().decode()
        print("health:", health.strip())
        body = json.dumps(
            {
                "model": "mmn",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 12,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        content = payload["choices"][0]["message"]["content"]
        print("completion:", content)
        print("openai_server_demo ok")
    except urllib.error.URLError as exc:
        raise SystemExit(f"server request failed: {exc}") from exc
    finally:
        server.stop()


if __name__ == "__main__":
    main()
