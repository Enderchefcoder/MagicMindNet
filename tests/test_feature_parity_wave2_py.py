"""Wave 2 parity: JSON mode / grammar constrain + OpenAI-compatible serve."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import magicmindnet as ai


def test_generate_json_mode_produces_object_like_text():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=11)
    out = bot.generate(
        'Return JSON: {"ok": true}',
        max_new_tokens=24,
        temperature=0.8,
        json_mode=True,
    )
    assert isinstance(out, str)
    # Must be parseable JSON object or array after strip
    text = out.strip()
    assert text.startswith("{") or text.startswith("[")
    json.loads(text)  # raises if invalid


def test_generate_grammar_charset_restricts_tokens():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=2)
    # Only digits allowed in new tokens (byte vocab)
    out = bot.generate(
        "n=",
        max_new_tokens=6,
        temperature=0.9,
        grammar="digit",
    )
    assert isinstance(out, str)
    assert all(c.isdigit() for c in out if c), f"non-digit in {out!r}"


def test_openai_compat_server_chat_completions(tmp_path):
    from magicmindnet.serve import OpenAIServer

    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=3)
    path = tmp_path / "tiny.mmn"
    bot.save(str(path))

    server = OpenAIServer(model_path=str(path), host="127.0.0.1", port=0)
    base = server.start()
    try:
        # Wait briefly for listen
        time.sleep(0.15)
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "mmn",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                    "temperature": 0.0,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)

        req2 = urllib.request.Request(
            base + "/v1/embeddings",
            data=json.dumps({"model": "mmn", "input": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=10) as resp:
            emb = json.loads(resp.read().decode())
        assert emb["object"] == "list"
        assert len(emb["data"][0]["embedding"]) == 16
    finally:
        server.stop()


def test_serve_exports():
    assert hasattr(ai, "serve") or "OpenAIServer" in dir(ai)
    from magicmindnet.serve import OpenAIServer

    assert OpenAIServer is not None
