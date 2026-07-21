"""OpenAI-compatible local HTTP server (stdlib only)."""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import magicmindnet as ai
from magicmindnet.chat import format_chat_messages

__all__ = ["OpenAIServer"]


class OpenAIServer:
    """Serve a saved MagicMindNet Chatbot over a small OpenAI-shaped HTTP API."""

    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self._model: Any | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._model_id = "mmn"

    def start(self) -> str:
        if self._httpd is not None:
            return f"http://{self.host}:{self.port}"
        self._model = ai.load(self.model_path)
        # Prefer filename stem as model id when available.
        name = self.model_path.replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            self._model_id = name.rsplit(".", 1)[0] or "mmn"
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # port=0 → ephemeral; record the bound port.
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._model = None


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


def _make_handler(server: OpenAIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _send(self, code: int, payload: dict[str, Any] | list[Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in _cors_headers().items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            for k, v in _cors_headers().items():
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True})
                return
            if path == "/v1/models":
                mid = server._model_id
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": mid,
                                "object": "model",
                                "owned_by": "magicmindnet",
                            }
                        ],
                    },
                )
                return
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": {"message": str(exc), "type": "invalid_request"}})
                return
            model = server._model
            if model is None:
                self._send(503, {"error": {"message": "server not started", "type": "unavailable"}})
                return
            if path == "/v1/chat/completions":
                self._chat(model, body)
                return
            if path == "/v1/embeddings":
                self._embed(model, body)
                return
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})

        def _chat(self, model: Any, body: dict[str, Any]) -> None:
            messages = body.get("messages") or []
            if not isinstance(messages, list) or not messages:
                self._send(
                    400,
                    {"error": {"message": "messages required", "type": "invalid_request"}},
                )
                return
            max_tokens = int(body.get("max_tokens") or body.get("max_new_tokens") or 32)
            temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
            model_name = str(body.get("model") or server._model_id)
            # Prefer chat_messages when available; fall back to format + generate.
            if hasattr(model, "chat_messages"):
                content = model.chat_messages(
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                prompt = format_chat_messages(messages)
                content = model.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
            self._send(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": model_name,
                },
            )

        def _embed(self, model: Any, body: dict[str, Any]) -> None:
            inp = body.get("input", "")
            model_name = str(body.get("model") or server._model_id)
            if isinstance(inp, list):
                vectors = model.embed(inp)
                data = [
                    {"object": "embedding", "embedding": vec, "index": i}
                    for i, vec in enumerate(vectors)
                ]
            else:
                vec = model.embed(str(inp))
                data = [{"object": "embedding", "embedding": vec, "index": 0}]
            self._send(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": model_name,
                },
            )

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible MagicMindNet server")
    parser.add_argument("--model", required=True, help="Path to a saved Chatbot checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = OpenAIServer(model_path=args.model, host=args.host, port=args.port)
    base = server.start()
    print(f"MagicMindNet OpenAI server listening at {base}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
