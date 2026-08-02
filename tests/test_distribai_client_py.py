"""DistribAIClient — admin HTTP API wire format against a stub orchestrator.

The stub replays DistribAI's admin routes (``/admin/health``, ``/admin/jobs``,
...) including Bearer-token auth, so every request/response shape the client
produces is checked without a live grid.
"""

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import magicmindnet as ai
from magicmindnet import distribai as bridge

ADMIN_SECRET = "test-admin-secret"


class _StubOrchestrator(BaseHTTPRequestHandler):
    requests: list = []

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        return self.headers.get("Authorization") == f"Bearer {ADMIN_SECRET}"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == "/admin/health":
            self._send({"ok": True, "active_nodes": 2, "queued_jobs": 1})
        elif self.path == "/admin/stats":
            self._send({"connected_nodes": 2})
        elif self.path == "/admin/nodes":
            self._send({"nodes": [{"node_id": "worker-01", "status": "online"}]})
        elif self.path == "/admin/jobs":
            self._send({"jobs": [{"job_id": "job-1", "status": "queued"}]})
        elif self.path.startswith("/admin/jobs/"):
            self._send({"job_id": self.path.rsplit("/", 1)[-1], "status": "running"})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/admin/jobs":
            if not self._authorized():
                self._send({"error": "authentication required"}, status=401)
                return
            type(self).requests.append(body)
            self._send({"ok": True, "job_id": "job-20260802-abc", "task_id": "task-def"})
        elif self.path.endswith("/retry"):
            self._send({"ok": True, "status": "queued"})
        else:
            self._send({"error": "not found"}, status=404)

    def do_DELETE(self):
        if self.path.startswith("/admin/jobs/"):
            self._send({"ok": True, "status": "cancelled"})
        else:
            self._send({"error": "not found"}, status=404)


@pytest.fixture
def stub_orchestrator():
    _StubOrchestrator.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StubOrchestrator)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def make_client(url):
    return bridge.DistribAIClient(admin_url=url, admin_secret=ADMIN_SECRET, timeout=5)


def test_client_defaults_follow_distribai_conventions(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_ADMIN_URL", raising=False)
    monkeypatch.delenv("DISTRIBAI_ADMIN_SECRET", raising=False)
    client = bridge.DistribAIClient()
    assert client.admin_url == bridge.DEFAULT_ADMIN_URL
    assert client.admin_secret == ""
    monkeypatch.setenv("ORCHESTRATOR_ADMIN_URL", "http://grid.example:8766/")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "s3cret")
    client = bridge.DistribAIClient()
    assert client.admin_url == "http://grid.example:8766"
    assert client.admin_secret == "s3cret"


def test_client_reads_health_stats_nodes_jobs(stub_orchestrator):
    client = make_client(stub_orchestrator)
    assert client.health()["ok"] is True
    assert client.stats()["connected_nodes"] == 2
    assert client.nodes()["nodes"][0]["node_id"] == "worker-01"
    assert client.jobs()["jobs"][0]["job_id"] == "job-1"
    assert client.job("job-1")["status"] == "running"
    assert client.cancel_job("job-1")["status"] == "cancelled"
    assert client.retry_job("job-1")["ok"] is True


def test_submit_job_sends_distribai_wire_format(stub_orchestrator):
    client = make_client(stub_orchestrator)
    package = bridge.build_script_package()
    arch = bridge.GRID_PROFILES["mmn-tiny"]
    response = client.submit_job(
        job_type="train",
        base_model="mmn-tiny",
        steps=10,
        dataset_ref="local://dataset.json",
        architecture_config=arch,
        script_package=package,
    )
    assert response["job_id"] == "job-20260802-abc"

    sent = _StubOrchestrator.requests[-1]
    assert sent["job_type"] == "train"
    assert sent["base_model"] == "mmn-tiny"
    assert sent["model_name"] == "mmn-tiny"
    assert sent["steps"] == 10
    assert sent["dataset_ref"] == "local://dataset.json"
    assert sent["architecture_config"]["family"] == "decoder_transformer"
    # Script package: base64 payload + execution paradigm + pinned digest.
    assert base64.b64decode(sent["script_package_b64"]) == package
    assert sent["hparams"]["execution_paradigm"] == "script"
    assert sent["hparams"]["package_sha256"] == hashlib.sha256(package).hexdigest()


def test_submit_job_rejects_invalid_architecture_locally(stub_orchestrator):
    client = make_client(stub_orchestrator)
    with pytest.raises(ValueError, match="unsupported architecture_config keys"):
        client.submit_job(architecture_config={"family": "decoder_transformer", "bogus": 1})
    assert _StubOrchestrator.requests == []


def test_submit_training_job_from_chatbot(stub_orchestrator):
    client = make_client(stub_orchestrator)
    bot = ai.Chatbot(vocab_size=48, n_layer=1, d_model=16, seed=3)
    response = client.submit_training_job(
        bot,
        dataset=[{"input": "hi", "output": "yo"}],
        steps=5,
        hyperparams={"epochs": 2},
    )
    assert response["task_id"] == "task-def"
    sent = _StubOrchestrator.requests[-1]
    assert sent["architecture_config"]["dim"] == 16
    assert sent["hparams"]["architecture_config"]["dim"] == 16
    assert sent["hparams"]["epochs"] == 2
    assert "script_package_b64" in sent


def test_submit_training_job_from_config_dict(stub_orchestrator):
    client = make_client(stub_orchestrator)
    response = client.submit_training_job(bridge.GRID_PROFILES["mmn-tiny"], steps=1)
    assert response["ok"] is True


def test_client_maps_http_errors_to_distribai_error(stub_orchestrator):
    unauthorized = bridge.DistribAIClient(admin_url=stub_orchestrator, admin_secret="wrong", timeout=5)
    with pytest.raises(bridge.DistribAIError, match="HTTP 401.*authentication required"):
        unauthorized.submit_job(steps=1)
    client = make_client(stub_orchestrator)
    with pytest.raises(bridge.DistribAIError, match="HTTP 404"):
        client._request("GET", "/admin/unknown")


def test_client_maps_unreachable_host_to_distribai_error():
    client = bridge.DistribAIClient(admin_url="http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(bridge.DistribAIError, match="cannot reach DistribAI orchestrator"):
        client.health()
