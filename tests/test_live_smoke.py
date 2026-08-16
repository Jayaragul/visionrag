"""End-to-end smoke test of the live path.

Uses the `stub` detector so the test needs no model download and no network,
and drives the real WebSocket route through FastAPI's test client -- the same
code path a phone exercises, minus the camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "apps"))

from fastapi.testclient import TestClient  # noqa: E402

import api.server as server  # noqa: E402
from visionrag.config import Config  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Point the server at a temp directory and the zero-dependency detector."""

    def fake_config() -> Config:
        cfg = Config()
        cfg.detector.backend = "stub"
        cfg.tracker.min_hits = 2
        # Analyse every frame. Live sessions timestamp by wall clock, so a
        # rate-limited scheduler would skip most frames in a test loop that
        # sends them far faster than real time -- and how many survived would
        # depend on machine speed.
        cfg.scheduler.mode = "all"
        cfg.store.db_path = str(tmp_path / "memory.db")
        cfg.store.evidence_dir = str(tmp_path / "evidence")
        return cfg

    monkeypatch.setattr(server, "_base_config", fake_config)
    # Redirect world memory into the temp dir. Without this, running the tests
    # writes real places and object instances into the repository's runs/.
    monkeypatch.setattr(server, "_runs_dir", lambda: tmp_path)
    server.SESSIONS.clear()
    with TestClient(server.app) as c:
        yield c


def frame_jpeg(ax: float, size=(480, 360)) -> bytes:
    """One frame with the orange object at `ax` and the green one fixed,
    matching the hue bands the stub detector looks for."""
    w, h = size
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (60, 60, 60), 1)
    for cx, colour in ((ax, (0, 140, 255)), (0.70, (0, 220, 0))):
        px, py = int(cx * w), int(0.5 * h)
        cv2.rectangle(img, (px - 24, py - 36), (px + 24, py + 36), colour, -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_session_lifecycle_and_events(client):
    session = client.post("/api/sessions", json={"retention_mode": "evidence"}).json()
    sid = session["session_id"]
    assert session["detector"] == "stub"

    results = []
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        # Sweep the orange object toward the green one, then hold.
        for ax in [0.15 + i * 0.03 for i in range(15)] + [0.58] * 10:
            ws.send_bytes(frame_jpeg(ax))
            results.append(ws.receive_json())

    processed = [r for r in results if not r.get("skipped")]
    assert processed, "no frames were processed"

    # Both objects should be detected and tracked with stable ids.
    last = processed[-1]
    assert len(last["detections"]) == 2
    assert {d["class"] for d in last["detections"]} == {"object_a", "object_b"}
    for d in last["detections"]:
        assert 0.0 <= d["box"]["x"] <= 1.0
        assert 0.0 <= d["box"]["y"] <= 1.0

    events = client.get(f"/api/sessions/{sid}/events").json()
    types = {e["type"] for e in events["events"]}
    assert "appeared" in types
    # The sweep closes the gap, so the pair must register as approaching.
    assert "approached" in types or "near" in types


def test_query_grounds_and_abstains(client):
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        for ax in [0.15 + i * 0.03 for i in range(15)]:
            ws.send_bytes(frame_jpeg(ax))
            ws.receive_json()

    answered = client.post(
        f"/api/sessions/{sid}/query", json={"question": "what appeared?"}
    ).json()
    assert answered["abstained"] is False
    assert answered["evidence_ids"], "an answer must cite evidence"
    assert answered["time_range"] is not None

    # A question the stored evidence cannot support must abstain rather than
    # produce a plausible-sounding guess.
    empty = client.post(
        f"/api/sessions/{sid}/query",
        json={"question": "what happened?", "min_confidence": 0.999},
    ).json()
    assert empty["abstained"] is True
    assert empty["evidence_ids"] == []


def test_delete_removes_session(client):
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        ws.send_bytes(frame_jpeg(0.3))
        ws.receive_json()

    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}/events").status_code == 404
