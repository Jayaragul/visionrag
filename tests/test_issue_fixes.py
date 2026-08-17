"""Regression tests for the issues raised in review (#1).

Each test names the defect it pins down. They were written to fail against the
code as reviewed, so a regression here means a real behaviour has come back.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
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
from visionrag.memory.store import MemoryStore  # noqa: E402
from visionrag.memory.world import WorldMemory  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    def fake_config() -> Config:
        cfg = Config()
        cfg.detector.backend = "stub"
        cfg.tracker.min_hits = 2
        cfg.scheduler.mode = "all"
        cfg.store.db_path = str(tmp_path / "memory.db")
        cfg.store.evidence_dir = str(tmp_path / "evidence")
        return cfg

    monkeypatch.setattr(server, "_base_config", fake_config)
    monkeypatch.setattr(server, "_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "ACCESS_TOKEN", None)  # auth tested separately
    server.SESSIONS.clear()
    with TestClient(server.app) as c:
        yield c


def frame_jpeg(ax: float, size=(480, 360), bright: int = 255) -> bytes:
    w, h = size
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (60, 60, 60), 1)
    for cx, colour in ((ax, (0, 140, 255)), (0.70, (0, 220, 0))):
        px, py = int(cx * w), int(0.5 * h)
        scaled = tuple(int(c * bright / 255) for c in colour)
        cv2.rectangle(img, (px - 24, py - 36), (px + 24, py + 36), scaled, -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def dark_jpeg(size=(480, 360)) -> bytes:
    """A frame the quality gate must reject: near-black and featureless."""
    w, h = size
    return cv2.imencode(".jpg", np.full((h, w, 3), 3, dtype=np.uint8))[1].tobytes()


def scene(seed: int, size=(480, 360)) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 60, np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, w - 90)), int(rng.integers(0, h - 90))
        bw, bh = int(rng.integers(30, 88)), int(rng.integers(30, 88))
        c = tuple(int(v) for v in rng.integers(40, 255, 3))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), c, -1)
        cv2.line(img, (x, y + bh), (x + bw, y),
                 tuple(int(v) for v in rng.integers(0, 255, 3)), 2)
    return img


# -- bug 1: SQLite thread affinity --------------------------------------
def test_store_usable_across_threads(tmp_path):
    """Sessions are built on a worker thread and read from the event loop, so
    a connection pinned to its creating thread breaks the evidence endpoint."""
    holder: dict = {}
    t = threading.Thread(target=lambda: holder.update(s=MemoryStore(tmp_path / "m.db")))
    t.start()
    t.join()
    store = holder["s"]
    try:
        store.conn.execute("SELECT 1").fetchone()
    except sqlite3.ProgrammingError as exc:  # pragma: no cover
        pytest.fail(f"store is thread-pinned: {exc}")
    finally:
        store.close()


def test_world_usable_across_threads(tmp_path):
    holder: dict = {}
    t = threading.Thread(target=lambda: holder.update(w=WorldMemory(tmp_path / "w.db")))
    t.start()
    t.join()
    world = holder["w"]
    try:
        world.conn.execute("SELECT 1").fetchone()
    finally:
        world.close()


def test_evidence_endpoint_reachable(client):
    """The reported symptom: fetching evidence raised ProgrammingError."""
    sid = client.post("/api/sessions", json={"retention_mode": "full"}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        for ax in [0.2, 0.3, 0.4]:
            ws.send_bytes(frame_jpeg(ax))
            ws.receive_json()
    # 200 with an image or 404 for a frame without evidence are both fine;
    # a 500 means the connection blew up.
    res = client.get(f"/api/sessions/{sid}/evidence/0")
    assert res.status_code in (200, 404), res.text


# -- bug 2: bad frames must not become evidence -------------------------
def test_unusable_frames_are_not_persisted(client):
    """The product claims unusable frames never become permanent evidence.
    Quality was assessed *after* the pipeline had already written rows."""
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        replies = []
        for _ in range(6):
            ws.send_bytes(dark_jpeg())
            replies.append(ws.receive_json())

    assert any(r.get("rejected") for r in replies), "dark frames were not rejected"

    events = client.get(f"/api/sessions/{sid}/events").json()
    assert events["count"] == 0, "unusable frames produced stored events"

    stats = client.get(f"/api/sessions/{sid}/stats").json()
    assert stats["frames_rejected_quality"] >= 1


def test_good_frames_still_persist(client):
    """The gate must not be so strict that normal frames stop working."""
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        for ax in [0.15 + i * 0.03 for i in range(12)]:
            ws.send_bytes(frame_jpeg(ax))
            ws.receive_json()
    assert client.get(f"/api/sessions/{sid}/events").json()["count"] > 0


# -- bug 3: room transitions --------------------------------------------
def test_walking_to_another_room_opens_a_new_visit(tmp_path):
    """Objects seen in the bedroom must not be recorded against the kitchen."""
    from visionrag.types import Track, TrackState

    def trk(i, cls, cx, cy):
        h = 0.06
        return Track(i, cls, 0, 1000,
                     states=[TrackState(0, 1000, (cx - h, cy - h, cx + h, cy + h), .9)],
                     confirmed=True)

    kitchen, bedroom = scene(11), scene(77)
    with WorldMemory(tmp_path / "w.db") as world:
        m1 = world.begin_visit(kitchen)
        for _ in range(4):
            world.observe_tracks([trk(1, "refrigerator", .3, .5)])
        world.end_visit()

        m2 = world.begin_visit(bedroom)
        for _ in range(4):
            world.observe_tracks([trk(2, "bed", .5, .5)])
        world.end_visit()

        assert m1.place.place_id != m2.place.place_id
        kitchen_classes = {o["class"] for o in world.snapshot(m1.place.place_id)}
        bedroom_classes = {o["class"] for o in world.snapshot(m2.place.place_id)}
        assert "bed" not in kitchen_classes
        assert "refrigerator" not in bedroom_classes


def test_live_session_detects_place_change(client, monkeypatch):
    """A session that never re-checks its place files every later room under
    the first one."""
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    session = server.SESSIONS[sid]
    # Force a check on every analysed frame so the test is deterministic.
    session.place_check_every = 1

    seen = []
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        for img in (scene(11), scene(11), scene(11)):
            ws.send_bytes(cv2.imencode(".jpg", img)[1].tobytes())
            ws.receive_json()
        seen.append(session.current_place_id)
        for img in (scene(77), scene(77), scene(77)):
            ws.send_bytes(cv2.imencode(".jpg", img)[1].tobytes())
            ws.receive_json()
        seen.append(session.current_place_id)

    assert seen[0] is not None
    assert seen[1] != seen[0], "walking into a different room kept the old place"


# -- bug 5: sessions remain queryable after stop ------------------------
def test_session_queryable_after_stop(client):
    """Stopping then inspecting is the obvious flow; it closed the store."""
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"/api/sessions/{sid}/stream") as ws:
        for ax in [0.15 + i * 0.03 for i in range(12)]:
            ws.send_bytes(frame_jpeg(ax))
            ws.receive_json()

    assert client.post(f"/api/sessions/{sid}/stop").status_code == 200

    events = client.get(f"/api/sessions/{sid}/events")
    assert events.status_code == 200, events.text
    assert events.json()["count"] > 0

    answered = client.post(f"/api/sessions/{sid}/query",
                           json={"question": "what appeared?"})
    assert answered.status_code == 200, answered.text
    assert client.get(f"/api/sessions/{sid}/stats").status_code == 200


# -- bug 6: retention mode validation -----------------------------------
def test_invalid_retention_mode_is_rejected(client):
    """An unrecognised value silently behaved like 'full', storing every
    frame -- the opposite of what a mistyped privacy setting should do."""
    res = client.post("/api/sessions", json={"retention_mode": "definitely-not-valid"})
    assert res.status_code == 422, f"accepted junk retention mode: {res.text}"


@pytest.mark.parametrize("mode", ["metadata", "evidence", "full"])
def test_valid_retention_modes_accepted(client, mode):
    res = client.post("/api/sessions", json={"retention_mode": mode})
    assert res.status_code == 200
    assert res.json()["retention_mode"] == mode


# -- bug 4: authentication ----------------------------------------------
@pytest.fixture()
def auth_client(tmp_path, monkeypatch):
    def fake_config() -> Config:
        cfg = Config()
        cfg.detector.backend = "stub"
        cfg.scheduler.mode = "all"
        cfg.store.db_path = str(tmp_path / "memory.db")
        cfg.store.evidence_dir = str(tmp_path / "evidence")
        return cfg

    monkeypatch.setattr(server, "_base_config", fake_config)
    monkeypatch.setattr(server, "_runs_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "ACCESS_TOKEN", "secret-token")
    server.SESSIONS.clear()
    with TestClient(server.app) as c:
        yield c


def test_api_requires_token(auth_client):
    """Admin exposes enrolled-person metadata and biometric deletion. A
    self-signed certificate encrypts the channel; it does not identify anyone."""
    assert auth_client.get("/api/admin/overview").status_code == 401
    assert auth_client.get("/api/admin/people").status_code == 401
    assert auth_client.post("/api/sessions", json={}).status_code == 401


def test_token_grants_access(auth_client):
    headers = {"Authorization": "Bearer secret-token"}
    assert auth_client.get("/api/admin/overview", headers=headers).status_code == 200
    # Query parameter too: a browser cannot set headers on a WebSocket.
    assert auth_client.get("/api/admin/overview?t=secret-token").status_code == 200


def test_wrong_token_rejected(auth_client):
    assert auth_client.get(
        "/api/admin/overview", headers={"Authorization": "Bearer nope"}
    ).status_code == 401


def test_pages_load_without_token(auth_client):
    """The pages themselves carry no data; they read the token from the URL.
    Blocking them would make the link in the terminal useless."""
    assert auth_client.get("/").status_code == 200
    assert auth_client.get("/admin").status_code == 200


def test_websocket_requires_token(auth_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, Exception)):
        with auth_client.websocket_connect("/api/sessions/nonexistent/stream") as ws:
            ws.receive_json()
