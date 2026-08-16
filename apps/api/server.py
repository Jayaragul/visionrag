"""FastAPI gateway (PRD 7).

Frames arrive as binary WebSocket messages and results return as JSON. The
phone previews locally at full rate; only sampled frames cross the wire, which
is what keeps both bandwidth and CPU bounded.

Detection is blocking and CPU-bound, so every call into perception is pushed
to a worker thread. Running it on the event loop would stall the socket and
the heartbeat along with it.

    python apps/api/server.py --host 0.0.0.0 --port 8443 --cert certs/dev.pem
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "apps"))

from fastapi import (  # noqa: E402
    FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from api import admin  # noqa: E402
from api.query import answer_question  # noqa: E402
from api.session import LiveSession  # noqa: E402
from visionrag.config import Config  # noqa: E402

WEB_DIR = _ROOT / "apps" / "web"
DEFAULT_CONFIG = _ROOT / "configs" / "live.yaml"

app = FastAPI(title="visionrag live")
SESSIONS: dict[str, LiveSession] = {}

# The dashboard reads live state from the same dict rather than importing the
# session module, which would create a cycle.
admin.SESSIONS = SESSIONS
app.include_router(admin.router)


def _base_config() -> Config:
    return Config.load(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else Config()


def _runs_dir() -> Path:
    """Root for everything a live session writes.

    A function rather than a constant so tests can redirect it; otherwise
    every test run scatters databases and evidence directories through the
    repository's runs/ folder.
    """
    return _ROOT / "runs" / "live"


def _world_db_path() -> str:
    return str(_runs_dir() / "world.db")


def _get(session_id: str) -> LiveSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")
    return session


# -- static -------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
async def appjs() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/admin")
async def admin_page() -> FileResponse:
    return FileResponse(WEB_DIR / "admin.html")


@app.get("/admin.js")
async def admin_js() -> FileResponse:
    return FileResponse(WEB_DIR / "admin.js", media_type="application/javascript")


# -- sessions -----------------------------------------------------------
class CreateSession(BaseModel):
    retention_mode: str = "evidence"


@app.post("/api/sessions")
async def create_session(
    request: Request, body: CreateSession | None = None
) -> dict:
    body = body or CreateSession()
    session_id = uuid.uuid4().hex[:12]

    cfg = _base_config()
    cfg.name = f"live-{session_id}"
    cfg.store.retention_mode = body.retention_mode
    runs = _runs_dir()
    cfg.store.evidence_dir = str(runs / session_id / "evidence")
    cfg.store.db_path = str(runs / "memory.db")

    # Truncated: enough to tell an iPhone from a laptop in the dashboard,
    # without storing a full fingerprinting string.
    device = (request.headers.get("user-agent") or "unknown")[:120]
    world_db = _world_db_path()
    session = await run_in_threadpool(
        LiveSession, session_id, cfg, device, world_db
    )
    SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "run_id": session.run_id,
        "retention_mode": cfg.store.retention_mode,
        # The client paces itself from this, so the server stays the single
        # source of truth for how much work it is willing to accept.
        "target_fps": cfg.scheduler.max_fps
        if cfg.scheduler.mode == "adaptive"
        else cfg.scheduler.fixed_fps,
        "detector": cfg.detector.backend,
        "world_memory": True,
    }


@app.websocket("/api/sessions/{session_id}/stream")
async def stream(websocket: WebSocket, session_id: str) -> None:
    session = SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    try:
        while True:
            blob = await websocket.receive_bytes()
            try:
                result = await run_in_threadpool(session.process_jpeg, blob)
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})
                continue
            # A skipped frame still gets a reply: the client holds exactly one
            # frame in flight and waits for a response before sending the
            # next, so silence here would stall the stream permanently.
            await websocket.send_json(result or {"skipped": True})
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/sessions/{session_id}/events")
async def get_events(
    session_id: str, min_confidence: float = 0.0, limit: int = 200
) -> dict:
    session = _get(session_id)
    events = await run_in_threadpool(
        session.store.timeline, session.run_id, None, None, None, None,
        min_confidence, limit,
    )
    return {"session_id": session_id, "count": len(events), "events": events}


class Query(BaseModel):
    question: str
    min_confidence: float = 0.3


@app.post("/api/sessions/{session_id}/query")
async def query(session_id: str, body: Query) -> dict:
    session = _get(session_id)
    return await run_in_threadpool(
        answer_question,
        session.store,
        body.question,
        session.run_id,
        int(session.elapsed_s * 1000),
        body.min_confidence,
    )


@app.get("/api/sessions/{session_id}/stats")
async def stats(session_id: str) -> dict:
    return _get(session_id).stats()


@app.get("/api/sessions/{session_id}/evidence/{frame_id}")
async def evidence(session_id: str, frame_id: int):
    session = _get(session_id)
    row = session.store.conn.execute(
        "SELECT evidence_uri FROM frames WHERE run_id = ? AND frame_id = ?",
        (session.run_id, frame_id),
    ).fetchone()
    if row is None or not row["evidence_uri"]:
        raise HTTPException(status_code=404, detail="no evidence stored for that frame")
    path = Path(row["evidence_uri"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="evidence file was deleted")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/sessions/{session_id}/stop")
async def stop(session_id: str) -> dict:
    session = _get(session_id)
    return await run_in_threadpool(session.close)


@app.delete("/api/sessions/{session_id}")
async def delete(session_id: str) -> JSONResponse:
    session = _get(session_id)
    await run_in_threadpool(session.delete)
    SESSIONS.pop(session_id, None)
    return JSONResponse({"deleted": session_id})


def main() -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert", help="TLS cert PEM (required for phone camera)")
    ap.add_argument("--key", help="TLS key PEM; defaults to --cert")
    args = ap.parse_args()

    kwargs: dict = {"host": args.host, "port": args.port}
    if args.cert:
        kwargs["ssl_certfile"] = args.cert
        kwargs["ssl_keyfile"] = args.key or args.cert
        scheme = "https"
    else:
        scheme = "http"
        print(
            "WARNING: no --cert given. Phone browsers refuse camera access "
            "over plain http to a LAN address; only localhost will work.\n"
            "         Generate one with: python scripts/make_cert.py"
        )
    print(f"serving on {scheme}://{args.host}:{args.port}")
    uvicorn.run(app, **kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
