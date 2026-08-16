"""Admin dashboard API.

Read-mostly view over everything the system holds: connected devices, places
mapped, objects remembered, events recorded, storage consumed, and enrolled
people.

Deliberately separated from the capture API. The capture path must stay fast
and narrow; this one runs arbitrary aggregate queries and is only used by an
operator looking at a screen.

Note on the numbers shown: they come from whatever databases exist on disk.
An empty dashboard means nothing has been recorded yet, not that recording is
broken.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionrag.memory.persistence import ObservationStatus, SemanticKind, Tier

router = APIRouter(prefix="/api/admin", tags=["admin"])

_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = _ROOT / "runs"
WORLD_DB = RUNS_DIR / "live" / "world.db"
EVENT_DB = RUNS_DIR / "live" / "memory.db"
FACE_DB = RUNS_DIR / "live" / "faces.db"

# Populated by server.py so the dashboard can see live state without importing
# the session module and creating a cycle.
SESSIONS: dict = {}


def _connect(path: Path) -> sqlite3.Connection | None:
    """Open read-only. A missing database is a normal state, not an error --
    the dashboard should render on a fresh install."""
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _dir_size(path: Path) -> tuple[int, int]:
    """(bytes, file count) for a directory tree."""
    if not path.exists():
        return 0, 0
    total = count = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            count += 1
    return total, count


def _db_size(path: Path) -> int:
    """SQLite in WAL mode keeps data in -wal until checkpoint, so the main
    file alone understates real usage."""
    if not path.exists():
        return 0
    total = path.stat().st_size
    for suffix in ("-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            total += side.stat().st_size
    return total


# -- overview -----------------------------------------------------------
@router.get("/overview")
async def overview() -> dict:
    places = objects = visits = events = 0
    tiers: dict[str, int] = {}
    kinds: dict[str, int] = {}

    world = _connect(WORLD_DB)
    if world:
        if _table_exists(world, "places"):
            places = world.execute("SELECT COUNT(*) FROM places").fetchone()[0]
        if _table_exists(world, "object_instances"):
            objects = world.execute(
                "SELECT COUNT(*) FROM object_instances WHERE state != 'tentative'"
            ).fetchone()[0]
            tiers = {
                r["tier"]: r["n"]
                for r in world.execute(
                    "SELECT tier, COUNT(*) n FROM object_instances "
                    "WHERE state != 'tentative' GROUP BY tier"
                )
            }
            kinds = {
                r["semantic_kind"]: r["n"]
                for r in world.execute(
                    "SELECT semantic_kind, COUNT(*) n FROM object_instances "
                    "WHERE state != 'tentative' GROUP BY semantic_kind"
                )
            }
        if _table_exists(world, "place_visits"):
            visits = world.execute("SELECT COUNT(*) FROM place_visits").fetchone()[0]
        world.close()

    ev = _connect(EVENT_DB)
    if ev and _table_exists(ev, "events"):
        events = ev.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if ev:
        ev.close()

    enrolled = 0
    faces = _connect(FACE_DB)
    if faces and _table_exists(faces, "people"):
        enrolled = faces.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    if faces:
        faces.close()

    live = [s for s in SESSIONS.values() if not s.closed]
    return {
        "devices_connected": len(live),
        "sessions_total": len(SESSIONS),
        "places_mapped": places,
        "objects_remembered": objects,
        "visits_recorded": visits,
        "events_recorded": events,
        "people_enrolled": enrolled,
        "objects_by_tier": tiers,
        "objects_by_kind": kinds,
        "world_memory_active": WORLD_DB.exists(),
    }


# -- devices ------------------------------------------------------------
@router.get("/devices")
async def devices() -> dict:
    out = []
    for session in SESSIONS.values():
        stats = session.stats()
        cost = stats.get("cost", {})
        latency = stats.get("latency", {})
        out.append(
            {
                "session_id": session.id,
                "device": getattr(session, "device", None) or "unknown",
                "connected": not session.closed,
                "elapsed_s": stats.get("elapsed_s"),
                "frames_received": stats.get("frames_received"),
                "frames_analysed": stats.get("scheduler", {}).get("frames_detected"),
                "detection_ratio": stats.get("scheduler", {}).get("detection_ratio"),
                "latency_p50_ms": latency.get("p50_ms"),
                "latency_p95_ms": latency.get("p95_ms"),
                "cpu_s": cost.get("total_cpu_s"),
                "place_id": getattr(session, "current_place_id", None),
                "quality": getattr(session, "last_quality", None),
                "retention_mode": session.cfg.store.retention_mode,
            }
        )
    out.sort(key=lambda d: (not d["connected"], d["session_id"]))
    return {"count": len(out), "devices": out}


# -- places -------------------------------------------------------------
@router.get("/places")
async def places() -> dict:
    world = _connect(WORLD_DB)
    if not world or not _table_exists(world, "places"):
        if world:
            world.close()
        return {"count": 0, "places": [], "note": "no world memory recorded yet"}

    rows = world.execute(
        """SELECT p.*,
                  (SELECT COUNT(*) FROM object_instances i
                    WHERE i.place_id = p.place_id AND i.state != 'tentative')
                    AS n_objects,
                  (SELECT COUNT(*) FROM place_visits v
                    WHERE v.place_id = p.place_id) AS n_visit_rows,
                  (SELECT AVG(coverage) FROM place_visits v
                    WHERE v.place_id = p.place_id AND v.ended IS NOT NULL)
                    AS avg_coverage
             FROM places p ORDER BY p.last_seen DESC"""
    ).fetchall()
    out = [
        {
            "place_id": r["place_id"],
            "label": r["label"],
            "n_visits": r["n_visits"],
            "n_visit_rows": r["n_visit_rows"],
            "n_objects": r["n_objects"],
            "avg_coverage": (
                round(r["avg_coverage"], 3) if r["avg_coverage"] is not None else None
            ),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            # Presence only -- coordinates are not surfaced in a dashboard that
            # may be left open on a screen.
            "has_gps": r["lat"] is not None,
        }
        for r in rows
    ]
    world.close()
    return {"count": len(out), "places": out}


@router.get("/places/{place_id}/objects")
async def place_objects(place_id: int) -> dict:
    world = _connect(WORLD_DB)
    if not world or not _table_exists(world, "object_instances"):
        if world:
            world.close()
        raise HTTPException(status_code=404, detail="no world memory recorded yet")

    now = time.time()
    rows = world.execute(
        "SELECT * FROM object_instances WHERE place_id = ? ORDER BY persistence DESC",
        (place_id,),
    ).fetchall()
    objects = [
        {
            "instance_id": r["id"],
            "class": r["cls"],
            "semantic_kind": r["semantic_kind"],
            "tier": r["tier"],
            "state": r["state"],
            # Stored at last visit close; recomputing needs the filter, which
            # this read-only view deliberately does not import.
            "persistence": round(r["persistence"], 4),
            "anchor": [round(r["anchor_x"], 3), round(r["anchor_y"], 3)],
            "times_seen": r["n_seen"],
            "opportunities": r["n_opportunities"],
            "frames_seen": r["frames_seen"],
            "ever_moved": bool(r["ever_moved"]),
            "last_seen": r["last_seen"],
            "age_days": round((now - r["first_seen"]) / 86400.0, 2),
        }
        for r in rows
    ]
    visits = []
    if _table_exists(world, "place_visits"):
        visits = [
            {
                "visit_id": v["id"],
                "started": v["started"],
                "ended": v["ended"],
                "coverage": round(v["coverage"], 3),
                "frames": v["frames"],
            }
            for v in world.execute(
                "SELECT * FROM place_visits WHERE place_id = ? "
                "ORDER BY started DESC LIMIT 20",
                (place_id,),
            )
        ]
    world.close()
    return {"place_id": place_id, "objects": objects, "visits": visits}


class LabelRequest(BaseModel):
    label: str


@router.post("/places/{place_id}/label")
async def label_place(place_id: int, body: LabelRequest) -> dict:
    if not WORLD_DB.exists():
        raise HTTPException(status_code=404, detail="no world memory recorded yet")
    conn = sqlite3.connect(str(WORLD_DB))
    try:
        cur = conn.execute(
            "UPDATE places SET label = ? WHERE place_id = ?",
            (body.label.strip()[:64], place_id),
        )
        conn.commit()
    finally:
        conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="no such place")
    return {"place_id": place_id, "label": body.label.strip()[:64]}


# -- events -------------------------------------------------------------
@router.get("/events")
async def recent_events(limit: int = 50) -> dict:
    conn = _connect(EVENT_DB)
    if not conn or not _table_exists(conn, "events"):
        if conn:
            conn.close()
        return {"count": 0, "events": [], "by_type": {}}

    by_type = {
        r["type"]: r["n"]
        for r in conn.execute(
            "SELECT type, COUNT(*) n FROM events GROUP BY type ORDER BY n DESC"
        )
    }
    rows = conn.execute(
        "SELECT id, run_id, type, t_start_ms, confidence, ego_suspect "
        "FROM events ORDER BY id DESC LIMIT ?",
        (min(limit, 500),),
    ).fetchall()
    events = [
        {
            "id": r["id"], "run_id": r["run_id"], "type": r["type"],
            "t_start_s": round(r["t_start_ms"] / 1000.0, 2),
            "confidence": round(r["confidence"], 3),
            "ego_suspect": bool(r["ego_suspect"]),
        }
        for r in rows
    ]
    conn.close()
    return {"count": len(events), "events": events, "by_type": by_type}


# -- storage ------------------------------------------------------------
@router.get("/storage")
async def storage() -> dict:
    evidence_bytes, evidence_files = _dir_size(RUNS_DIR / "live")
    dbs = {
        "world.db": _db_size(WORLD_DB),
        "memory.db": _db_size(EVENT_DB),
        "faces.db": _db_size(FACE_DB),
    }
    # Evidence directories sit under runs/live alongside the databases, so
    # subtract them to avoid reporting the same bytes twice.
    db_total = sum(dbs.values())
    return {
        "databases": {k: v for k, v in dbs.items() if v > 0},
        "database_bytes": db_total,
        "evidence_bytes": max(0, evidence_bytes - db_total),
        "evidence_files": evidence_files,
        "total_bytes": evidence_bytes,
        "runs_dir": str(RUNS_DIR),
    }


# -- enrolled people ----------------------------------------------------
@router.get("/people")
async def people() -> dict:
    conn = _connect(FACE_DB)
    if not conn or not _table_exists(conn, "people"):
        if conn:
            conn.close()
        return {"count": 0, "people": [], "note": "face recognition not in use"}
    rows = [
        {
            "person_id": r["person_id"], "name": r["name"],
            "consent_note": r["consent_note"],
            "enrolled_at": r["enrolled_at"], "n_templates": r["n_templates"],
        }
        for r in conn.execute("SELECT * FROM people ORDER BY name")
    ]
    conn.close()
    return {"count": len(rows), "people": rows}


@router.delete("/people/{person_id}")
async def forget_person(person_id: int) -> dict:
    """Delete a person and every biometric template of them.

    The only correct response to a withdrawal of consent, so it is exposed
    directly rather than hidden behind a config file.
    """
    if not FACE_DB.exists():
        raise HTTPException(status_code=404, detail="no face database")
    conn = sqlite3.connect(str(FACE_DB))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM face_templates WHERE person_id = ?", (person_id,))
        cur = conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
        conn.commit()
    finally:
        conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="no such person")
    return {"deleted": person_id}


@router.get("/taxonomy")
async def taxonomy() -> dict:
    """The vocabularies the UI renders, so labels stay in one place."""
    return {
        "tiers": [Tier.STATIC, Tier.SEMI_STATIC, Tier.DYNAMIC, Tier.UNKNOWN],
        "kinds": [
            SemanticKind.FIXTURE, SemanticKind.FURNITURE, SemanticKind.MOVABLE,
            SemanticKind.PERSON, SemanticKind.ANIMAL, SemanticKind.VEHICLE,
            SemanticKind.UNKNOWN,
        ],
        "observation_statuses": [
            ObservationStatus.SEEN, ObservationStatus.CHECKED_MISSING,
            ObservationStatus.OCCLUDED_UNCERTAIN, ObservationStatus.NOT_OBSERVED,
        ],
    }
