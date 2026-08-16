"""Structured memory (PRD 5).

Exact facts live in SQLite; embeddings are added later for semantic recall.
The split is deliberate and is part of the argument: "when did X happen" and
"which track was near which" are relational queries that a vector index
answers badly, while "something about a bag near a door" is what embeddings
are for. Collapsing both into one vector store is the failure mode this design
avoids.

Every run also records its config fingerprint, git commit and hardware, so any
number in the paper traces back to the code and settings that produced it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..types import SCHEMA_VERSION, Event, Track

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    schema_version  INTEGER NOT NULL,
    config_name     TEXT,
    config_hash     TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    git_commit      TEXT,
    hardware_json   TEXT,
    video_path      TEXT,
    video_duration_s REAL,
    cost_json       TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    frame_id     INTEGER NOT NULL,
    ts_ms        INTEGER NOT NULL,
    detector_ran INTEGER NOT NULL,
    evidence_uri TEXT,
    PRIMARY KEY (run_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(run_id, ts_ms);

CREATE TABLE IF NOT EXISTS observations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL,
    ts_ms    INTEGER NOT NULL,
    track_id INTEGER,
    cls      TEXT NOT NULL,
    score    REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    vx_raw REAL, vy_raw REAL,
    vx_comp REAL, vy_comp REAL
);
CREATE INDEX IF NOT EXISTS idx_obs_track ON observations(run_id, track_id, ts_ms);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(run_id, ts_ms);

CREATE TABLE IF NOT EXISTS tracks (
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    track_id      INTEGER NOT NULL,
    cls           TEXT NOT NULL,
    first_seen_ms INTEGER NOT NULL,
    last_seen_ms  INTEGER NOT NULL,
    n_states      INTEGER NOT NULL,
    PRIMARY KEY (run_id, track_id)
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    t_start_ms   INTEGER NOT NULL,
    t_end_ms     INTEGER NOT NULL,
    participants TEXT NOT NULL,   -- JSON array of track_ids
    confidence   REAL NOT NULL,
    ego_suspect  INTEGER NOT NULL DEFAULT 0,
    evidence     TEXT NOT NULL,   -- JSON array of frame_ids
    attrs        TEXT NOT NULL    -- JSON object
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(run_id, t_start_ms);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, type);

CREATE TABLE IF NOT EXISTS regions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label    TEXT NOT NULL,
    polygon  TEXT NOT NULL,       -- JSON array of [x, y] pairs, normalised
    stability REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,    -- 'event' | 'track' | 'frame'
    source_id   INTEGER NOT NULL,
    model_id    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings(run_id, source_type);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.run_id: int | None = None

    # -- lifecycle --------------------------------------------------------
    def start_run(
        self,
        config: Any,
        git_commit: str | None,
        hardware: dict,
        video_path: str | None,
        video_duration_s: float | None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO runs
               (schema_version, config_name, config_hash, config_json,
                git_commit, hardware_json, video_path, video_duration_s)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                SCHEMA_VERSION,
                getattr(config, "name", None),
                config.fingerprint(),
                json.dumps(config.model_dump(), sort_keys=True),
                git_commit,
                json.dumps(hardware),
                video_path,
                video_duration_s,
            ),
        )
        self.conn.commit()
        self.run_id = int(cur.lastrowid)
        return self.run_id

    def finish_run(self, cost_summary: dict) -> None:
        self.conn.execute(
            "UPDATE runs SET cost_json = ? WHERE id = ?",
            (json.dumps(cost_summary), self._run()),
        )
        self.conn.commit()

    def _run(self) -> int:
        if self.run_id is None:
            raise RuntimeError("start_run() must be called first")
        return self.run_id

    # -- writes -----------------------------------------------------------
    def add_frame(
        self, frame_id: int, ts_ms: int, detector_ran: bool, evidence_uri: str | None
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO frames
               (run_id, frame_id, ts_ms, detector_ran, evidence_uri)
               VALUES (?,?,?,?,?)""",
            (self._run(), frame_id, ts_ms, int(detector_ran), evidence_uri),
        )

    def add_observations(self, tracks: Iterable[Track]) -> None:
        rows = []
        for t in tracks:
            s = t.last
            rows.append(
                (
                    self._run(), s.frame_id, s.ts_ms, t.track_id, t.cls, s.score,
                    *s.box, *s.v_raw, *s.v_compensated,
                )
            )
        if rows:
            self.conn.executemany(
                """INSERT INTO observations
                   (run_id, frame_id, ts_ms, track_id, cls, score,
                    x1, y1, x2, y2, vx_raw, vy_raw, vx_comp, vy_comp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def upsert_tracks(self, tracks: Iterable[Track]) -> None:
        rows = [
            (self._run(), t.track_id, t.cls, t.first_seen_ms, t.last_seen_ms, len(t.states))
            for t in tracks
        ]
        if rows:
            self.conn.executemany(
                """INSERT INTO tracks
                   (run_id, track_id, cls, first_seen_ms, last_seen_ms, n_states)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(run_id, track_id) DO UPDATE SET
                     last_seen_ms = excluded.last_seen_ms,
                     n_states     = excluded.n_states""",
                rows,
            )

    def add_events(self, events: Iterable[Event]) -> None:
        rows = [
            (
                self._run(), e.type.value, e.t_start_ms, e.t_end_ms,
                json.dumps(e.participants), e.confidence, int(e.ego_suspect),
                json.dumps(e.evidence_frames), json.dumps(e.attrs),
            )
            for e in events
        ]
        if rows:
            self.conn.executemany(
                """INSERT INTO events
                   (run_id, type, t_start_ms, t_end_ms, participants,
                    confidence, ego_suspect, evidence, attrs)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def commit(self) -> None:
        self.conn.commit()

    # -- reads ------------------------------------------------------------
    def timeline(
        self,
        run_id: int | None = None,
        t_start_ms: int | None = None,
        t_end_ms: int | None = None,
        types: list[str] | None = None,
        track_id: int | None = None,
        min_confidence: float = 0.0,
        limit: int = 200,
    ) -> list[dict]:
        """Structured retrieval. This is the half a vector index cannot do:
        exact time bounds, exact participants, exact types."""
        sql = ["SELECT * FROM events WHERE run_id = ? AND confidence >= ?"]
        args: list[Any] = [run_id or self._run(), min_confidence]
        if t_start_ms is not None:
            sql.append("AND t_end_ms >= ?")
            args.append(t_start_ms)
        if t_end_ms is not None:
            sql.append("AND t_start_ms <= ?")
            args.append(t_end_ms)
        if types:
            sql.append(f"AND type IN ({','.join('?' * len(types))})")
            args.extend(types)
        sql.append("ORDER BY t_start_ms ASC LIMIT ?")
        args.append(limit)

        rows = self.conn.execute(" ".join(sql), args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["participants"] = json.loads(d["participants"])
            d["evidence"] = json.loads(d["evidence"])
            d["attrs"] = json.loads(d["attrs"])
            d["ego_suspect"] = bool(d["ego_suspect"])
            if track_id is not None and track_id not in d["participants"]:
                continue
            out.append(d)
        return out

    def run_summary(self, run_id: int | None = None) -> dict:
        rid = run_id or self._run()
        run = self.conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
        if run is None:
            raise ValueError(f"no such run: {rid}")
        counts = {
            name: self.conn.execute(
                f"SELECT COUNT(*) FROM {name} WHERE run_id = ?", (rid,)
            ).fetchone()[0]
            for name in ("frames", "observations", "tracks", "events")
        }
        by_type = {
            r["type"]: r["n"]
            for r in self.conn.execute(
                "SELECT type, COUNT(*) n FROM events WHERE run_id = ? "
                "GROUP BY type ORDER BY n DESC",
                (rid,),
            )
        }
        ego = self.conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ? AND ego_suspect = 1", (rid,)
        ).fetchone()[0]
        return {
            "run_id": rid,
            "config_hash": run["config_hash"],
            "git_commit": run["git_commit"],
            "video_path": run["video_path"],
            "video_duration_s": run["video_duration_s"],
            "counts": counts,
            "events_by_type": by_type,
            "ego_suspect_events": ego,
            "cost": json.loads(run["cost_json"]) if run["cost_json"] else None,
        }

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
