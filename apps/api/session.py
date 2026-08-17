"""Live session state.

Wraps one `IngestPipeline` so phone frames go through exactly the same
perception path as file ingest. There is no separate "live pipeline" -- that
was the point of keeping `process_frame()` independent of `VideoSource`.

Concurrency: each session holds a lock and is processed on a worker thread.
Detection is CPU-bound and blocking, so it must never run on the event loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from visionrag.config import Config
from visionrag.ingest.quality import assess
from visionrag.ingest.video import Frame
from visionrag.memory.store import MemoryStore
from visionrag.memory.world import WorldMemory
from visionrag.pipeline import IngestPipeline
from visionrag.types import FrameResult


# Sentinel for "no candidate pending". Cannot be None, because None is itself
# a meaningful candidate: it means "this view matches no known place".
_NO_PENDING = object()


def _serialise(result: FrameResult, latency_ms: float) -> dict:
    """Wire format for the phone (PRD 7.1). Boxes are normalised xywh so the
    client can scale them to whatever size it is rendering at."""
    return {
        "frame_id": result.frame_id,
        "timestamp_ms": result.ts_ms,
        "latency_ms": round(latency_ms, 1),
        "detections": [
            {
                "track_id": t.track_id,
                "class": t.cls,
                "confidence": round(t.last.score, 3),
                "box": {
                    "x": round(t.last.box[0], 4),
                    "y": round(t.last.box[1], 4),
                    "w": round(t.last.box[2] - t.last.box[0], 4),
                    "h": round(t.last.box[3] - t.last.box[1], 4),
                },
            }
            for t in result.tracks
        ],
        "events": [
            {
                "type": e.type.value,
                "t_start_ms": e.t_start_ms,
                "t_end_ms": e.t_end_ms,
                "participants": e.participants,
                "confidence": round(e.confidence, 3),
                "ego_suspect": e.ego_suspect,
                "attrs": e.attrs,
            }
            for e in result.events
        ],
        "camera": (
            {
                "moving": result.camera.magnitude >= 0.002,
                "magnitude": round(result.camera.magnitude, 5),
            }
            if result.camera and result.camera.ok
            else None
        ),
    }


class LiveSession:
    def __init__(
        self,
        session_id: str,
        cfg: Config,
        device: str | None = None,
        world_db: str | None = None,
    ) -> None:
        self.id = session_id
        self.cfg = cfg
        self.device = device
        self.store = MemoryStore(cfg.store.db_path)
        self.pipeline = IngestPipeline(cfg, self.store)
        self.run_id = self.pipeline.begin_session(video_path=f"live:{session_id}")
        self.started_at = time.time()
        self.lock = threading.Lock()
        self._seq = 0
        self._skipped = 0
        self.closed = False

        # Persistent spatial memory. Opened lazily on the first analysed frame,
        # because recognising a place needs real pixels.
        self.world = WorldMemory(world_db) if world_db else None
        self.current_place_id: int | None = None
        self.place_label: str | None = None
        self.place_is_new = False
        self._visit_open = False
        self.last_quality: dict | None = None
        self._rejected_frames = 0
        # How often to re-run place recognition, in analysed frames. At ~3 fps
        # that is roughly every three seconds: fast enough to catch someone
        # walking between rooms, cheap enough to be irrelevant to the budget.
        self.place_check_every = 8
        self._frames_since_place_check = 0
        self._pending_place: object = _NO_PENDING
        self.place_changes = 0

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def process_jpeg(self, blob: bytes) -> dict | None:
        """Decode one JPEG and run it through perception.

        Returns None when the scheduler declined the frame, which the client
        treats as a cheap no-op rather than an error.
        """
        t0 = time.perf_counter()
        with self.lock:
            if self.closed:
                return None
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("could not decode JPEG frame")

            # Quality is judged BEFORE anything is detected or written. The
            # pipeline persists frames, observations and events inside
            # process_frame(), so assessing afterwards could not un-write
            # them -- a blurred frame had already become permanent evidence,
            # contradicting the product's central promise. Rejecting here also
            # saves the detector's ~190 ms on a frame that could not support a
            # conclusion anyway.
            quality = assess(img)
            self.last_quality = quality.as_dict()
            if not quality.usable:
                self._rejected_frames += 1
                self._seq += 1
                return {
                    "frame_id": self._seq - 1,
                    "timestamp_ms": int(self.elapsed_s * 1000),
                    "rejected": True,
                    "quality": self.last_quality,
                    "reasons": list(quality.reasons),
                    "detections": [],
                    "events": [],
                }

            # Live timestamps come from the wall clock, not a frame index: the
            # client's send rate is variable, and event rules reason in
            # seconds. A frame-index clock would make dwell times wrong.
            ts_ms = int(self.elapsed_s * 1000)
            frame = Frame(
                frame_id=self._seq,
                ts_ms=ts_ms,
                image=img,
                source_size=(img.shape[1], img.shape[0]),
            )
            self._seq += 1
            result = self.pipeline.process_frame(frame)
            if result is None:
                self._skipped += 1
                return None

            payload = _serialise(result, (time.perf_counter() - t0) * 1000)
            payload["quality"] = self.last_quality
            if self.world is not None:
                payload["place"] = self._update_world(img, result)
            self.store.commit()
        return payload

    def _update_world(self, image, result: FrameResult) -> dict:
        """Recognise the place and record what is visible in it.

        The place is re-checked periodically, not resolved once and trusted
        forever. A session that matched on its first frame and never looked
        again filed every later room under the first one -- walk from the
        kitchen to the bedroom and the bed became a kitchen object.

        A re-check costs a descriptor plus a shortlist match (~5 ms), so it
        runs every `place_check_every` analysed frames rather than on all.
        """
        if not self._visit_open:
            match = self.world.begin_visit(image, run_id=self.run_id)
            self._visit_open = True
            self.current_place_id = match.place.place_id
            self.place_label = match.place.label
            self.place_is_new = match.is_new
        else:
            self._frames_since_place_check += 1
            if self._frames_since_place_check >= self.place_check_every:
                self._frames_since_place_check = 0
                self._maybe_switch_place(image)

        moving = {
            t.track_id
            for t in result.tracks
            if (t.last.v_compensated[0] ** 2 + t.last.v_compensated[1] ** 2) ** 0.5
            >= self.cfg.events.stop_speed_thresh
        }
        self.world.observe_tracks(result.tracks, moving_ids=moving)
        return {
            "place_id": self.current_place_id,
            "label": self.place_label,
            "is_new": self.place_is_new,
        }

    def _maybe_switch_place(self, image) -> None:
        """Close the current visit and open a new one if the room changed.

        Two cases count as "moved", and missing the second was the whole bug:

        * the view matches a *different* known place, and
        * the view matches **nothing** -- which is exactly what walking into a
          room the system has never seen looks like. Treating no-match as
          "stay put" meant a brand-new room silently inherited the previous
          room's identity.

        A switch needs the same verdict on two consecutive checks. One
        badly-timed frame -- a hand across the lens, an odd angle, a doorway
        halfway between two rooms -- should not tear a visit in half, and a
        visit split wrongly costs a spurious round of absence evidence for
        everything in the room.
        """
        match = self.world.index.match(image)
        candidate = match.place.place_id if match.place is not None else None

        if candidate == self.current_place_id:
            self._pending_place = _NO_PENDING
            return
        if candidate != self._pending_place:
            self._pending_place = candidate
            return

        self._pending_place = _NO_PENDING
        # Closing the old visit is what records absence for it. Skipping that
        # would leave the previous room with a visit that never resolves and
        # never contributes evidence that anything was removed.
        self.world.end_visit()
        new_match = self.world.begin_visit(image, run_id=self.run_id)
        self.current_place_id = new_match.place.place_id
        self.place_label = new_match.place.label
        self.place_is_new = new_match.is_new
        self.place_changes += 1
        # Tracks belong to the room they were seen in. Carrying them over
        # would attach the kitchen's objects to the bedroom's instances.
        self.pipeline.tracker.reset()

    def stats(self) -> dict:
        s = self.pipeline.stats(self.elapsed_s)
        s["session_id"] = self.id
        s["run_id"] = self.run_id
        s["elapsed_s"] = round(self.elapsed_s, 1)
        s["frames_received"] = self._seq
        s["frames_skipped"] = self._skipped
        s["frames_rejected_quality"] = self._rejected_frames
        s["device"] = self.device
        s["place_id"] = self.current_place_id
        s["quality"] = self.last_quality
        s["place_changes"] = self.place_changes
        return s

    def close(self) -> dict:
        with self.lock:
            if self.closed:
                return {}
            self.closed = True
            summary = self.pipeline.end_session(self.elapsed_s)
            if self.world is not None:
                # end_visit is idempotent, so an explicit stop followed by a
                # socket disconnect cannot double-count absence evidence.
                summary["visit"] = self.world.end_visit()
                self.world.close()
                self.world = None
            # The store stays open. Stopping a session and then asking what it
            # saw is the obvious next thing to do, and closing here made every
            # later events/query/evidence call fail. It is closed in delete(),
            # which is when the data is actually going away.
            self.store.commit()
            return summary

    def delete(self) -> None:
        """Remove all stored data for this session (PRD 9).

        Evidence files are deleted before the DB rows, so a crash in between
        leaves rows pointing at missing files rather than orphaned images with
        no record that they exist.
        """
        with self.lock:
            if not self.closed:
                self.closed = True
                try:
                    self.pipeline.end_session(self.elapsed_s)
                except Exception:
                    pass
            if self.world is not None:
                # Close the visit before dropping the handle, or the place is
                # left with an open visit row that never resolves.
                try:
                    self.world.end_visit()
                    self.world.close()
                except Exception:
                    pass
                self.world = None
            evidence_dir = Path(self.cfg.store.evidence_dir)
            if evidence_dir.exists():
                for f in evidence_dir.glob("*.jpg"):
                    f.unlink(missing_ok=True)
            if self.run_id is not None:
                try:
                    self.store.conn.execute(
                        "DELETE FROM runs WHERE id = ?", (self.run_id,)
                    )
                    self.store.conn.commit()
                except Exception:
                    pass
            try:
                self.store.close()
            except Exception:
                pass
