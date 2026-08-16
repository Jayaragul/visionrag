"""Ingest orchestration.

Wires the stages together and attributes compute to each one. The ordering
matters: ego-motion is estimated *before* tracking so track velocities can be
compensated in the same pass, and events are induced *after* tracking so they
see settled identities.

Cost is measured per stage from the start, because "which stage dominates" is
a result the paper needs, not a debugging convenience.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .config import Config, git_commit
from .cost import CostLedger, hardware_fingerprint, latency_report
from .ingest.detect import build_detector
from .ingest.egomotion import EgoMotionEstimator
from .ingest.scheduler import FrameScheduler
from .ingest.track import Tracker
from .ingest.video import Frame, VideoSource
from .memory.events import EventInducer
from .memory.store import MemoryStore
from .types import Detection, FrameResult


class IngestPipeline:
    def __init__(self, cfg: Config, store: MemoryStore | None = None) -> None:
        self.cfg = cfg
        self.cost = CostLedger()
        with self.cost.stage("detector_init"):
            self.detector = build_detector(cfg.detector)
        self.tracker = Tracker(cfg.tracker)
        self.ego = EgoMotionEstimator(cfg.egomotion)
        self.scheduler = FrameScheduler(cfg.scheduler)
        self.events = EventInducer(cfg.events, cfg.egomotion)
        self.store = store
        self._latencies: list[float] = []
        self._evidence_dir = Path(cfg.store.evidence_dir)

    # -- per frame --------------------------------------------------------
    def process_frame(self, frame: Frame) -> FrameResult | None:
        """Returns None when the scheduler skipped this frame."""
        import time

        t0 = time.perf_counter()

        with self.cost.stage("schedule"):
            if not self.scheduler.should_detect(frame.ts_ms, frame.image):
                return None

        with self.cost.stage("egomotion"):
            camera = self.ego.estimate(frame.image)

        with self.cost.stage("detect"):
            detections: list[Detection] = self.detector.detect(frame.image)

        with self.cost.stage("track"):
            newly_confirmed, lost = self.tracker.update(
                detections, frame.frame_id, frame.ts_ms, camera
            )
            active = self.tracker.active()

        # A new confirmed object is exactly when extra frames are worth
        # spending -- characterise it while it is still in view.
        if newly_confirmed:
            self.scheduler.trigger_burst()

        with self.cost.stage("events"):
            events = self.events.process(
                active, newly_confirmed, lost, frame.frame_id, frame.ts_ms, camera
            )

        with self.cost.stage("persist"):
            if self.store is not None:
                uri = self._save_evidence(frame, bool(events))
                self.store.add_frame(frame.frame_id, frame.ts_ms, True, uri)
                self.store.add_observations(active)
                self.store.upsert_tracks(active)
                self.store.add_events(events)

        self._latencies.append(time.perf_counter() - t0)
        return FrameResult(
            frame_id=frame.frame_id,
            ts_ms=frame.ts_ms,
            detections=detections,
            tracks=active,
            events=events,
            camera=camera,
            detector_ran=True,
        )

    def _save_evidence(self, frame: Frame, has_events: bool) -> str | None:
        """Retention policy (PRD 9). `evidence` mode keeps pixels only for
        frames that support an event, which is what makes evidence citation
        possible without storing the whole session."""
        mode = self.cfg.store.retention_mode
        if mode == "metadata":
            return None
        if mode == "evidence" and not has_events:
            return None
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self._evidence_dir / f"frame_{frame.frame_id:07d}.jpg"
        cv2.imwrite(
            str(path),
            frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.store.evidence_jpeg_quality],
        )
        return str(path)

    # -- session lifecycle ------------------------------------------------
    # Split out of run() so a live phone session can drive process_frame()
    # directly without a VideoSource. File ingest and live ingest then share
    # exactly one perception implementation.
    def begin_session(
        self, video_path: str | None = None, duration_s: float | None = None
    ) -> int | None:
        if self.store is None:
            return None
        return self.store.start_run(
            config=self.cfg,
            git_commit=git_commit(),
            hardware=hardware_fingerprint(),
            video_path=video_path,
            video_duration_s=duration_s,
        )

    def end_session(self, duration_s: float | None = None) -> dict:
        summary = self.stats(duration_s)
        if self.store is not None:
            self.store.commit()
            self.store.finish_run(summary["cost"])
        return summary

    def stats(self, duration_s: float | None = None) -> dict:
        return {
            "scheduler": self.scheduler.stats(),
            "tracks": len(self.tracker.tracks),
            "cost": self.cost.summary(duration_s),
            "latency": latency_report(self._latencies),
            "config_hash": self.cfg.fingerprint(),
            "git_commit": git_commit(),
            "hardware": hardware_fingerprint(),
        }

    # -- whole video ------------------------------------------------------
    def run(self, progress: bool = False) -> dict:
        source = VideoSource(self.cfg.video)
        self.begin_session(str(source.path), source.duration_s)

        n_events = 0
        try:
            for frame in source:
                result = self.process_frame(frame)
                if result:
                    n_events += len(result.events)
                if progress and frame.frame_id % 60 == 0:
                    print(
                        f"  frame {frame.frame_id:5d}  "
                        f"t={frame.ts_ms / 1000:6.2f}s  events={n_events}",
                        flush=True,
                    )
        finally:
            source.close()

        summary = self.end_session(source.duration_s)
        summary["video"] = {
            "path": str(source.path),
            "duration_s": round(source.duration_s, 3),
            "fps": round(source.fps, 3),
            "frames": source.frame_count,
            "size": [source.width, source.height],
        }
        summary["events"] = n_events
        return summary
