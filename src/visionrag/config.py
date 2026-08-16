"""Run configuration.

Every experiment is fully described by one config object. The config is
hashed and stored alongside results so a number in the paper can always be
traced back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    # "stub" needs no model download and is what the tests use, so CI never
    # depends on network or on a specific checkpoint.
    backend: str = Field("torchvision", description="stub | torchvision | onnx")
    model: str = "ssdlite320_mobilenet_v3_large"
    onnx_path: str | None = None
    input_size: int = 320
    score_thresh: float = 0.40
    nms_iou: float = 0.50
    max_detections: int = 50
    # Primary knob for sweeping the compute axis in E1. Pinned rather than
    # left to the runtime default, which varies by machine and would make
    # CPU-second numbers non-comparable across runs.
    num_threads: int = 4
    classes: list[str] | None = None  # None = keep all classes


class TrackerConfig(BaseModel):
    iou_thresh: float = 0.30
    max_misses: int = 8  # frames a track survives without a match
    min_hits: int = 3  # matches before a track is confirmed and can emit events
    max_tracks: int = 200


class EgoMotionConfig(BaseModel):
    enabled: bool = True
    max_features: int = 300
    quality: float = 0.01
    min_distance: int = 8
    ransac_thresh: float = 3.0
    # Below this normalised translation the camera is treated as static.
    static_thresh: float = 0.002
    # Ego-motion counterfactual (claim S1): if removing estimated camera
    # motion drops an object's apparent speed below this fraction of its raw
    # speed, camera motion plausibly explains the motion and the event is
    # flagged rather than asserted.
    counterfactual_ratio: float = 0.35


class SchedulerConfig(BaseModel):
    mode: str = Field("adaptive", description="all | fixed | adaptive")
    fixed_fps: float = 2.0
    min_fps: float = 1.0
    max_fps: float = 5.0
    burst_fps: float = 8.0
    # Mean absolute frame difference above which the scene counts as moving.
    motion_thresh: float = 2.5
    burst_frames: int = 15  # burst duration after a new object appears


class EventConfig(BaseModel):
    near_thresh: float = 0.15  # normalised centroid distance
    far_thresh: float = 0.25  # hysteresis: leaving "near" needs a larger gap
    approach_window_s: float = 1.5
    approach_min_delta: float = 0.04  # required distance decrease over window
    stop_speed_thresh: float = 0.02  # normalised units/sec
    stop_min_duration_s: float = 0.8
    remained_near_min_s: float = 2.0
    min_confidence: float = 0.30  # events below this are not persisted


class StoreConfig(BaseModel):
    db_path: str = "runs/memory.db"
    evidence_dir: str = "runs/evidence"
    # metadata | evidence | full  (PRD 9)
    retention_mode: str = "evidence"
    evidence_jpeg_quality: int = 80


class VideoConfig(BaseModel):
    path: str | None = None
    start_s: float = 0.0
    max_duration_s: float | None = None
    # Frames are decoded at source rate; the scheduler decides what to analyse.
    resize_long_edge: int | None = 640


class Config(BaseModel):
    name: str = "default"
    seed: int = 0
    video: VideoConfig = VideoConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    egomotion: EgoMotionConfig = EgoMotionConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    events: EventConfig = EventConfig()
    store: StoreConfig = StoreConfig()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def fingerprint(self) -> str:
        """Stable hash of the full config. Two runs with the same fingerprint
        used identical settings."""
        blob = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def git_commit() -> str | None:
    """Recorded with every run so results trace to code (see PAPER_PLAN 8)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return None
