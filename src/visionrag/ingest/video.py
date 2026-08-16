"""Frame source.

Decoding is deliberately separated from analysis: the source yields every
frame at native rate and the scheduler decides which ones cost compute. That
separation is what makes the compute-vs-accuracy sweep possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import VideoConfig


@dataclass(slots=True)
class Frame:
    frame_id: int
    ts_ms: int
    image: np.ndarray  # BGR, possibly resized
    source_size: tuple[int, int]  # (w, h) before resize


class VideoSource:
    """Reads a video file and yields `Frame`s.

    Resizing happens here rather than in the detector so that every downstream
    stage -- ego-motion included -- sees the same pixels, and the cost of
    resizing is attributed once.
    """

    def __init__(self, cfg: VideoConfig) -> None:
        if not cfg.path:
            raise ValueError("video.path is not set")
        path = Path(cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        self.cfg = cfg
        self.path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def _resize(self, img: np.ndarray) -> np.ndarray:
        long_edge = self.cfg.resize_long_edge
        if not long_edge:
            return img
        h, w = img.shape[:2]
        if max(h, w) <= long_edge:
            return img
        scale = long_edge / float(max(h, w))
        # INTER_AREA is the right choice for downscaling; INTER_LINEAR
        # introduces aliasing that shows up as spurious optical flow.
        return cv2.resize(
            img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
        )

    def __iter__(self) -> Iterator[Frame]:
        start_ms = self.cfg.start_s * 1000.0
        if start_ms > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        limit_ms = (
            start_ms + self.cfg.max_duration_s * 1000.0
            if self.cfg.max_duration_s
            else None
        )
        frame_id = 0
        while True:
            ok, img = self._cap.read()
            if not ok:
                break
            # Prefer the container timestamp; fall back to frame index when the
            # container does not report one (common with some MKV muxes).
            ts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if not ts_ms or ts_ms <= 0:
                ts_ms = (frame_id / self.fps) * 1000.0 + start_ms
            if limit_ms is not None and ts_ms > limit_ms:
                break
            yield Frame(
                frame_id=frame_id,
                ts_ms=int(round(ts_ms)),
                image=self._resize(img),
                source_size=(self.width, self.height),
            )
            frame_id += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_synthetic_video(
    path: str | Path,
    seconds: float = 12.0,
    fps: int = 15,
    size: tuple[int, int] = (640, 480),
    camera_shake: bool = False,
) -> Path:
    """Generate a deterministic test clip with known ground truth.

    Object A approaches static object B, dwells near it, then retreats. The
    trajectories are analytic, so the correct event sequence is known exactly
    and the fixture doubles as a free labelled set for the event-rule tests.

    Geometry is chosen so the two boxes never overlap at closest approach
    (centroid gap 0.12 vs. box width 0.0875 normalised). Overlapping boxes
    merge into one contour under the stub detector, which breaks association
    and would make the fixture test the tracker's occlusion behaviour by
    accident rather than the event rules on purpose.

    With `camera_shake` the whole scene translates while true object motion is
    unchanged, so every *additional* motion event is a false positive. That is
    the controlled test for the ego-motion counterfactual (claim S1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter (missing codec?)")

    n = int(seconds * fps)
    rng = np.random.default_rng(0)
    for i in range(n):
        t = i / n
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        # Static texture gives optical flow something to lock onto; without it
        # camera-motion estimation has no features and always reports failure.
        for gx in range(0, w, 40):
            cv2.line(img, (gx, 0), (gx, h), (60, 60, 60), 1)
        for gy in range(0, h, 40):
            cv2.line(img, (0, gy), (w, gy), (60, 60, 60), 1)

        # A: approaches B, dwells, then retreats. Stops at 0.58 so the
        # centroid gap to B is 0.12 -- inside `near_thresh` (0.15) but wider
        # than the boxes, so they stay visually separate.
        if t < 0.35:
            ax = 0.15 + (t / 0.35) * 0.43       # approach: 0.55 -> 0.12 apart
        elif t < 0.70:
            ax = 0.58                            # dwell ~4.2 s
        else:
            ax = 0.58 - ((t - 0.70) / 0.30) * 0.38  # retreat: back to 0.50 apart
        ay = 0.50
        # B: static reference object.
        bx, by = 0.70, 0.50

        dx = dy = 0
        if camera_shake:
            dx = int(18 * np.sin(t * 2 * np.pi * 3) + rng.normal(0, 1.5))
            dy = int(12 * np.cos(t * 2 * np.pi * 2) + rng.normal(0, 1.5))
            img = np.roll(img, (dy, dx), axis=(0, 1))

        # Colours must land inside the stub detector's hue bands: orange
        # -> OpenCV hue ~16, pure green -> ~60. Keep them far apart so a
        # small rendering change cannot make the two classes collide.
        for cx, cy, colour in ((ax, ay, (0, 140, 255)), (bx, by, (0, 220, 0))):
            px, py = int(cx * w) + dx, int(cy * h) + dy
            cv2.rectangle(img, (px - 28, py - 44), (px + 28, py + 44), colour, -1)
        writer.write(img)

    writer.release()
    return path
