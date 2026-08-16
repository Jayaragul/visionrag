"""Adaptive frame scheduling (PRD 4.3).

Detection dominates ingest cost, so *when to detect* is the main lever on the
compute axis. The scheduler spends detections where the scene is changing and
saves them where it is not.

Three modes:

* ``all``      -- analyse every frame. The upper bound on both accuracy and
                  cost, and the reference point the sweep is measured against.
                  Also what tests use, since it makes results independent of
                  how fast frames happen to arrive.
* ``fixed``    -- detect at a constant rate. The control condition; the E1
                  sweep uses this so each point on the curve is unambiguous.
* ``adaptive`` -- rate follows scene activity, with a temporary burst after a
                  new object appears. This is the proposed method (claim S3).

Note that pacing is driven by frame timestamps, which in live mode come from
the wall clock. Feeding frames faster than real time therefore does *not*
produce a higher analysis rate in ``fixed``/``adaptive`` -- the scheduler
throttles to the configured rate in seconds of video, not frames delivered.

The motion signal is a downsampled mean absolute frame difference. It costs
well under a millisecond, which is the point: a scheduler that costs as much
as the work it saves is not a saving.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SchedulerConfig


class FrameScheduler:
    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self._prev_small: np.ndarray | None = None
        self._last_detect_ms: int | None = None
        self._burst_remaining = 0
        self._motion_ema = 0.0
        self.n_considered = 0
        self.n_detected = 0

    def motion_score(self, image: np.ndarray) -> float:
        """Mean absolute difference on a 64x64 grayscale thumbnail."""
        small = cv2.cvtColor(
            cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.int16)
        if self._prev_small is None:
            self._prev_small = small
            return float("inf")  # always analyse the first frame
        score = float(np.abs(small - self._prev_small).mean())
        self._prev_small = small
        # Smoothed so a single noisy frame does not trigger a rate change.
        self._motion_ema = 0.7 * self._motion_ema + 0.3 * score
        return score

    def trigger_burst(self) -> None:
        """Called when something notable appears; buys a short high-rate window
        so a new object is characterised while it is still visible."""
        self._burst_remaining = self.cfg.burst_frames

    def target_fps(self) -> float:
        if self.cfg.mode == "all":
            return float("inf")
        if self.cfg.mode == "fixed":
            return self.cfg.fixed_fps
        if self._burst_remaining > 0:
            return self.cfg.burst_fps
        if self._motion_ema < self.cfg.motion_thresh:
            return self.cfg.min_fps
        # Scale linearly between min and max over one threshold-width band.
        excess = (self._motion_ema - self.cfg.motion_thresh) / max(
            1e-6, self.cfg.motion_thresh
        )
        span = self.cfg.max_fps - self.cfg.min_fps
        return min(self.cfg.max_fps, self.cfg.min_fps + span * min(1.0, excess))

    def should_detect(self, ts_ms: int, image: np.ndarray) -> bool:
        self.n_considered += 1
        self.motion_score(image)

        if self.cfg.mode == "all":
            self._last_detect_ms = ts_ms
            self.n_detected += 1
            return True

        if self._last_detect_ms is None:
            self._last_detect_ms = ts_ms
            self.n_detected += 1
            return True

        interval_ms = 1000.0 / max(1e-6, self.target_fps())
        if (ts_ms - self._last_detect_ms) >= interval_ms:
            self._last_detect_ms = ts_ms
            self.n_detected += 1
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
            return True
        return False

    def stats(self) -> dict:
        return {
            "frames_considered": self.n_considered,
            "frames_detected": self.n_detected,
            "detection_ratio": round(
                self.n_detected / max(1, self.n_considered), 4
            ),
            "final_motion_ema": round(self._motion_ema, 4),
        }
