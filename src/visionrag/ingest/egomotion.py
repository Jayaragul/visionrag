"""Global camera-motion estimation.

Handheld and egocentric video makes the whole scene translate, and a naive
event layer reads that as every object moving at once. This module estimates
the dominant frame-to-frame image transform so object motion can be separated
from camera motion.

This is the mechanism behind claim S1. The estimate is used two ways:

1. Subtracted from track velocities (see `track.Tracker._make_state`).
2. As a *counterfactual gate*: before asserting a motion event, ask whether
   camera motion alone could explain the observation. Events that fail the
   check are flagged, not dropped -- the suppression rate is itself a
   reported number, and silently discarding events would make the ablation
   impossible to interpret.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EgoMotionConfig
from ..types import CameraMotion

_FAILED = CameraMotion(dx=0.0, dy=0.0, scale=1.0, rotation=0.0, inliers=0, ok=False)


class EgoMotionEstimator:
    """Sparse Lucas-Kanade flow plus a partial-affine RANSAC fit.

    Sparse flow rather than dense: dense flow costs more CPU than the detector
    itself at these resolutions, which would undermine the very claim being
    made.
    """

    def __init__(self, cfg: EgoMotionConfig) -> None:
        self.cfg = cfg
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def estimate(self, image: np.ndarray) -> CameraMotion:
        if not self.cfg.enabled:
            return _FAILED

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self._prev_gray is None:
            self._prev_gray = gray
            return _FAILED

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.quality,
            minDistance=self.cfg.min_distance,
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) < 12:
            self._prev_gray = gray
            return _FAILED

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
        )
        self._prev_gray = gray
        if next_pts is None or status is None:
            return _FAILED

        good_prev = prev_pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]
        if len(good_prev) < 12:
            return _FAILED

        # Partial affine (translation, rotation, uniform scale) rather than a
        # full homography: it is far more stable with few features, and the
        # residual perspective error is small over one frame interval.
        matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.ransac_thresh,
        )
        if matrix is None:
            return _FAILED

        n_inliers = int(inliers.sum()) if inliers is not None else 0
        if n_inliers < 8:
            return _FAILED

        a, b = matrix[0, 0], matrix[0, 1]
        return CameraMotion(
            dx=float(matrix[0, 2] / w),  # normalised, matching box coordinates
            dy=float(matrix[1, 2] / h),
            scale=float(np.hypot(a, b)),
            rotation=float(np.arctan2(b, a)),
            inliers=n_inliers,
            ok=True,
        )

    def is_static(self, motion: CameraMotion) -> bool:
        return not motion.ok or motion.magnitude < self.cfg.static_thresh

    def explains_motion(
        self,
        v_raw: tuple[float, float],
        v_compensated: tuple[float, float],
    ) -> bool:
        """The counterfactual: could camera motion alone explain this?

        True when compensation removes most of the apparent speed -- i.e. the
        object barely moved relative to the world and the camera did the work.
        """
        raw_speed = float(np.hypot(*v_raw))
        comp_speed = float(np.hypot(*v_compensated))
        if raw_speed < 1e-6:
            return False
        return (comp_speed / raw_speed) < self.cfg.counterfactual_ratio
