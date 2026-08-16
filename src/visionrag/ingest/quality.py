"""Frame quality assessment and illumination normalisation.

Two jobs, both of which fix real weaknesses elsewhere in the system:

1. **Gate confidence.** A detector returns boxes with confident-looking scores
   on a motion-blurred or nearly black frame. Those scores are not meaningful,
   and treating them as evidence lets junk into permanent memory. Quality
   multiplies into confidence, and a bad enough frame is refused outright.

2. **Stabilise place recognition.** The place descriptor is built from HSV
   histograms, which is exactly why turning a lamp on can make a room stop
   matching itself. Normalising illumination before describing removes most
   of that sensitivity.

All metrics are computed on a fixed-size downscale so thresholds mean the same
thing at any capture resolution -- Laplacian variance in particular scales with
image size, and a threshold tuned at 640px is meaningless at 1280px.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Every metric is measured on a long edge of this many pixels.
_REFERENCE_EDGE = 256


@dataclass(slots=True)
class FrameQuality:
    """What the image itself can support.

    `score` is a multiplier in [0, 1] for downstream confidence; `usable` is
    the hard floor below which a frame should not produce evidence at all.
    """

    brightness: float      # 0 = black, 1 = white
    contrast: float        # RMS contrast, 0-1
    sharpness: float       # Laplacian variance, resolution-normalised
    clipped_dark: float    # fraction of near-black pixels
    clipped_bright: float  # fraction of near-white pixels
    colour_cast: float     # 0 = neutral; >0 warm, <0 cool
    score: float
    usable: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "sharpness": round(self.sharpness, 2),
            "clipped_dark": round(self.clipped_dark, 4),
            "clipped_bright": round(self.clipped_bright, 4),
            "colour_cast": round(self.colour_cast, 4),
            "score": round(self.score, 4),
            "usable": self.usable,
            "reasons": list(self.reasons),
        }


# Thresholds. Deliberately permissive: refusing a usable frame costs a missed
# observation, which the persistence filter absorbs, while accepting an
# unusable one writes wrong facts into permanent memory.
DARK_LIMIT = 0.12
BRIGHT_LIMIT = 0.93
MIN_CONTRAST = 0.04
MIN_SHARPNESS = 25.0
MAX_CLIPPING = 0.45


def _reference(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = _REFERENCE_EDGE / float(max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def assess(image: np.ndarray) -> FrameQuality:
    small = _reference(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    brightness = float(gray_f.mean())
    contrast = float(gray_f.std())
    # Variance of the Laplacian: the standard no-reference blur measure. Low
    # variance means few sharp edges, which is either blur or a blank scene --
    # both equally bad reasons to trust a detection.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped_dark = float((gray < 6).mean())
    clipped_bright = float((gray > 249).mean())

    b, g, r = (float(c.mean()) for c in cv2.split(small))
    # Positive = red-heavy (tungsten), negative = blue-heavy (daylight/shade).
    colour_cast = (r - b) / max(1.0, r + b)

    reasons: list[str] = []
    if brightness < DARK_LIMIT:
        reasons.append("too dark")
    if brightness > BRIGHT_LIMIT:
        reasons.append("overexposed")
    if contrast < MIN_CONTRAST:
        # Near-uniform frame: lens covered, pointed at a blank wall, or the
        # camera failed to expose at all.
        reasons.append("no contrast (lens covered?)")
    if sharpness < MIN_SHARPNESS:
        reasons.append("blurred")
    if max(clipped_dark, clipped_bright) > MAX_CLIPPING:
        reasons.append("heavily clipped")

    # Graded score: each factor saturates at "good enough" rather than
    # rewarding extremes, so a very sharp frame cannot compensate for darkness.
    exposure_term = 1.0 - min(1.0, abs(brightness - 0.45) / 0.45)
    contrast_term = min(1.0, contrast / 0.18)
    sharpness_term = min(1.0, sharpness / 120.0)
    clipping_term = 1.0 - min(1.0, max(clipped_dark, clipped_bright) / MAX_CLIPPING)
    score = float(
        np.clip(
            0.30 * exposure_term
            + 0.25 * contrast_term
            + 0.30 * sharpness_term
            + 0.15 * clipping_term,
            0.0, 1.0,
        )
    )

    return FrameQuality(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        clipped_dark=clipped_dark,
        clipped_bright=clipped_bright,
        colour_cast=colour_cast,
        score=score,
        usable=not reasons,
        reasons=tuple(reasons),
    )


def normalise_illumination(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """CLAHE on the L channel of Lab.

    Equalising lightness while leaving colour alone removes most of the
    difference between the same room under a lamp, daylight, and an overcast
    afternoon. Plain histogram equalisation on RGB would shift hue and make the
    colour histograms *less* comparable, which is the opposite of the goal.

    Contrast-*limited* equalisation matters here: unrestricted equalisation
    amplifies sensor noise in flat regions into fake texture, which then
    produces spurious ORB features.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def relight(image: np.ndarray, gain: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Simulate a lighting change. Used by tests and the eval harness."""
    x = np.clip(image.astype(np.float32) * gain, 0, 255) / 255.0
    return (np.power(x, gamma) * 255.0).astype(np.uint8)
