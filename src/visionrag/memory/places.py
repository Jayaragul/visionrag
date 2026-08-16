"""Topological place memory.

Objects persist *somewhere*. Before you can ask "is the chair still there?"
you need "there" -- a stable identity for a location across visits, so the
same spot seen on Tuesday and Friday is recognised as one place.

Full metric SLAM is out of reach on a monocular phone camera with no
odometry and no GPU, and it is not needed. This builds a **topological** map:
places are nodes recognised by appearance, not coordinates. Cheaper, robust
to having no depth, and sufficient to anchor object persistence.

Hierarchical matching
---------------------
Two signals, each covering the other's weakness:

* **GPS (coarse).** Free from the phone. Outdoors it fixes position to
  5-20 m, far too coarse to distinguish a desk from a couch, but more than
  enough to rule out every place in another building. Shortlisting only.
* **Appearance (fine).** Picks the exact node from the shortlist, then ORB
  geometry confirms it.

Appearance alone suffers perceptual aliasing -- two similar corridors in
different buildings match each other. GPS alone cannot resolve places within a
room, and indoors is often unavailable or accurate only to 20-50 m. Gating
appearance search by a GPS radius removes most aliasing at no cost, and the
system degrades to appearance-only when there is no fix.

Coordinates
-----------
All keypoints are stored **normalised to [0, 1]**, so the homography maps
normalised live coordinates to normalised canonical ones and is independent
of frame size. Storing raw pixels would silently break the transform the
moment capture resolution changed.

Privacy
-------
Location is among the most sensitive data a camera product can hold. GPS is
strictly opt-in, never required, and stored in the same local database as
everything else -- covered by the same retention mode and session delete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(slots=True)
class GeoFix:
    """One GPS reading from the phone.

    `accuracy_m` is the browser's own 68% confidence radius and is the field
    that matters most: indoors it balloons to hundreds of metres, and a fix
    that imprecise must not be allowed to veto an appearance match.
    """

    lat: float
    lon: float
    accuracy_m: float = 100.0
    heading_deg: float | None = None

    def distance_to(self, other: "GeoFix") -> float:
        """Haversine great-circle distance in metres."""
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dp = p2 - p1
        dl = math.radians(other.lon - self.lon)
        a = (
            math.sin(dp / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        )
        return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class Place:
    place_id: int
    descriptor: np.ndarray
    geo: GeoFix | None = None
    label: str | None = None
    n_visits: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0
    # (normalised keypoints Nx2, ORB descriptors) from the first sighting.
    # Not refreshed on revisit: a stable reference view is what "canonical"
    # means, and letting it drift would make stored anchors drift with it.
    features: tuple[np.ndarray, np.ndarray] | None = None

    def merge_descriptor(self, other: np.ndarray, weight: float = 0.15) -> None:
        """Running average, so a place's appearance tracks slow drift in
        lighting and layout without being overwritten by one odd view."""
        self.descriptor = (1 - weight) * self.descriptor + weight * other
        norm = np.linalg.norm(self.descriptor)
        if norm > 0:
            self.descriptor /= norm


@dataclass(slots=True)
class PlaceVerification:
    """Result of geometrically verifying a candidate place match.

    Two different bars live here and conflating them is a mistake:

    * ``verified`` -- is this the same place? A *recognition* decision.
    * ``homography_trusted`` -- is the transform good enough to move object
      anchors with? A *geometry* decision.

    Recognising a room needs far less evidence than trusting a warp. A
    homography is planar and a room is not, so parallax from chairs and tables
    means the fit is never exact; a confidently wrong homography is worse than
    none, because it moves anchors decisively to the wrong place.
    """

    verified: bool
    homography: np.ndarray | None = None  # live -> canonical, NORMALISED
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float | None = None
    homography_trusted: bool = False
    reason: str | None = None


@dataclass(slots=True)
class PlaceMatch:
    """What `PlaceIndex.match()` / `.observe()` return.

    A result object rather than a widening tuple: `WorldMemory` needs the
    transform, the evidence behind it, and the reason a match failed, and
    threading those through as positional values invites exactly the sort of
    silent mix-up this is meant to prevent.
    """

    place: Place | None
    appearance_score: float
    verification: PlaceVerification | None = None
    is_new: bool = False

    @property
    def matched(self) -> bool:
        return self.place is not None

    @property
    def homography(self) -> np.ndarray | None:
        """The live -> canonical transform, or None if it is not trustworthy."""
        v = self.verification
        if v is None or not v.homography_trusted:
            return None
        return v.homography

    def to_canonical(self, x: float, y: float) -> tuple[float, float] | None:
        """Map a normalised live-frame point into the place's canonical frame.

        Returns None when there is no trustworthy transform -- callers must
        treat that as "cannot place this observation", not as "it is where it
        appears to be".
        """
        h = self.homography
        if h is None:
            return None
        p = h @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def frame_corners_canonical(self) -> np.ndarray | None:
        """The live frame's four corners in canonical coordinates.

        This is what makes observation coverage computable: the union of these
        polygons across a visit is the region actually looked at, and only
        objects inside it can be called missing.
        """
        h = self.homography
        if h is None:
            return None
        corners = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        ).T
        out = h @ corners
        w = out[2]
        if np.any(np.abs(w) < 1e-9):
            return None
        return (out[:2] / w).T  # 4x2


class PlaceIndex:
    """Recognises whether the current view is a known place.

    The descriptor is hand-crafted rather than learned: it costs ~5 ms, needs
    no extra model in the ingest loop, and keeps the compute claim intact. A
    learned global descriptor (NetVLAD-class) would be more robust to
    viewpoint change and is the obvious upgrade -- at the price of another
    network per frame, which is exactly what this project argues against.
    """

    def __init__(
        self,
        # Biased toward splitting rather than merging. Splitting a place costs
        # two partial histories; merging two places corrupts object beliefs --
        # the kitchen chair becomes evidence about the office. Measured on a
        # synthetic set, 0.72 without geometric verification merged 3 of 11
        # distinct scenes; 0.82 merged none.
        match_thresh: float = 0.82,
        # Appearance is a *shortlist*, not a gate, whenever geometry can
        # decide. Measured on a synthetic panorama, the tiled histogram falls
        # below 0.82 at only ~12% camera pan, while ORB still verifies the
        # same place at 40% pan with 114 inliers. Using the weaker signal as
        # the gate means the stronger one never gets consulted -- so candidates
        # are retrieved down to `shortlist_thresh` and geometry accepts or
        # rejects them. With verification disabled, `match_thresh` applies
        # instead, because then appearance is all there is.
        shortlist_thresh: float = 0.45,
        shortlist_size: int = 5,
        gps_gate_m: float = 60.0,
        gps_trust_accuracy_m: float = 50.0,
        verify_geometry: bool = True,
        # Recognition bar: enough to say "same place".
        min_match_inliers: int = 12,
        # Geometry bar: enough to move object anchors with. Deliberately
        # higher, and gated on ratio and reprojection error as well as count.
        min_warp_inliers: int = 20,
        min_inlier_ratio: float = 0.35,
        # A large consensus makes the ratio moot. The ratio exists to catch a
        # homography fitted to a handful of coincidental matches; scale changes
        # (walking closer, zooming) generate many spurious candidates and drag
        # the ratio down even when the surviving consensus is excellent.
        strong_inliers: int = 30,
        max_reprojection_error: float = 0.02,  # normalised units
        ransac_thresh: float = 0.01,  # normalised units (~6 px at 640)
        tiles: int = 4,
    ) -> None:
        self.match_thresh = match_thresh
        self.shortlist_thresh = shortlist_thresh
        self.shortlist_size = shortlist_size
        self.gps_gate_m = gps_gate_m
        self.gps_trust_accuracy_m = gps_trust_accuracy_m
        self.verify_geometry = verify_geometry
        self.min_match_inliers = min_match_inliers
        self.min_warp_inliers = min_warp_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.strong_inliers = strong_inliers
        self.max_reprojection_error = max_reprojection_error
        self.ransac_thresh = ransac_thresh
        self.tiles = tiles
        self.places: dict[int, Place] = {}
        self._next_id = 1
        self._orb = cv2.ORB_create(nfeatures=600) if verify_geometry else None
        self._matcher = (
            cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            if verify_geometry
            else None
        )

    # -- description ------------------------------------------------------
    def describe(self, image: np.ndarray) -> np.ndarray:
        """Tiled colour + gradient-orientation histogram.

        Tiling preserves coarse spatial layout, which a global histogram
        discards -- without it a red wall on the left and the same wall on the
        right are indistinguishable, and unrelated rooms with similar palettes
        collide.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # 0..360

        parts: list[np.ndarray] = []
        th, tw = h // self.tiles, w // self.tiles
        for ty in range(self.tiles):
            for tx in range(self.tiles):
                y0, y1 = ty * th, (ty + 1) * th if ty < self.tiles - 1 else h
                x0, x1 = tx * tw, (tx + 1) * tw if tx < self.tiles - 1 else w
                cell_hsv = hsv[y0:y1, x0:x1]
                hist_h = cv2.calcHist([cell_hsv], [0], None, [8], [0, 180]).ravel()
                hist_s = cv2.calcHist([cell_hsv], [1], None, [4], [0, 256]).ravel()
                # Magnitude-weighted orientation: strong edges should count for
                # more than sensor noise in flat regions.
                hist_o, _ = np.histogram(
                    ang[y0:y1, x0:x1].ravel(),
                    bins=8,
                    range=(0, 360),
                    weights=mag[y0:y1, x0:x1].ravel(),
                )
                parts.extend([hist_h, hist_s, hist_o.astype(np.float32)])

        vec = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _extract(
        self, image: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """ORB keypoints, **normalised to [0, 1]**, plus descriptors.

        Normalising here rather than at use time is what makes the homography
        resolution-independent.
        """
        if not self.verify_geometry:
            return None, None
        h, w = image.shape[:2]
        kp, desc = self._orb.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
        )
        if desc is None or not kp:
            return None, None
        pts = np.array([[k.pt[0] / w, k.pt[1] / h] for k in kp], dtype=np.float32)
        return pts, desc

    # -- matching ---------------------------------------------------------
    def _gps_candidates(self, geo: GeoFix | None) -> list[Place] | None:
        """Shortlist by GPS, or None when GPS cannot be trusted.

        Returning None rather than an empty list matters: "no usable fix" must
        fall back to searching every place, not conclude that no place is
        nearby. Indoors, where accuracy is worst, is where most observations
        happen.
        """
        if geo is None or geo.accuracy_m > self.gps_trust_accuracy_m:
            return None
        radius = self.gps_gate_m + geo.accuracy_m
        return [
            p
            for p in self.places.values()
            if p.geo is None or p.geo.distance_to(geo) <= radius
        ]

    def verify(self, image: np.ndarray, place: Place) -> PlaceVerification:
        """Geometric check of a candidate, with an explicit quality verdict."""
        if not self.verify_geometry:
            # Opting out of geometric checking means opting into the
            # assumption that frames are directly comparable, so the transform
            # is the identity. Returning no transform instead would silently
            # stop anything from being anchored at all.
            return PlaceVerification(
                verified=True,
                homography=np.eye(3, dtype=np.float64),
                homography_trusted=True,
                reason="verification disabled: assuming identity",
            )
        if place.features is None:
            return PlaceVerification(verified=True, reason="no reference features")

        live_pts, live_desc = self._extract(image)
        if live_desc is None or len(live_desc) < self.min_match_inliers:
            # Cannot verify. Inability to check is not evidence against, so the
            # match stands -- but the transform is emphatically not trusted.
            return PlaceVerification(verified=True, reason="too few live features")

        ref_pts, ref_desc = place.features
        if len(ref_desc) < self.min_match_inliers:
            return PlaceVerification(verified=True, reason="too few reference features")

        pairs = self._matcher.knnMatch(live_desc, ref_desc, k=2)
        good = [
            m
            for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * n.distance
        ]
        if len(good) < self.min_match_inliers:
            return PlaceVerification(
                verified=False, matches=len(good), reason="too few good matches"
            )

        src = live_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = ref_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        # src=live, dst=canonical, so H maps live -> canonical. This direction
        # is load-bearing; reversing it produces anchor drift that looks random.
        h_mat, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if h_mat is None or mask is None:
            return PlaceVerification(
                verified=False, matches=len(good), reason="homography failed"
            )

        inliers = int(mask.sum())
        ratio = inliers / max(1, len(good))
        error = self._reprojection_error(h_mat, src, dst, mask)

        plausible, degeneracy = self._is_plausible(h_mat)

        verified = inliers >= self.min_match_inliers and plausible
        trusted = (
            verified
            and inliers >= self.min_warp_inliers
            and error is not None
            and error <= self.max_reprojection_error
            # Either a decent share of matches agreed, or so many agreed in
            # absolute terms that the share stops mattering.
            and (ratio >= self.min_inlier_ratio or inliers >= self.strong_inliers)
        )
        if not plausible:
            reason = f"degenerate homography: {degeneracy}"
        elif not verified:
            reason = "too few inliers"
        elif not trusted:
            reason = (
                f"weak geometry (inliers={inliers}, ratio={ratio:.2f}, "
                f"err={None if error is None else round(error, 4)})"
            )
        else:
            reason = None

        return PlaceVerification(
            verified=verified,
            homography=h_mat if trusted else None,
            matches=len(good),
            inliers=inliers,
            inlier_ratio=round(ratio, 4),
            reprojection_error=error,
            homography_trusted=trusted,
            reason=reason,
        )

    @staticmethod
    def _is_plausible(h_mat: np.ndarray) -> tuple[bool, str | None]:
        """Reject geometrically degenerate homographies.

        Inlier count and reprojection error cannot catch this. A homography
        fitted to a cluster of correspondences on repetitive texture can
        collapse the whole frame to a point and still report a large, tightly
        fitting consensus -- measured on a synthetic case: 37 inliers, 0.005
        reprojection error, and every frame corner mapping to the same spot.

        The standard sanity checks are on the mapped unit square: it must not
        be mirrored, collapsed, exploded, or turned inside out.
        """
        if not np.all(np.isfinite(h_mat)):
            return False, "non-finite homography"

        # Reflection check on the linear part.
        if np.linalg.det(h_mat[:2, :2]) <= 1e-9:
            return False, "mirrored or singular"

        corners = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        ).T
        out = h_mat @ corners
        w = out[2]
        # All corners must stay in front of the camera.
        if np.any(w <= 1e-9):
            return False, "corner behind camera plane"
        quad = (out[:2] / w).T

        # Shoelace area of the mapped unit square. The source has area 1, so
        # this is directly a scale factor.
        x, y = quad[:, 0], quad[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area < 0.02:
            return False, f"collapsed (area {area:.4f})"
        if area > 25.0:
            return False, f"exploded (area {area:.1f})"

        # Convexity: consecutive edge cross products must all share a sign.
        # Computed directly rather than via np.cross, whose 2-D form is
        # deprecated in NumPy 2.
        edges = np.roll(quad, -1, axis=0) - quad
        nxt = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return False, "non-convex quad"

        return True, None

    @staticmethod
    def _reprojection_error(
        h_mat: np.ndarray, src: np.ndarray, dst: np.ndarray, mask: np.ndarray
    ) -> float | None:
        """Median symmetric error over inliers, in normalised units.

        Median rather than mean: a handful of parallax outliers that RANSAC
        happened to keep should not condemn an otherwise sound fit.
        """
        keep = mask.ravel().astype(bool)
        if keep.sum() == 0:
            return None
        projected = cv2.perspectiveTransform(src[keep], h_mat)
        return float(
            np.median(np.linalg.norm(projected - dst[keep], axis=2).ravel())
        )

    def match(self, image: np.ndarray, geo: GeoFix | None = None) -> PlaceMatch:
        descriptor = self.describe(image)
        candidates = self._gps_candidates(geo)
        if candidates is None:
            candidates = list(self.places.values())
        if not candidates:
            return PlaceMatch(place=None, appearance_score=0.0)

        # Descriptors are L2-normalised, so a dot product is cosine similarity.
        scored = sorted(
            ((float(descriptor @ p.descriptor), p) for p in candidates),
            key=lambda s: s[0],
            reverse=True,
        )
        best_score = scored[0][0]

        if not self.verify_geometry:
            # No geometry available, so appearance has to be the decision and
            # the strict threshold applies.
            if best_score < self.match_thresh:
                return PlaceMatch(place=None, appearance_score=best_score)
            return PlaceMatch(
                place=scored[0][1],
                appearance_score=best_score,
                verification=self.verify(image, scored[0][1]),
            )

        # Retrieve-then-verify: take the top few plausible candidates and let
        # geometry pick. An appearance score too low to shortlist is treated as
        # unrelated, but anything above the shortlist floor gets a real check.
        shortlist = [
            (score, place)
            for score, place in scored[: self.shortlist_size]
            if score >= self.shortlist_thresh
        ]
        last: PlaceVerification | None = None
        for score, place in shortlist:
            verification = self.verify(image, place)
            last = verification
            if verification.verified:
                return PlaceMatch(
                    place=place,
                    appearance_score=score,
                    verification=verification,
                )
        return PlaceMatch(
            place=None, appearance_score=best_score, verification=last
        )

    def observe(
        self, image: np.ndarray, ts_ms: int, geo: GeoFix | None = None
    ) -> PlaceMatch:
        """Recognise the current view, creating a new place if it is unknown."""
        result = self.match(image, geo)
        descriptor = self.describe(image)

        if result.place is None:
            live_pts, live_desc = self._extract(image)
            place = Place(
                place_id=self._next_id,
                descriptor=descriptor,
                geo=geo,
                first_seen_ms=ts_ms,
                last_seen_ms=ts_ms,
                features=(live_pts, live_desc) if live_desc is not None else None,
            )
            self.places[place.place_id] = place
            self._next_id += 1
            # For a brand-new place, this frame *is* the canonical frame, so
            # the transform is the identity and is trustworthy by definition.
            # Without this the first visit could warp nothing and would be
            # unable to anchor any object at all.
            result = PlaceMatch(
                place=place,
                appearance_score=result.appearance_score,
                verification=PlaceVerification(
                    verified=True,
                    homography=np.eye(3, dtype=np.float64),
                    homography_trusted=True,
                    reason="new place: canonical frame",
                ),
                is_new=True,
            )
        else:
            place = result.place
            place.merge_descriptor(descriptor)
            if geo is not None and (
                place.geo is None or geo.accuracy_m < place.geo.accuracy_m
            ):
                # Keep the most precise fix seen for this place, not the most
                # recent -- a good outdoor fix should not be replaced by a poor
                # indoor one.
                place.geo = geo

        place.last_seen_ms = ts_ms
        return result

    def begin_visit(self, place: Place) -> None:
        place.n_visits += 1
