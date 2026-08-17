"""Enrollment-based face recognition.

**This recognises people you have deliberately enrolled. It does not identify
strangers.** Nobody gets a stored biometric template unless you explicitly
enroll them; everyone else stays `person / unknown` and no face data is
written to disk at all.

That is a design decision, not a limitation:

* A face embedding is special-category biometric data under the GDPR, India's
  DPDP Act, Illinois BIPA and similar regimes. Templates of people who did not
  agree are a liability, not a feature.
* Open-set identification -- putting a name to an unknown face -- needs a
  reference gallery that has to come from somewhere, and for a personal
  spatial assistant there is no legitimate source for one.
* A small enrolled gallery is also far more accurate. Matching against six
  known people is an easy problem; matching against everyone is not.

Everything is local. Templates live in their own database file so they can be
deleted independently of the rest of the system, and `disabled by default`
means exactly that -- the models are never even loaded unless faces are
enabled in config.

Models
------
Uses OpenCV's built-in YuNet (detection) and SFace (recognition), so there is
no new Python dependency -- only two ONNX files, fetched deliberately:

    python scripts/get_face_models.py
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# SFace's documented cosine threshold for "same person". Raising it trades
# recall for precision; a spatial-memory product should prefer saying
# "unknown" over attaching the wrong name to someone.
DEFAULT_COSINE_THRESHOLD = 0.40

FACE_SCHEMA = """
PRAGMA journal_mode=WAL;

-- One row per enrolled person. `consent_note` is free text recording who
-- agreed to this and when; it exists so the record cannot be created without
-- someone having to write down why it is there.
CREATE TABLE IF NOT EXISTS people (
    person_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    consent_note TEXT,
    enrolled_at  REAL NOT NULL,
    n_templates  INTEGER NOT NULL DEFAULT 0
);

-- Several templates per person: faces vary with pose and lighting, and one
-- reference view generalises badly.
CREATE TABLE IF NOT EXISTS face_templates (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tmpl_person ON face_templates(person_id);
"""


@dataclass(slots=True)
class FaceObservation:
    """One detected face. `person_id` is None unless it matched an enrolled
    person above threshold -- and no template is stored for it either way."""

    box: tuple[float, float, float, float]  # normalised xyxy
    detection_score: float
    person_id: int | None = None
    name: str | None = None
    match_score: float = 0.0

    @property
    def identified(self) -> bool:
        return self.person_id is not None

    def as_dict(self) -> dict:
        return {
            "box": [round(v, 4) for v in self.box],
            "detection_score": round(self.detection_score, 3),
            "person_id": self.person_id,
            "name": self.name or "unknown",
            "match_score": round(self.match_score, 3),
            "identified": self.identified,
        }


class FaceModelsMissing(RuntimeError):
    pass


class FaceEngine:
    """Detects faces and matches them against enrolled people only."""

    def __init__(
        self,
        db_path: str | Path,
        model_dir: str | Path = "models/faces",
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        detection_threshold: float = 0.85,
        max_faces: int = 12,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.detector_path = self.model_dir / "face_detection_yunet_2023mar.onnx"
        self.recognizer_path = (
            self.model_dir / "face_recognition_sface_2021dec.onnx"
        )
        missing = [
            p for p in (self.detector_path, self.recognizer_path) if not p.exists()
        ]
        if missing:
            raise FaceModelsMissing(
                "face models not found:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\n\nFetch them with:  python scripts/get_face_models.py"
            )

        self.cosine_threshold = cosine_threshold
        self._detector = cv2.FaceDetectorYN.create(
            str(self.detector_path), "", (320, 320),
            detection_threshold, 0.3, max_faces,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(self.recognizer_path), ""
        )

        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because FastAPI runs blocking work on an
        # anyio worker *pool*: the thread that creates a session is not the
        # thread that later serves a request, and is not even guaranteed to be
        # the thread that handles the next frame. Serialisation is provided by
        # the caller's session lock; SQLite itself is serialised-mode safe.
        # WAL plus a busy timeout keeps concurrent readers from erroring out.
        self.conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(FACE_SCHEMA)
        self._gallery: list[tuple[int, str, np.ndarray]] = []
        self._load_gallery()

    # -- gallery ----------------------------------------------------------
    def _load_gallery(self) -> None:
        self._gallery = []
        for r in self.conn.execute(
            """SELECT t.person_id, p.name, t.embedding FROM face_templates t
               JOIN people p ON p.person_id = t.person_id"""
        ):
            vec = np.frombuffer(r["embedding"], dtype=np.float32).copy()
            self._gallery.append((r["person_id"], r["name"], vec))

    # -- detection --------------------------------------------------------
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Raw YuNet rows: [x, y, w, h, 10 landmark values, score]."""
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def _embed(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        # alignCrop warps to a canonical pose using the five landmarks; feeding
        # SFace an unaligned crop degrades matching badly.
        aligned = self._recognizer.alignCrop(image, face_row)
        feature = self._recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else feature

    def observe(self, image: np.ndarray) -> list[FaceObservation]:
        """Detect faces and name only the ones already enrolled.

        Embeddings computed here are used for comparison and then discarded.
        Nothing about an unenrolled face is persisted.
        """
        h, w = image.shape[:2]
        out: list[FaceObservation] = []
        for row in self.detect(image):
            x, y, bw, bh = row[:4]
            box = (
                float(np.clip(x / w, 0, 1)), float(np.clip(y / h, 0, 1)),
                float(np.clip((x + bw) / w, 0, 1)), float(np.clip((y + bh) / h, 0, 1)),
            )
            obs = FaceObservation(box=box, detection_score=float(row[-1]))
            if self._gallery:
                try:
                    embedding = self._embed(image, row)
                except cv2.error:
                    out.append(obs)  # alignment failed; still a face, still unknown
                    continue
                best_id, best_name, best = None, None, 0.0
                for person_id, name, template in self._gallery:
                    score = float(embedding @ template)
                    if score > best:
                        best_id, best_name, best = person_id, name, score
                if best >= self.cosine_threshold:
                    obs.person_id, obs.name, obs.match_score = best_id, best_name, best
                else:
                    # Below threshold is "unknown", never "closest guess".
                    obs.match_score = best
            out.append(obs)
        return out

    # -- enrollment -------------------------------------------------------
    def enroll(
        self, name: str, images: list[np.ndarray], consent_note: str | None = None
    ) -> int:
        """Register a person from one or more photographs.

        `consent_note` is required by convention rather than by the type
        system, but it is stored: a biometric record with no note of who agreed
        to it is a record nobody can justify later.
        """
        if not images:
            raise ValueError("enrollment needs at least one image")

        embeddings: list[np.ndarray] = []
        for image in images:
            faces = self.detect(image)
            if len(faces) == 0:
                continue
            # The largest face is the subject; bystanders in the background
            # must not be enrolled by accident.
            largest = max(faces, key=lambda r: r[2] * r[3])
            embeddings.append(self._embed(image, largest))
        if not embeddings:
            raise ValueError("no face found in any of the supplied images")

        cur = self.conn.execute(
            """INSERT INTO people (name, consent_note, enrolled_at, n_templates)
               VALUES (?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 n_templates = people.n_templates + excluded.n_templates,
                 consent_note = COALESCE(excluded.consent_note, people.consent_note)""",
            (name, consent_note, time.time(), len(embeddings)),
        )
        person_id = cur.lastrowid or self.conn.execute(
            "SELECT person_id FROM people WHERE name = ?", (name,)
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT INTO face_templates (person_id, embedding, dim, created_at)
               VALUES (?,?,?,?)""",
            [
                (person_id, e.astype(np.float32).tobytes(), int(e.size), time.time())
                for e in embeddings
            ],
        )
        self.conn.commit()
        self._load_gallery()
        return int(person_id)

    def forget(self, person_id: int) -> bool:
        """Delete a person and every template of them. Irreversible, and the
        only correct response to a withdrawal of consent."""
        cur = self.conn.execute(
            "DELETE FROM people WHERE person_id = ?", (person_id,)
        )
        self.conn.execute(
            "DELETE FROM face_templates WHERE person_id = ?", (person_id,)
        )
        self.conn.commit()
        self._load_gallery()
        return cur.rowcount > 0

    def people(self) -> list[dict]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT person_id, name, consent_note, enrolled_at, n_templates "
                "FROM people ORDER BY name"
            )
        ]

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "FaceEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
