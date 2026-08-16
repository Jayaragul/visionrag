"""Fetch the face detection and recognition models.

Deliberately a separate, explicit step rather than an automatic download.
Face recognition is off by default and pulling ~37 MB of biometric model
weights should be something you chose to do, not something that happened
while you were running a demo.

    python scripts/get_face_models.py

Both models ship with OpenCV's own model zoo and run on CPU:

  YuNet  (~340 KB)  face detection
  SFace  (~37 MB)   face embedding for matching against enrolled people
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# opencv_zoo stores weights in Git LFS, so raw.githubusercontent.com returns a
# ~130-byte pointer file rather than the model. The media endpoint resolves LFS
# objects properly -- and because a pointer file is small but valid-looking,
# `verify` below checks the size rather than mere existence.
BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,   # ~233 KB
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30_000_000,  # ~37 MB
    ),
}


def download(url: str, dest: Path) -> None:
    # Only emit on a change of whole percent. Writing every block turns a
    # 37 MB download into thousands of lines in any non-tty log.
    last = [-1]

    def report(count: int, block: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, count * block * 100 // total)
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {dest.name}  {pct:3d}%", end="", flush=True)

    # Download to a temporary name so an interrupted transfer cannot leave a
    # truncated file that later looks present and valid.
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=report)
    tmp.replace(dest)
    print(f"\r  {dest.name}  done ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/faces")
    args = ap.parse_args()

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"fetching face models into {out}\n")

    for name, (url, min_bytes) in MODELS.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"  {name}  already present")
            continue
        if dest.exists():
            print(f"  {name}  present but too small -- refetching")
            dest.unlink()
        try:
            download(url, dest)
        except Exception as exc:
            print(f"\n  FAILED {name}: {exc}")
            return 1
        if dest.stat().st_size < min_bytes:
            print(
                f"\n  {name} downloaded but is only {dest.stat().st_size} bytes -- "
                "this is an LFS pointer, not the model."
            )
            dest.unlink()
            return 1

    print(
        "\nFace recognition is still disabled until you turn it on in config.\n"
        "It only recognises people you explicitly enroll -- it does not\n"
        "identify strangers, and stores nothing for unenrolled faces."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
