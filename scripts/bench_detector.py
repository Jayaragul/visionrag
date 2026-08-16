"""Detector cost benchmark.

Detection dominates ingest cost, so the sustainable analysis rate follows
directly from this measurement. Every frame-rate default in the live config
should be traceable to a run of this script on the target machine -- an
efficiency claim on unmeasured hardware is not a claim.

    python scripts/bench_detector.py
    python scripts/bench_detector.py --threads 1,2,4 --sizes 320,640 --json out.json

Reports CPU-seconds as well as wall-seconds. On a multi-core box these differ
by roughly the thread count, and conflating them is the easiest way to
overstate efficiency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionrag.config import DetectorConfig  # noqa: E402
from visionrag.cost import hardware_fingerprint  # noqa: E402
from visionrag.ingest.detect import build_detector  # noqa: E402


def make_frame(size: int) -> np.ndarray:
    """A structured scene rather than noise: random pixels produce almost no
    detections and unrealistically fast NMS, which flatters the numbers."""
    h = int(size * 0.75)
    img = np.full((h, size, 3), 60, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for gx in range(0, size, 40):
        cv2.line(img, (gx, 0), (gx, h), (90, 90, 90), 1)
    for i in range(6):
        cx, cy = rng.integers(60, size - 60), rng.integers(60, h - 60)
        colour = tuple(int(c) for c in rng.integers(80, 255, 3))
        cv2.rectangle(img, (cx - 40, cy - 60), (cx + 40, cy + 60), colour, -1)
    return img


def bench_one(
    backend: str, model: str, threads: int, size: int, iters: int, warmup: int
) -> dict:
    cfg = DetectorConfig(
        backend=backend, model=model, num_threads=threads, input_size=size
    )
    det = build_detector(cfg)
    frame = make_frame(size)

    for _ in range(warmup):  # first calls include lazy allocation and caching
        det.detect(frame)

    walls, n_det = [], 0
    cpu0 = time.process_time()
    for _ in range(iters):
        t0 = time.perf_counter()
        dets = det.detect(frame)
        walls.append(time.perf_counter() - t0)
        n_det += len(dets)
    cpu_total = time.process_time() - cpu0

    walls_ms = sorted(w * 1000 for w in walls)
    p95 = walls_ms[max(0, int(round(0.95 * len(walls_ms))) - 1)]
    mean_wall = statistics.fmean(walls_ms)
    return {
        "backend": backend,
        "model": model,
        "threads": threads,
        "input_size": size,
        "iters": iters,
        "wall_ms_mean": round(mean_wall, 2),
        "wall_ms_p50": round(statistics.median(walls_ms), 2),
        "wall_ms_p95": round(p95, 2),
        "cpu_ms_mean": round(1000 * cpu_total / iters, 2),
        # The ceiling if detection were the only cost and one frame ran at a
        # time. Real sustainable rate is lower -- see the headroom note below.
        "max_fps_1_stream": round(1000.0 / mean_wall, 2),
        "detections_per_frame": round(n_det / iters, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torchvision")
    ap.add_argument("--model", default="ssdlite320_mobilenet_v3_large")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--sizes", default="320,480,640")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]

    hw = hardware_fingerprint()
    print(f"hardware: {hw['processor']}")
    print(f"cores: {hw['physical_cores']} physical / {hw['logical_cores']} logical")
    print(f"backend: {args.backend}  model: {args.model}\n")

    header = (
        f"{'size':>5} {'thr':>4} {'wall_ms':>9} {'p95_ms':>8} "
        f"{'cpu_ms':>8} {'max_fps':>8} {'dets':>6}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for size in sizes:
        for t in threads:
            try:
                r = bench_one(
                    args.backend, args.model, t, size, args.iters, args.warmup
                )
            except Exception as exc:  # a size/thread combo failing is not fatal
                print(f"{size:>5} {t:>4}   FAILED: {exc}")
                continue
            results.append(r)
            print(
                f"{r['input_size']:>5} {r['threads']:>4} {r['wall_ms_mean']:>9} "
                f"{r['wall_ms_p95']:>8} {r['cpu_ms_mean']:>8} "
                f"{r['max_fps_1_stream']:>8} {r['detections_per_frame']:>6}"
            )

    if results:
        best = min(results, key=lambda r: r["wall_ms_mean"])
        # Two thirds of the detector-only ceiling, to leave room for
        # ego-motion, tracking, JPEG decode, persistence and the OS. A
        # scheduler pinned at the ceiling has no slack and degrades into
        # unbounded queueing the moment anything else runs.
        recommended = max(1.0, round(best["max_fps_1_stream"] * 0.66, 1))
        print(
            f"\nfastest: size={best['input_size']} threads={best['threads']} "
            f"-> {best['wall_ms_mean']} ms ({best['max_fps_1_stream']} fps ceiling)"
        )
        print(f"suggested scheduler.max_fps (66% of ceiling): {recommended}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"hardware": hw, "results": results}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
