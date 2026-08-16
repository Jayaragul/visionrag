"""Command-line entry points.

    python -m visionrag demo                     # synthetic clip, no downloads
    python -m visionrag ingest VIDEO [-c CONFIG]
    python -m visionrag timeline [--db PATH] [--run N]
    python -m visionrag summary  [--db PATH] [--run N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .memory.store import MemoryStore
from .pipeline import IngestPipeline


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end check on a generated clip with known ground truth.

    Uses the stub detector, so this runs with no model download and no network
    -- the point is to verify the pipeline wiring, not detection quality.
    """
    from .ingest.video import make_synthetic_video

    out_dir = Path(args.out)
    clip = out_dir / ("shake.mp4" if args.shake else "static.mp4")
    print(f"generating synthetic clip -> {clip}")
    make_synthetic_video(clip, seconds=args.seconds, camera_shake=args.shake)

    cfg = Config()
    cfg.name = "demo-shake" if args.shake else "demo-static"
    cfg.video.path = str(clip)
    cfg.detector.backend = "stub"
    cfg.tracker.min_hits = 2
    cfg.scheduler.mode = "fixed"
    cfg.scheduler.fixed_fps = 5.0
    cfg.egomotion.enabled = not args.no_ego
    cfg.store.db_path = str(out_dir / "memory.db")
    cfg.store.evidence_dir = str(out_dir / "evidence")

    with MemoryStore(cfg.store.db_path) as store:
        pipe = IngestPipeline(cfg, store)
        summary = pipe.run(progress=args.verbose)
        run_id = store.run_id
        events = store.timeline(limit=100)

    print("\n--- run summary ---")
    _print_json(summary)
    print(f"\n--- event timeline (run {run_id}) ---")
    for e in events:
        flag = "  [ego-suspect]" if e["ego_suspect"] else ""
        print(
            f"  {e['t_start_ms'] / 1000:6.2f}s  {e['type']:<18} "
            f"tracks={e['participants']}  conf={e['confidence']:.2f}{flag}"
        )
    if not events:
        print("  (none)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if args.config else Config()
    cfg.video.path = args.video
    if args.backend:
        cfg.detector.backend = args.backend
    if args.threads:
        cfg.detector.num_threads = args.threads
    if args.fps:
        cfg.scheduler.mode = "fixed"
        cfg.scheduler.fixed_fps = args.fps
    if args.max_duration:
        cfg.video.max_duration_s = args.max_duration
    if args.db:
        cfg.store.db_path = args.db
    if args.no_ego:
        cfg.egomotion.enabled = False

    with MemoryStore(cfg.store.db_path) as store:
        pipe = IngestPipeline(cfg, store)
        summary = pipe.run(progress=True)
        summary["run_id"] = store.run_id
    _print_json(summary)
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    with MemoryStore(args.db) as store:
        store.run_id = args.run or _latest_run(store)
        events = store.timeline(
            types=args.type or None,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )
    for e in events:
        flag = "  [ego-suspect]" if e["ego_suspect"] else ""
        print(
            f"{e['t_start_ms'] / 1000:8.2f}s  {e['type']:<18} "
            f"tracks={str(e['participants']):<10} conf={e['confidence']:.2f}{flag}"
        )
    print(f"\n{len(events)} event(s)")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    with MemoryStore(args.db) as store:
        store.run_id = args.run or _latest_run(store)
        _print_json(store.run_summary())
    return 0


def _latest_run(store: MemoryStore) -> int:
    row = store.conn.execute("SELECT MAX(id) FROM runs").fetchone()
    if row is None or row[0] is None:
        raise SystemExit("no runs in database")
    return int(row[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="visionrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run end-to-end on a synthetic clip")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--seconds", type=float, default=12.0)
    d.add_argument("--shake", action="store_true", help="add camera shake")
    d.add_argument("--no-ego", action="store_true", help="disable compensation")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("ingest", help="ingest a video file")
    i.add_argument("video")
    i.add_argument("-c", "--config")
    i.add_argument("--backend", choices=["stub", "torchvision", "onnx"])
    i.add_argument("--threads", type=int)
    i.add_argument("--fps", type=float, help="fixed analysis fps")
    i.add_argument("--max-duration", type=float)
    i.add_argument("--db")
    i.add_argument("--no-ego", action="store_true")
    i.set_defaults(func=cmd_ingest)

    t = sub.add_parser("timeline", help="print the event timeline")
    t.add_argument("--db", default="runs/memory.db")
    t.add_argument("--run", type=int)
    t.add_argument("--type", action="append")
    t.add_argument("--min-confidence", type=float, default=0.0)
    t.add_argument("--limit", type=int, default=200)
    t.set_defaults(func=cmd_timeline)

    s = sub.add_parser("summary", help="print run metadata and cost")
    s.add_argument("--db", default="runs/memory.db")
    s.add_argument("--run", type=int)
    s.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
