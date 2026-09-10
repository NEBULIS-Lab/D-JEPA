import argparse
from datetime import datetime, timezone
from pathlib import Path

from djepa_robot.data import audit_dataset
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description="Read-only raw PiPER recording audit; no RGB decoding")
    p.add_argument("--root", type=Path, required=True, help="recording root containing lerobot/ and raw_hdf5/")
    p.add_argument("--tasks", nargs="+", default=["corn_to_plate", "red_cube_to_plate", "red_pen_to_bucket"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = new_output(args.output)
    reports = [audit_dataset(args.root / "lerobot" / f"piper_d455_{task}", args.root / "raw_hdf5" / task)
               for task in args.tasks]
    write_json(output / "audit.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "datasets": reports})
    for task, r in zip(args.tasks, reports):
        print(f"{task}: {r['valid_episodes']}/{r['raw_files']} structurally valid, "
              f"{r['raw_frames']} frames, {r['gap_count']} gaps > 2/nominal_fps seconds")
    print(output / "audit.json")
    return 0 if all(r["metadata_counts_match"] for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
