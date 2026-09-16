#!/usr/bin/env python3
"""Publish an authenticated four-task RoboTwin2 source-admission bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import djepa.robotics.inventory as inventory_module
from djepa.robotics.inventory import (
    SourceInventory,
    TaskSpec,
    inspect_task,
    make_split,
    verify_source_inventories,
)


_TASK_SCHEMA = "D-JEPA-RoboTwin2-task-set-v1"
_INVENTORY_SCHEMA = "D-JEPA-RoboTwin2-source-inventory-v1"
_SPLIT_SCHEMA = "D-JEPA-RoboTwin2-episode-splits-v1"
_DECISION_SCHEMA = "D-JEPA-RoboTwin2-admission-decision-v1"
_COMPLETION_SCHEMA = "D-JEPA-RoboTwin2-admission-completion-v1"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_canonical(path: Path, value: Any) -> str:
    data = _canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tasks(path: Path) -> tuple[tuple[TaskSpec, str], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != _TASK_SCHEMA:
        raise ValueError(f"tasks config schema must be {_TASK_SCHEMA}")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise ValueError("tasks config must contain exactly four tasks")

    parsed: list[tuple[TaskSpec, str]] = []
    for index, row in enumerate(tasks):
        if not isinstance(row, dict) or set(row) != {"name", "dataset_name"}:
            raise ValueError(f"task {index} must contain only name and dataset_name")
        name = row["name"]
        dataset_name = row["dataset_name"]
        if not all(isinstance(value, str) and value for value in (name, dataset_name)):
            raise ValueError(f"task {index} fields must be nonempty strings")
        parsed.append((TaskSpec(name=name, dataset_name=dataset_name), "manipulation"))

    names = [spec.name for spec, _ in parsed]
    datasets = [spec.dataset_name for spec, _ in parsed]
    if len(set(names)) != 4 or len(set(datasets)) != 4:
        raise ValueError("task names and dataset names must be unique")
    return tuple(parsed)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_inventory(dataset_root: Path, tasks_config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.partial-{os.getpid()}")
    staging_created = False
    try:
        staging.mkdir()
        staging_created = True
        task_rows = _load_tasks(tasks_config)
        inventories: tuple[SourceInventory, ...] = tuple(
            inspect_task(dataset_root, spec) for spec, _ in task_rows
        )
        splits = tuple(make_split(inventory) for inventory in inventories)

        payloads = {
            "source_inventory.json": {
                "schema": _INVENTORY_SCHEMA,
                "tasks": [asdict(inventory) for inventory in inventories],
            },
            "episode_splits.json": {
                "schema": _SPLIT_SCHEMA,
                "tasks": [asdict(split) for split in splits],
            },
            "decision.json": {
                "schema": _DECISION_SCHEMA,
                "status": "source_admitted",
                "tasks": [spec.name for spec, _ in task_rows],
                "application": "bimanual_manipulation",
            },
        }
        artifact_hashes = {
            name: _write_canonical(staging / name, payload)
            for name, payload in payloads.items()
        }

        # Re-read and re-hash the source immediately before the completion marker.
        verify_source_inventories(dataset_root, inventories)
        producer_hashes = {
            "inventory_module": _file_sha256(Path(inventory_module.__file__).resolve()),
            "producer_script": _file_sha256(Path(__file__).resolve()),
            "tasks_config": _file_sha256(tasks_config.resolve()),
        }
        _write_canonical(
            staging / "completion.json",
            {
                "schema": _COMPLETION_SCHEMA,
                "status": "complete",
                "artifact_sha256": artifact_hashes,
                "producer_sha256": producer_hashes,
            },
        )
        _fsync_directory(staging)
        os.rename(staging, output)
        staging_created = False
        _fsync_directory(output.parent)
    except Exception:
        error_path = output.with_name(f"{output.name}.error-{os.getpid()}.json")
        try:
            _write_canonical(
                error_path,
                {
                    "schema": _COMPLETION_SCHEMA,
                    "status": "error",
                    "output": output.name,
                    "error": f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}",
                },
            )
        finally:
            if staging_created:
                shutil.rmtree(staging)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tasks-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        publish_inventory(args.dataset_root, args.tasks_config, args.output)
    except Exception as error:
        print(f"inventory admission failed: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
