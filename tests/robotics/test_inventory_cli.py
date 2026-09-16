from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from djepa.robotics.inventory import (
    TaskSpec,
    inspect_task,
    verify_source_inventories,
)


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "robotics" / "inventory.py"


TASKS = (
    ("grab_roller", "robotwin_grab_roller", "manipulation"),
    ("place_bread_skillet", "robotwin_place_bread_skillet", "manipulation"),
    ("put_object_cabinet", "robotwin_put_object_cabinet", "manipulation"),
    ("handover_block", "robotwin_handover_block", "manipulation"),
)


def _make_task(root: Path, dataset_name: str) -> None:
    task = root / dataset_name / "1.0.0"
    task.mkdir(parents=True)
    shard_name = f"{dataset_name}-train.tfrecord-00000-of-00001"
    # Two payload bytes plus 16 bytes of TFRecord framing for each of 50 records.
    (task / shard_name).write_bytes(b"ab" + bytes(50 * 16))
    (task / "dataset_info.json").write_text(
        json.dumps(
            {
                "fileFormat": "tfrecord",
                "name": dataset_name,
                "splits": [
                    {
                        "name": "train",
                        "numBytes": "2",
                        "shardLengths": ["50"],
                    }
                ],
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    (task / "features.json").write_text(
        json.dumps({"schema": "fixture", "images": [240, 320, 3]}),
        encoding="utf-8",
    )


def _make_four_tasks(root: Path) -> None:
    for _, dataset_name, _ in TASKS:
        _make_task(root, dataset_name)


def _write_config(path: Path, tasks=TASKS) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "D-JEPA-RoboTwin2-task-set-v1",
                "tasks": [
                    {"name": name, "dataset_name": dataset_name}
                    for name, dataset_name, stage in tasks
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_cli(source: Path, config: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(source),
            "--tasks-config",
            str(config),
            "--output",
            str(output),
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_inventory_cli_publishes_completion_last(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_four_tasks(source)
    config = tmp_path / "tasks.json"
    _write_config(config)
    output = tmp_path / "result"

    run = _run_cli(source, config, output)

    assert run.returncode == 0, run.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "completion.json",
        "decision.json",
        "episode_splits.json",
        "source_inventory.json",
    ]
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    assert decision["tasks"] == [task[0] for task in TASKS]
    assert decision["application"] == "bimanual_manipulation"
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    assert set(completion["artifact_sha256"]) == {
        "decision.json",
        "episode_splits.json",
        "source_inventory.json",
    }
    assert set(completion["producer_sha256"]) == {
        "inventory_module",
        "producer_script",
        "tasks_config",
    }
    assert all(len(value) == 64 for value in completion["producer_sha256"].values())


def test_inventory_cli_refuses_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_four_tasks(source)
    config = tmp_path / "tasks.json"
    _write_config(config)
    output = tmp_path / "result"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    run = _run_cli(source, config, output)

    assert run.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "completion.json").exists()


def test_inventory_cli_rejects_unexpected_fifth_task(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_four_tasks(source)
    _make_task(source, "robotwin_unexpected")
    config = tmp_path / "tasks.json"
    _write_config(config, (*TASKS, ("unexpected", "robotwin_unexpected", "manipulation")))
    output = tmp_path / "result"

    run = _run_cli(source, config, output)

    assert run.returncode != 0
    assert not output.exists()
    assert list(tmp_path.glob("result.error-*.json"))


def test_source_verification_detects_change_after_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_four_tasks(source)
    inventories = tuple(
        inspect_task(source, TaskSpec(name=name, dataset_name=dataset_name))
        for name, dataset_name, _ in TASKS
    )
    changed = source / TASKS[0][1] / "1.0.0" / "features.json"
    changed.write_text(json.dumps({"schema": "changed"}), encoding="utf-8")

    with pytest.raises(ValueError, match="source changed"):
        verify_source_inventories(source, inventories)
