import json
from pathlib import Path

import pytest

from djepa.robotics.inventory import (
    TaskSpec,
    inspect_task,
    make_split,
    write_split_manifest,
)


def make_fake_task(root: Path, *, shard_lengths: list[int]) -> Path:
    task = root / "robotwin_grab_roller" / "1.0.0"
    task.mkdir(parents=True)
    info = {
        "fileFormat": "tfrecord",
        "moduleName": "abc",
        "name": "robotwin_grab_roller",
        "splits": [
            {
                "name": "train",
                "numBytes": "2",
                "shardLengths": [str(value) for value in shard_lengths],
            }
        ],
        "version": "1.0.0",
    }
    (task / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    (task / "features.json").write_text(
        json.dumps({"schema": "fixture", "images": [240, 320, 3]}),
        encoding="utf-8",
    )
    shard_count = len(shard_lengths)
    for index, shard_length in enumerate(shard_lengths):
        name = f"robotwin_grab_roller-train.tfrecord-{index:05d}-of-{shard_count:05d}"
        # TFDS numBytes excludes TFRecord's 16-byte framing per serialized record.
        (task / name).write_bytes(bytes([index + 1]) + bytes(shard_length * 16))
    return task


def test_inventory_authenticates_fifty_episode_task(tmp_path: Path) -> None:
    make_fake_task(tmp_path, shard_lengths=[25, 25])

    inventory = inspect_task(
        tmp_path,
        TaskSpec("grab_roller", "robotwin_grab_roller"),
    )

    assert inventory.episode_count == 50
    assert inventory.shard_lengths == (25, 25)
    assert len(inventory.files) == 4
    assert all(item.sha256 for item in inventory.files)
    assert inventory.schema_sha256 == inventory.files[1].sha256


def test_split_is_episode_level_disjoint_and_fixed_size(tmp_path: Path) -> None:
    make_fake_task(tmp_path, shard_lengths=[25, 25])
    inventory = inspect_task(
        tmp_path,
        TaskSpec("grab_roller", "robotwin_grab_roller"),
    )

    split = make_split(inventory)

    assert tuple(map(len, (split.train, split.calibration, split.validation))) == (35, 5, 10)
    assert set(split.train).isdisjoint(split.calibration)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.calibration).isdisjoint(split.validation)
    assert sorted((*split.train, *split.calibration, *split.validation)) == list(range(50))
    assert split == make_split(inventory)


def test_inventory_rejects_non_fifty_episode_task(tmp_path: Path) -> None:
    make_fake_task(tmp_path, shard_lengths=[24, 25])

    with pytest.raises(ValueError, match="exactly 50"):
        inspect_task(tmp_path, TaskSpec("grab_roller", "robotwin_grab_roller"))


def test_inventory_rejects_bytes_outside_tfrecord_framing(tmp_path: Path) -> None:
    task = make_fake_task(tmp_path, shard_lengths=[25, 25])
    shard = task / "robotwin_grab_roller-train.tfrecord-00000-of-00002"
    shard.write_bytes(shard.read_bytes() + b"x")

    with pytest.raises(ValueError, match="TFRecord framing"):
        inspect_task(tmp_path, TaskSpec("grab_roller", "robotwin_grab_roller"))


def test_inventory_records_but_does_not_hash_derived_statistics(tmp_path: Path) -> None:
    task = make_fake_task(tmp_path, shard_lengths=[25, 25])
    statistics = task / "dataset_statistics_fixture.json"
    statistics.write_text("{}", encoding="utf-8")
    statistics.chmod(0)

    inventory = inspect_task(tmp_path, TaskSpec("grab_roller", "robotwin_grab_roller"))

    assert inventory.excluded_derived_files == ("dataset_statistics_fixture.json",)
    assert all("dataset_statistics_" not in item.relative_path for item in inventory.files)


def test_split_manifest_refuses_overwrite(tmp_path: Path) -> None:
    make_fake_task(tmp_path, shard_lengths=[25, 25])
    inventory = inspect_task(
        tmp_path,
        TaskSpec("grab_roller", "robotwin_grab_roller"),
    )
    split = make_split(inventory)
    output = tmp_path / "splits.json"

    write_split_manifest(output, [split])
    first = output.read_bytes()

    with pytest.raises(FileExistsError):
        write_split_manifest(output, [split])
    assert output.read_bytes() == first
