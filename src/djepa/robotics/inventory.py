"""Read-only source inventory and episode-level split construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


_VERSION = "1.0.0"
_EPISODES = 50
_TFRECORD_FRAMING_BYTES = 16
_SPLIT_DOMAIN = b"D-JEPA-RoboTwin2-split-v1\0"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str


@dataclass(frozen=True)
class FileIdentity:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    task: TaskSpec
    version: str
    episode_count: int
    shard_lengths: tuple[int, ...]
    files: tuple[FileIdentity, ...]
    excluded_derived_files: tuple[str, ...]
    schema_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class SplitManifest:
    task: TaskSpec
    source_sha256: str
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    validation: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _one_train_split(info: dict[str, object]) -> dict[str, object]:
    splits = info.get("splits")
    if not isinstance(splits, list) or len(splits) != 1 or not isinstance(splits[0], dict):
        raise ValueError("dataset_info must contain exactly one split")
    split = splits[0]
    if split.get("name") != "train":
        raise ValueError("the only dataset split must be train")
    return split


def inspect_task(dataset_root: Path | str, spec: TaskSpec) -> SourceInventory:
    """Authenticate one immutable 50-episode RoboTwin2 RLDS task."""

    root = Path(dataset_root)
    version_root = root / spec.dataset_name / _VERSION
    info_path = version_root / "dataset_info.json"
    schema_path = version_root / "features.json"
    if not info_path.is_file() or not schema_path.is_file():
        raise FileNotFoundError(f"missing dataset_info.json or features.json for {spec.dataset_name}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("name") != spec.dataset_name:
        raise ValueError("dataset_info name does not match task specification")
    if info.get("version") != _VERSION:
        raise ValueError(f"dataset version must be {_VERSION}")
    if str(info.get("fileFormat", "")).lower() != "tfrecord":
        raise ValueError("dataset fileFormat must be tfrecord")

    split = _one_train_split(info)
    raw_lengths = split.get("shardLengths")
    if not isinstance(raw_lengths, list) or not raw_lengths:
        raise ValueError("train split must declare nonempty shardLengths")
    try:
        shard_lengths = tuple(int(value) for value in raw_lengths)
    except (TypeError, ValueError) as error:
        raise ValueError("shardLengths must contain integers") from error
    if any(value <= 0 for value in shard_lengths):
        raise ValueError("shardLengths must be positive")
    episode_count = sum(shard_lengths)
    if episode_count != _EPISODES:
        raise ValueError(f"RoboTwin demonstration sources must contain exactly 50 episodes, got {episode_count}")

    shard_count = len(shard_lengths)
    shard_paths = [
        version_root
        / f"{spec.dataset_name}-train.tfrecord-{index:05d}-of-{shard_count:05d}"
        for index in range(shard_count)
    ]
    missing = [path.name for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing TFRecord shards: {missing}")

    required = {info_path.name, schema_path.name, *(path.name for path in shard_paths)}
    excluded_derived_files = tuple(
        sorted(
            path.name
            for path in version_root.glob("dataset_statistics_*.json")
            if path.is_file()
        )
    )
    allowed = required | set(excluded_derived_files)
    unexpected = sorted(
        path.name for path in version_root.iterdir() if path.is_file() and path.name not in allowed
    )
    if unexpected:
        raise ValueError(f"unexpected source files: {unexpected}")

    declared_bytes = int(split.get("numBytes", -1))
    observed_bytes = sum(path.stat().st_size for path in shard_paths)
    expected_observed_bytes = declared_bytes + episode_count * _TFRECORD_FRAMING_BYTES
    if expected_observed_bytes != observed_bytes:
        raise ValueError(
            "declared payload bytes plus TFRecord framing do not match observed bytes: "
            f"{declared_bytes} + {episode_count} * {_TFRECORD_FRAMING_BYTES} "
            f"!= {observed_bytes}"
        )

    source_paths = sorted(version_root / name for name in required)
    identities = tuple(
        FileIdentity(
            relative_path=str(path.relative_to(root)),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in source_paths
    )
    schema_sha256 = next(
        item.sha256 for item in identities if Path(item.relative_path).name == "features.json"
    )
    source_payload = {
        "task": asdict(spec),
        "version": _VERSION,
        "episode_count": episode_count,
        "shard_lengths": shard_lengths,
        "files": [asdict(item) for item in identities],
        "excluded_derived_files": excluded_derived_files,
        "schema_sha256": schema_sha256,
    }
    source_sha256 = hashlib.sha256(_canonical_bytes(source_payload)).hexdigest()
    return SourceInventory(
        task=spec,
        version=_VERSION,
        episode_count=episode_count,
        shard_lengths=shard_lengths,
        files=identities,
        excluded_derived_files=excluded_derived_files,
        schema_sha256=schema_sha256,
        source_sha256=source_sha256,
    )


def make_split(inventory: SourceInventory) -> SplitManifest:
    """Create the fixed 35/5/10 split from task and episode identity."""

    if inventory.episode_count != _EPISODES:
        raise ValueError("split construction requires exactly 50 episodes")
    ranked = sorted(
        range(_EPISODES),
        key=lambda ordinal: hashlib.sha256(
            _SPLIT_DOMAIN
            + inventory.task.dataset_name.encode("utf-8")
            + b"\0"
            + str(ordinal).encode("ascii")
        ).digest(),
    )
    return SplitManifest(
        task=inventory.task,
        source_sha256=inventory.source_sha256,
        train=tuple(ranked[:35]),
        calibration=tuple(ranked[35:40]),
        validation=tuple(ranked[40:]),
    )


def verify_source_inventories(
    dataset_root: Path | str,
    inventories: Iterable[SourceInventory],
) -> None:
    """Re-authenticate sources immediately before publishing derived metadata."""

    for expected in inventories:
        observed = inspect_task(dataset_root, expected.task)
        if observed != expected:
            raise ValueError(
                f"source changed after inventory for {expected.task.dataset_name}: "
                f"expected {expected.source_sha256}, observed {observed.source_sha256}"
            )


def write_split_manifest(path: Path | str, manifests: Iterable[SplitManifest]) -> None:
    """Create a canonical split manifest and refuse replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(manifests)
    payload = {
        "schema": "D-JEPA-RoboTwin2-episode-splits-v1",
        "tasks": [asdict(row) for row in rows],
    }
    data = _canonical_bytes(payload)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
