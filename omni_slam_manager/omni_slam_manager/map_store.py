"""Crash-safe, immutable map asset storage.

FAST-LIO remains responsible for producing a valid PCD.  MapStore imports that
file into a versioned directory, writes a checksum manifest, fsyncs the complete
version, and only then atomically switches the ``current`` symlink.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import BinaryIO, Dict, Iterator, Optional, Tuple
import uuid


_MAP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^v([0-9]{6})$")
_CHUNK_SIZE = 1024 * 1024


class MapStoreError(RuntimeError):
    """Raised when a map cannot be safely imported or verified."""


@dataclass(frozen=True)
class MapAsset:
    map_id: str
    version: int
    directory: Path
    pcd_path: Path
    manifest_path: Path
    checksum: str
    size_bytes: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stream(stream: BinaryIO) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class MapStore:
    """Manage immutable map versions under a single protected root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise MapStoreError(f"map root is not a directory: {self.root}")
        self._lock_path = self.root / ".map_store.lock"

    @staticmethod
    def validate_map_id(map_id: str) -> str:
        if not _MAP_ID_PATTERN.fullmatch(map_id):
            raise MapStoreError(
                "map_id must be 1-64 characters using letters, digits, '.', '_' or '-'"
            )
        return map_id

    def import_pcd(
        self,
        map_id: str,
        source: Path | str,
        *,
        calibration_hash: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> MapAsset:
        """Import a non-empty PCD and atomically publish it as current."""

        map_id = self.validate_map_id(map_id)
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise MapStoreError(f"source PCD is not a non-empty regular file: {source_path}")

        with self._exclusive_lock():
            map_directory = self.root / map_id
            versions_directory = map_directory / "versions"
            versions_directory.mkdir(parents=True, exist_ok=True)
            version = self._next_version(versions_directory)
            version_name = f"v{version:06d}"
            final_directory = versions_directory / version_name
            staging_directory = versions_directory / (
                f".tmp-{version_name}-{uuid.uuid4().hex}"
            )
            current_tmp = map_directory / f".current-{uuid.uuid4().hex}"

            try:
                staging_directory.mkdir(mode=0o750)
                pcd_path = staging_directory / "map.pcd"
                checksum, size_bytes = self._copy_and_hash(source_path, pcd_path)
                if size_bytes <= 0:
                    raise MapStoreError("source PCD became empty while importing")

                manifest = {
                    "schema_version": 1,
                    "map_id": map_id,
                    "version": version,
                    "checksum_algorithm": "sha256",
                    "checksum": checksum,
                    "size_bytes": size_bytes,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "calibration_hash": calibration_hash,
                    "source_name": source_path.name,
                    "metadata": metadata or {},
                }
                manifest_path = staging_directory / "manifest.json"
                with manifest_path.open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                _fsync_directory(staging_directory)
                os.replace(staging_directory, final_directory)
                _fsync_directory(versions_directory)

                os.symlink(f"versions/{version_name}", current_tmp)
                os.replace(current_tmp, map_directory / "current")
                _fsync_directory(map_directory)
            except Exception:
                if current_tmp.is_symlink():
                    current_tmp.unlink()
                if staging_directory.exists():
                    shutil.rmtree(staging_directory)
                raise

            return MapAsset(
                map_id=map_id,
                version=version,
                directory=final_directory,
                pcd_path=final_directory / "map.pcd",
                manifest_path=final_directory / "manifest.json",
                checksum=checksum,
                size_bytes=size_bytes,
            )

    def current(self, map_id: str, *, verify: bool = True) -> MapAsset:
        map_id = self.validate_map_id(map_id)
        map_directory = self.root / map_id
        versions_directory = map_directory / "versions"
        current = map_directory / "current"
        if not current.is_symlink():
            raise MapStoreError(f"map has no published current version: {map_id}")
        resolved = current.resolve(strict=True)
        if resolved.parent != versions_directory.resolve() or not _VERSION_PATTERN.fullmatch(
            resolved.name
        ):
            raise MapStoreError(f"current map pointer escapes version store: {current}")
        asset = self._asset_from_directory(map_id, resolved)
        if verify:
            self.verify(asset)
        return asset

    def verify(self, asset: MapAsset) -> None:
        with asset.pcd_path.open("rb") as stream:
            checksum, size_bytes = _hash_stream(stream)
        if size_bytes != asset.size_bytes or checksum != asset.checksum:
            raise MapStoreError(
                f"map checksum mismatch for {asset.map_id} v{asset.version:06d}"
            )

    def _asset_from_directory(self, map_id: str, directory: Path) -> MapAsset:
        manifest_path = directory / "manifest.json"
        pcd_path = directory / "map.pcd"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = int(manifest["version"])
            checksum = str(manifest["checksum"])
            size_bytes = int(manifest["size_bytes"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise MapStoreError(f"invalid map manifest: {manifest_path}: {error}") from error
        if manifest.get("schema_version") != 1 or manifest.get("map_id") != map_id:
            raise MapStoreError(f"manifest identity mismatch: {manifest_path}")
        if (
            manifest.get("checksum_algorithm") != "sha256"
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or size_bytes <= 0
        ):
            raise MapStoreError(f"manifest checksum fields are invalid: {manifest_path}")
        if not pcd_path.is_file() or pcd_path.stat().st_size <= 0:
            raise MapStoreError(f"map PCD is missing or empty: {pcd_path}")
        return MapAsset(
            map_id=map_id,
            version=version,
            directory=directory,
            pcd_path=pcd_path,
            manifest_path=manifest_path,
            checksum=checksum,
            size_bytes=size_bytes,
        )

    @staticmethod
    def _copy_and_hash(source: Path, destination: Path) -> Tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as source_stream, destination.open("xb") as target_stream:
            while True:
                chunk = source_stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                target_stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        return digest.hexdigest(), size

    @staticmethod
    def _next_version(versions_directory: Path) -> int:
        versions = []
        for entry in versions_directory.iterdir():
            match = _VERSION_PATTERN.fullmatch(entry.name)
            if match and entry.is_dir():
                versions.append(int(match.group(1)))
        return max(versions, default=0) + 1

    def _exclusive_lock(self) -> Iterator[None]:
        class _Lock:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.stream: Optional[BinaryIO] = None

            def __enter__(self) -> None:
                self.stream = self.path.open("a+b")
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                assert self.stream is not None
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
                self.stream.close()

        return _Lock(self._lock_path)
