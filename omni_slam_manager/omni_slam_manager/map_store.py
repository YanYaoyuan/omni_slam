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
import threading
from typing import BinaryIO, Callable, Dict, Iterator, Optional, Tuple
import uuid


_MAP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^v([0-9]{6})$")
_CHUNK_SIZE = 1024 * 1024


class MapStoreError(RuntimeError):
    """Raised when a map cannot be safely imported or verified."""


class MapStoreImportCancelled(MapStoreError):
    """不可变地图版本到达提交点前观察到取消时抛出。"""


@dataclass(frozen=True)
class MapAsset:
    map_id: str
    version: int
    directory: Path
    pcd_path: Path
    manifest_path: Path
    checksum: str
    size_bytes: int
    # FAST-LIO 关键帧导出（位姿、局部扫描、Scan Context），旧版本地图为 None。
    keyframes_directory: Optional[Path] = None


@dataclass(frozen=True)
class MapCatalogEntry:
    """一个已校验、可由定位运行时选择的 current 地图版本。"""

    map_id: str
    version: int
    checksum: str
    created_at: str
    size_bytes: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stream(stream: BinaryIO) -> Tuple[str, int]:
    return _hash_stream_with_cancel(stream, None)


def _hash_stream_with_cancel(
    stream: BinaryIO,
    cancel_requested: Optional[Callable[[], bool]],
) -> Tuple[str, int]:
    """流式计算摘要，并在资产提交点之前持续响应取消。"""

    digest = hashlib.sha256()
    size = 0
    while True:
        if cancel_requested is not None and cancel_requested():
            raise MapStoreImportCancelled("map import canceled before commit")
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    if cancel_requested is not None and cancel_requested():
        raise MapStoreImportCancelled("map import canceled before commit")
    return digest.hexdigest(), size


class MapStore:
    """Manage immutable map versions under a single protected root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise MapStoreError(f"map root is not a directory: {self.root}")
        self._lock_path = self.root / ".map_store.lock"
        # 已发布的版本不可变；文件或 manifest 身份变化后必须重新校验。
        self._verified_files: dict[Path, tuple] = {}
        self._verify_lock = threading.Lock()

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
        publish_current: bool = True,
        cancel_requested: Optional[Callable[[], bool]] = None,
        body_path_source: Optional[Path] = None,
        keyframes_source: Optional[Path] = None,
    ) -> MapAsset:
        """导入、校验并原子发布非空 PCD。

        在可能耗时较长的复制和校验阶段持续采样 ``cancel_requested``。首次原子
        rename 前观察到取消时会删除临时目录，不发布版本，也不更新 ``current``。
        rename 开始后进入很短的不可取消提交区，避免资产已经发布而调用方却收到
        CANCELED 终态。
        """

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
                self._raise_if_cancelled(cancel_requested)
                checksum, size_bytes = self._copy_and_hash(
                    source_path, pcd_path, cancel_requested
                )
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
                if body_path_source is not None:
                    body_path_source = Path(body_path_source).resolve(strict=True)
                    if body_path_source.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
                        raise MapStoreError("body trajectory predates the saved PCD")
                    body_checksum, body_size = self._copy_and_hash(
                        body_path_source, staging_directory / "body_path.txt", cancel_requested
                    )
                    if body_size <= 0:
                        raise MapStoreError("saved body trajectory is empty")
                    manifest["body_path"] = {
                        "file": "body_path.txt", "checksum": body_checksum, "size_bytes": body_size
                    }
                if keyframes_source is not None:
                    manifest["keyframes"] = self._import_keyframes(
                        Path(keyframes_source),
                        source_path,
                        staging_directory / "keyframes",
                        cancel_requested,
                    )
                manifest_path = staging_directory / "manifest.json"
                with manifest_path.open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                _fsync_directory(staging_directory)
                # 发布前重新读取落盘内容。校验期间仍可取消；之后进入不可取消的
                # 原子提交点，确保 Action 终态与磁盘事实不会互相矛盾。
                with pcd_path.open("rb") as stream:
                    verified_checksum, verified_size = _hash_stream_with_cancel(
                        stream, cancel_requested
                    )
                if verified_checksum != checksum or verified_size != size_bytes:
                    raise MapStoreError("staged PCD checksum verification failed")
                self._raise_if_cancelled(cancel_requested)
                os.replace(staging_directory, final_directory)
                _fsync_directory(versions_directory)

                if publish_current:
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
                keyframes_directory=(
                    final_directory / "keyframes" if "keyframes" in manifest else None
                ),
            )

    def verify_keyframes(self, asset: MapAsset) -> None:
        """完整校验关键帧导出的每个文件；定位热路径不需要调用。"""

        if asset.keyframes_directory is None:
            raise MapStoreError(f"map version has no keyframes: {asset.map_id}")
        try:
            manifest = json.loads(asset.manifest_path.read_text(encoding="utf-8"))
            files = manifest["keyframes"]["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise MapStoreError(f"invalid keyframe manifest: {asset.manifest_path}") from error
        for name, expected in files.items():
            path = asset.keyframes_directory / name
            try:
                with path.open("rb") as stream:
                    checksum, size_bytes = _hash_stream(stream)
            except OSError as error:
                raise MapStoreError(f"keyframe file is unreadable: {path}") from error
            if checksum != expected.get("checksum") or size_bytes != expected.get("size_bytes"):
                raise MapStoreError(f"keyframe checksum mismatch: {path}")

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

    def version(self, map_id: str, version: int, *, verify: bool = True) -> MapAsset:
        map_id = self.validate_map_id(map_id)
        if version <= 0 or version > 999999:
            raise MapStoreError("map version must be in range 1..999999")
        directory = self.root / map_id / "versions" / f"v{version:06d}"
        if not directory.is_dir():
            raise MapStoreError(f"map version does not exist: {map_id} v{version:06d}")
        asset = self._asset_from_directory(map_id, directory)
        if asset.version != version:
            raise MapStoreError(f"map manifest version mismatch: {directory}")
        if verify:
            self.verify(asset)
        return asset

    def list_current(self) -> list[MapCatalogEntry]:
        """
        返回所有可安全选择的 current 地图。

        临时目录、越界符号链接、缺失 manifest 以及校验失败的资产均不进入
        目录，避免 APP 在冷启动后看到一个定位必然失败或内容已漂移的地图。
        """

        catalog: list[MapCatalogEntry] = []
        for map_directory in self.root.iterdir():
            if (
                not map_directory.is_dir()
                or map_directory.is_symlink()
                or not _MAP_ID_PATTERN.fullmatch(map_directory.name)
            ):
                continue
            try:
                asset = self.current(map_directory.name, verify=True)
                manifest = json.loads(asset.manifest_path.read_text(encoding="utf-8"))
                created_at = str(manifest["created_at"])
                created = datetime.fromisoformat(created_at)
                if created.tzinfo is None:
                    raise ValueError("created_at must include a timezone")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, MapStoreError):
                continue
            catalog.append(
                MapCatalogEntry(
                    map_id=asset.map_id,
                    version=asset.version,
                    checksum=asset.checksum,
                    created_at=created_at,
                    size_bytes=asset.size_bytes,
                )
            )
        catalog.sort(key=lambda item: (item.created_at, item.map_id), reverse=True)
        return catalog

    def verify(self, asset: MapAsset) -> None:
        # 列目录不应每次重复读取所有大地图。ctime 同时覆盖同大小改写后恢复
        # mtime 的情况；首次访问及文件替换、改写后仍执行完整 SHA-256 校验。
        def identity(stat: os.stat_result) -> tuple:
            return (
                stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns,
                asset.checksum, asset.size_bytes,
            )

        with self._verify_lock, asset.pcd_path.open("rb") as stream:
            before = identity(os.fstat(stream.fileno()))
            if self._verified_files.get(asset.pcd_path) == before:
                return
            self._verified_files.pop(asset.pcd_path, None)
            checksum, size_bytes = _hash_stream(stream)
            if (
                size_bytes != asset.size_bytes or checksum != asset.checksum
                or identity(os.fstat(stream.fileno())) != before
                or identity(asset.pcd_path.stat()) != before
            ):
                raise MapStoreError(
                    f"map checksum mismatch for {asset.map_id} v{asset.version:06d}"
                )
            self._verified_files[asset.pcd_path] = before

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
        keyframes_directory = directory / "keyframes"
        return MapAsset(
            map_id=map_id,
            version=version,
            directory=directory,
            pcd_path=pcd_path,
            manifest_path=manifest_path,
            checksum=checksum,
            size_bytes=size_bytes,
            keyframes_directory=(
                keyframes_directory
                if "keyframes" in manifest and keyframes_directory.is_dir()
                else None
            ),
        )

    @staticmethod
    def _raise_if_cancelled(
        cancel_requested: Optional[Callable[[], bool]],
    ) -> None:
        if cancel_requested is not None and cancel_requested():
            raise MapStoreImportCancelled("map import canceled before commit")

    @classmethod
    def _import_keyframes(
        cls,
        source: Path,
        pcd_source: Path,
        destination: Path,
        cancel_requested: Optional[Callable[[], bool]],
    ) -> Dict[str, object]:
        """复制 FAST-LIO 关键帧导出，返回写入 manifest 的逐文件校验信息。"""

        source = source.resolve(strict=True)
        index = source / "keyframes.txt"
        scans = source / "scans"
        if not source.is_dir() or not index.is_file() or not scans.is_dir():
            raise MapStoreError(f"keyframe export is incomplete: {source}")
        # FAST-LIO 在 PCD 之后写关键帧；更旧的 keyframes.txt 属于上一次保存。
        if index.stat().st_mtime_ns < pcd_source.stat().st_mtime_ns:
            raise MapStoreError("keyframe export predates the saved PCD")
        names = ["keyframes.txt", "scan_context.bin"] + sorted(
            f"scans/{entry.name}" for entry in scans.iterdir() if entry.name.endswith(".pcd")
        )
        destination.mkdir(mode=0o750)
        (destination / "scans").mkdir(mode=0o750)
        files: Dict[str, object] = {}
        for name in names:
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise MapStoreError(f"keyframe file is not a regular file: {path}")
            checksum, size_bytes = cls._copy_and_hash(path, destination / name, cancel_requested)
            files[name] = {"checksum": checksum, "size_bytes": size_bytes}
        _fsync_directory(destination / "scans")
        _fsync_directory(destination)
        with (destination / "keyframes.txt").open(encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip() and not line.startswith("#"))
        if count <= 0:
            raise MapStoreError("keyframe export contains no keyframes")
        return {
            "directory": "keyframes",
            "format": "HENG_KEYFRAMES_1",
            "count": count,
            "files": files,
        }

    @staticmethod
    def _copy_and_hash(
        source: Path,
        destination: Path,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as source_stream, destination.open("xb") as target_stream:
            while True:
                MapStore._raise_if_cancelled(cancel_requested)
                chunk = source_stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                target_stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            MapStore._raise_if_cancelled(cancel_requested)
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
