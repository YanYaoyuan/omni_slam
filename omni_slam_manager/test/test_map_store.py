import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omni_slam_manager import map_store

from omni_slam_manager.map_store import (
    MapStore,
    MapStoreError,
    MapStoreImportCancelled,
)


def _pcd(path: Path, payload: bytes = b"pcd-map-data") -> Path:
    path.write_bytes(payload)
    return path


class MapStoreTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_import_publishes_versioned_asset_and_manifest(self):
        source = _pcd(self.root / "source.pcd")
        store = MapStore(self.root / "maps")

        asset = store.import_pcd(
            "plant-a",
            source,
            calibration_hash="calibration-sha256",
            metadata={"frame_id": "map"},
        )

        self.assertEqual(asset.version, 1)
        self.assertEqual(asset.pcd_path.read_bytes(), b"pcd-map-data")
        self.assertEqual(asset.checksum, hashlib.sha256(b"pcd-map-data").hexdigest())
        self.assertTrue((self.root / "maps/plant-a/current").is_symlink())
        manifest = json.loads(asset.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["map_id"], "plant-a")
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["calibration_hash"], "calibration-sha256")
        self.assertEqual(manifest["metadata"], {"frame_id": "map"})
        self.assertEqual(store.current("plant-a"), asset)

    def test_import_creates_immutable_monotonic_versions(self):
        source = _pcd(self.root / "source.pcd", b"first")
        store = MapStore(self.root / "maps")
        first = store.import_pcd("warehouse", source)
        source.write_bytes(b"second")
        second = store.import_pcd("warehouse", source)

        self.assertEqual(first.version, 1)
        self.assertEqual(second.version, 2)
        self.assertEqual(first.pcd_path.read_bytes(), b"first")
        self.assertEqual(store.current("warehouse").pcd_path.read_bytes(), b"second")
        self.assertEqual(store.version("warehouse", 1), first)

    def test_import_can_stage_version_without_switching_current(self):
        source = _pcd(self.root / "source.pcd", b"first")
        store = MapStore(self.root / "maps")
        first = store.import_pcd("warehouse", source)
        source.write_bytes(b"candidate")
        candidate = store.import_pcd("warehouse", source, publish_current=False)

        self.assertEqual(candidate.version, 2)
        self.assertEqual(store.current("warehouse"), first)
        self.assertEqual(store.version("warehouse", 2), candidate)

    def test_cancel_during_copy_publishes_no_version_and_keeps_current(self):
        source = _pcd(self.root / "source.pcd", b"first")
        store = MapStore(self.root / "maps")
        current = store.import_pcd("warehouse", source)
        source.write_bytes(b"x" * (3 * 1024 * 1024))
        observations = 0

        def cancel_requested():
            nonlocal observations
            observations += 1
            return observations >= 3

        with self.assertRaises(MapStoreImportCancelled):
            store.import_pcd(
                "warehouse", source, cancel_requested=cancel_requested
            )

        self.assertEqual(store.current("warehouse"), current)
        versions = self.root / "maps/warehouse/versions"
        self.assertEqual(
            sorted(path.name for path in versions.iterdir()), ["v000001"]
        )
        self.assertFalse(any(path.name.startswith(".tmp-") for path in versions.iterdir()))

    def test_catalog_returns_only_verified_current_versions_newest_first(self):
        source = _pcd(self.root / "source.pcd", b"first")
        store = MapStore(self.root / "maps")
        first = store.import_pcd("factory-a", source)
        source.write_bytes(b"second")
        second = store.import_pcd("factory-b", source)

        catalog = store.list_current()

        self.assertEqual([item.map_id for item in catalog], ["factory-b", "factory-a"])
        by_id = {item.map_id: item for item in catalog}
        self.assertEqual(by_id["factory-a"].version, first.version)
        self.assertEqual(by_id["factory-a"].checksum, first.checksum)
        self.assertEqual(by_id["factory-b"].size_bytes, second.size_bytes)
        self.assertTrue(by_id["factory-a"].created_at)

    def test_catalog_omits_corrupt_and_symlinked_assets(self):
        source = _pcd(self.root / "source.pcd")
        store = MapStore(self.root / "maps")
        corrupt = store.import_pcd("corrupt", source)
        corrupt.pcd_path.write_bytes(b"changed")
        outside = self.root / "outside-map"
        outside.mkdir()
        (store.root / "escaped").symlink_to(outside, target_is_directory=True)

        self.assertEqual(store.list_current(), [])

    def test_repeated_catalog_verifies_unchanged_pcd_only_once(self):
        store = MapStore(self.root / "maps")
        store.import_pcd("plant-a", _pcd(self.root / "source.pcd"))
        with mock.patch.object(map_store, "_hash_stream", wraps=map_store._hash_stream) as hash_file:
            first = store.list_current()
            self.assertEqual(store.list_current(), first)
            self.assertEqual(len(first), 1)
            self.assertEqual(hash_file.call_count, 1)

    def test_cached_catalog_rejects_same_size_edit_with_restored_mtime(self):
        store = MapStore(self.root / "maps")
        asset = store.import_pcd("plant-a", _pcd(self.root / "source.pcd", b"first"))
        self.assertEqual(len(store.list_current()), 1)
        before = asset.pcd_path.stat()
        asset.pcd_path.write_bytes(b"other")
        os.utime(asset.pcd_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(store.list_current(), [])

    def test_cached_catalog_rejects_manifest_checksum_change(self):
        store = MapStore(self.root / "maps")
        asset = store.import_pcd("plant-a", _pcd(self.root / "source.pcd"))
        self.assertEqual(len(store.list_current()), 1)
        manifest = json.loads(asset.manifest_path.read_text())
        manifest["checksum"] = "0" * 64
        asset.manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(store.list_current(), [])

    def test_cached_catalog_revalidates_replaced_file(self):
        store = MapStore(self.root / "maps")
        asset = store.import_pcd("plant-a", _pcd(self.root / "source.pcd", b"first"))
        self.assertEqual(len(store.list_current()), 1)
        replacement = _pcd(self.root / "replacement.pcd", b"other")
        os.replace(replacement, asset.pcd_path)
        self.assertEqual(store.list_current(), [])

    def test_invalid_map_id_is_rejected(self):
        source = _pcd(self.root / "source.pcd")
        for map_id in ("", "../escape", "with/slash", " space"):
            with self.subTest(map_id=map_id), self.assertRaises(MapStoreError):
                MapStore(self.root / "maps").import_pcd(map_id, source)

    def test_empty_source_is_rejected_without_publishing_current(self):
        source = _pcd(self.root / "empty.pcd", b"")
        store = MapStore(self.root / "maps")
        with self.assertRaises(MapStoreError):
            store.import_pcd("plant-a", source)
        self.assertFalse((self.root / "maps/plant-a/current").exists())

    def test_verify_detects_corrupted_pcd(self):
        source = _pcd(self.root / "source.pcd")
        store = MapStore(self.root / "maps")
        asset = store.import_pcd("plant-a", source)
        asset.pcd_path.write_bytes(b"corrupted")

        with self.assertRaisesRegex(MapStoreError, "checksum mismatch"):
            store.current("plant-a")

    def test_current_pointer_cannot_escape_version_store(self):
        store = MapStore(self.root / "maps")
        map_directory = self.root / "maps/plant-a"
        map_directory.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (map_directory / "current").symlink_to(outside)

        with self.assertRaisesRegex(MapStoreError, "escapes version store"):
            store.current("plant-a", verify=False)


if __name__ == "__main__":
    unittest.main()
