import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from omni_slam_manager.map_store import MapStore, MapStoreError


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
