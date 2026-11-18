"""One frozen raw-scan execution check; no labels, maps, or semantic scoring."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import utonia


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/benchmarks/utonia_evidence_v1"
SCAN = ROOT / ("build/datasets/kitti360_windows_v1/data_3d_raw/"
               "2013_05_28_drive_0000_sync/velodyne_points/data/0000000242.bin")
SEED = 6985480


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assets = json.loads((OUT / "assets.json").read_text())
    # Check all pinned original source files and the checkpoint before execution.
    for name, expected in assets["files"].items():
        assert digest(Path(name)) == expected, name
    assert torch.cuda.is_available()
    checkpoint = Path(assets["checkpoint"])
    protocol = dict(
        purpose="execution and repeatability only; not a semantic capability test",
        checkpoint_sha256=assets["checkpoint_sha256"],
        source_commit=assets["source_commit"], scan=str(SCAN),
        scan_sha256=digest(SCAN), seed=SEED, forwards=2,
        input="whole raw ego-frame Velodyne scan; zero color and normals",
        transform="native outdoor default(scale=0.2, apply_z_positive=False)",
        custom_config=dict(enable_flash=False, enc_patch_size=[1024] * 5),
        precision="float32; no autocast", upcast_levels=4,
        labels_read=0, new_maps=0, new_training=0,
        script_sha256=digest(Path(__file__)),
    )
    protocol_file = OUT / "execution_protocol.json"
    if protocol_file.exists():
        assert json.loads(protocol_file.read_text()) == protocol
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")
    result_file = OUT / "execution_check.json"
    if result_file.exists():
        raise RuntimeError("Execution check already exists; refusing to overwrite")
    raw = np.fromfile(SCAN, dtype=np.float32).reshape(-1, 4)
    assert np.isfinite(raw).all()
    xyz = raw[:, :3].copy()
    torch.cuda.set_per_process_memory_fraction(0.85)
    utonia.utils.set_seed(SEED)
    model = utonia.load(str(checkpoint), custom_config=protocol["custom_config"]).cuda().eval()
    transform = utonia.transform.default(0.2, apply_z_positive=False)
    first = None
    timings = []
    for attempt in range(2):
        utonia.utils.set_seed(SEED)
        point = transform(dict(coord=xyz.copy(), color=np.zeros_like(xyz), normal=np.zeros_like(xyz)))
        inverse = point["inverse"].numpy().copy()
        coords = point["coord"].numpy().copy()
        assert inverse.shape == (len(xyz),)
        assert inverse.min() >= 0 and inverse.max() < len(coords)
        error = np.linalg.norm(coords[inverse] - xyz * 0.2, axis=1)
        assert error.max() < np.sqrt(3) * 0.01 + 1e-5
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            point = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()}
            point = model(point)
            for _ in range(4):
                parent = point.pop("pooling_parent")
                pooling_inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[pooling_inverse]], dim=-1)
                point = parent
            assert "pooling_parent" not in point
            assert point.feat.shape == (len(coords), 1386)
            features = point.feat.cpu().numpy().copy()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        assert np.isfinite(features).all()
        if first is None:
            first = features
            first_inverse = inverse
            first_coords = coords
        else:
            assert np.array_equal(first_inverse, inverse)
            assert np.array_equal(first_coords, coords)
            max_abs = float(np.max(np.abs(first - features)))
            repeat_close = bool(np.allclose(first, features, rtol=1e-4, atol=1e-5))
        print(f"FORWARD {attempt + 1}: {features.shape}; {timings[-1]:.3f}s", flush=True)
        del point
    result = dict(
        execution_passed=True, repeat_allclose=repeat_close, repeat_max_abs=max_abs,
        raw_points=len(xyz), grid_points=len(coords), feature_width=features.shape[1],
        max_inverse_distance_scaled=float(error.max()), seconds=timings,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(), cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        native_strict_load=True, feature_sha256=hashlib.sha256(first.tobytes()).hexdigest(),
        protocol_sha256=digest(protocol_file),
        semantic_benefit="not tested", new_maps=0, new_scores=0,
    )
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert repeat_close, "Forward executed but numerical repeatability check failed"


if __name__ == "__main__":
    main()
