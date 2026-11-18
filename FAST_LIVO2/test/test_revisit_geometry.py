import sys
from pathlib import Path
import unittest

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from revisit_geometry import (
    gravity_rotation,
    normal_information,
    registration_cloud,
    validate_pair,
    optimize_graph,
)


class RevisitGeometryTest(unittest.TestCase):
    def test_gravity_rotation_preserves_lengths_and_does_not_add_arbitrary_yaw(self):
        up = np.array([0.1, 0, 1])
        up /= np.linalg.norm(up)
        r = gravity_rotation(up)
        np.testing.assert_allclose(r @ up, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1)
        self.assertAlmostEqual(Rotation.from_matrix(r).magnitude(), np.arccos(up[2]))
        np.testing.assert_allclose(
            gravity_rotation([0, 0, -1]) @ np.array([0, 0, -1]), [0, 0, 1]
        )

    def test_ground_plane_alone_cannot_verify_six_dof_loop(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(-5, 5, (500, 3))
        p[:, 2] = 0
        n = np.tile([0, 0, 1.0], (500, 1))
        _, ratio = normal_information(p, n)
        self.assertLess(ratio, 1e-10)

    def test_registration_retains_actual_height_difference(self):
        rng = np.random.default_rng(11)
        world = rng.uniform(-4, 4, (6000, 3))
        world[:2000, 0] = 6
        world[2000:4000, 1] = 5
        world[4000:, 2] = -1
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
        pose[:3, 3] = [0.4, -0.2, 2]
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(world)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector((world - pose[:3, 3]) @ pose[:3, :3])
        guess = pose.copy()
        guess[2, 3] += 2
        result, audit = validate_pair(
            registration_cloud(source),
            registration_cloud(target),
            guess,
            np.array([0, 0, 1.0]),
        )
        self.assertIsNotNone(result, audit)
        self.assertAlmostEqual(result[0][2, 3], 2, delta=0.05)
        self.assertGreater(audit["observability_ratio"], 1e-4)

    def test_pose_graph_closes_drift_without_flattening_real_slope(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, :3, 3] = np.array(
            [[0, 0, 0], [3, 0, 2], [3, 3, 3], [0, 3, 2], [0, 0, 0.0]]
        )
        poses[:, 2, 3] += np.arange(5) * 0.3
        loop = {
            "i": 4,
            "j": 0,
            "matrix": np.eye(4).tolist(),
            "information": (np.eye(6) * 1000).tolist(),
        }
        result, _ = optimize_graph(poses, [loop])
        self.assertLess(abs(result[-1, 2, 3]), 0.1)
        self.assertGreater(result[2, 2, 3], 2.5)


if __name__ == "__main__":
    unittest.main()
