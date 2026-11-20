"""SLAM-02: per-scan match buffers must follow the scan size, not 100000.

The static checks run everywhere. When a host C++ compiler is available the
real ``ensure_scan_buffers`` body is extracted from laserMapping.cpp and
exercised under AddressSanitizer with a scan larger than the old fixed size.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE_ROOT / "src" / "laserMapping.cpp"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class ScanBufferBoundsTests(unittest.TestCase):
    def test_no_fixed_size_per_point_arrays(self):
        source = _source()
        self.assertIsNone(re.search(r"res_last\s*\[\s*\d+\s*\]\s*=", source))
        self.assertIsNone(re.search(r"point_selected_surf\s*\[\s*\d+\s*\]\s*=", source))
        self.assertNotIn("memset(point_selected_surf", source)
        self.assertNotIn("memset(res_last", source)

    def test_match_model_sizes_effect_clouds_before_indexed_writes(self):
        model = _function(_source(), "void h_share_model(")
        self.assertNotIn("laserCloudOri->clear()", model)
        self.assertNotIn("corr_normvect->clear()", model)
        ensure = model.index("ensure_scan_buffers(feats_down_size)")
        resize = model.index("laserCloudOri->resize(feats_down_size)")
        first_write = model.index("laserCloudOri->points[effct_feat_num]")
        self.assertLess(ensure, first_write)
        self.assertLess(resize, first_write)
        self.assertIn("laserCloudOri->resize(effct_feat_num)", model)

    def test_frame_path_grows_buffers_with_scan(self):
        source = _source()
        normvec = source.index("normvec->resize(feats_down_size);")
        self.assertIn(
            "ensure_scan_buffers(feats_down_size);",
            source[normvec:normvec + 200],
        )

    @unittest.skipUnless(shutil.which("g++"), "host g++ is not available")
    def test_ensure_scan_buffers_covers_scans_above_old_limit(self):
        helper = _function(_source(), "void ensure_scan_buffers(int point_count)")
        program = (
            "#include <cstdint>\n#include <cstdio>\n#include <vector>\n"
            "std::vector<float> res_last;\n"
            "std::vector<std::uint8_t> point_selected_surf;\n"
            + helper +
            "\nint main() {\n"
            "  ensure_scan_buffers(0);\n"
            "  if (!point_selected_surf.empty()) return 1;\n"
            "  for (int scan : {1000, 250000, 5000, 400000}) {\n"
            "    ensure_scan_buffers(scan);\n"
            "    for (int i = 0; i < scan; ++i) {\n"
            "      point_selected_surf[i] = 0U; res_last[i] = 1.0f;\n"
            "    }\n"
            "  }\n"
            "  if (point_selected_surf.size() != 400000U) return 2;\n"
            "  ensure_scan_buffers(400001);\n"
            "  if (point_selected_surf[400000] != 1U || res_last[400000] != -1000.0f) return 3;\n"
            "  return 0;\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "scan_buffers.cpp"
            binary = Path(directory) / "scan_buffers"
            source_path.write_text(program, encoding="utf-8")
            subprocess.run(
                ["g++", "-std=c++17", "-fsanitize=address,undefined",
                 "-fno-sanitize-recover=all", "-o", str(binary), str(source_path)],
                check=True,
            )
            result = subprocess.run([str(binary)], check=False)
            self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
