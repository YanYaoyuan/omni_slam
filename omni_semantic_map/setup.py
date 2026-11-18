from setuptools import setup
from glob import glob

setup(
    name="omni_semantic_map",
    version="0.1.0",
    description="Keyframe-anchored semantic mapping and evidence-authorized object revision",
    python_requires=">=3.10",
    packages=["omni_semantic_map"],
    # The graph kernels compile beside a caller-selected cache at runtime.
    # Wheel/colcon installs must retain their source, not only the Python wrappers.
    package_data={"omni_semantic_map": ["*.cpp"]},
    install_requires=["numpy", "scipy", "PyYAML"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/omni_semantic_map"]),
        ("share/omni_semantic_map", ["package.xml"]),
        ("share/omni_semantic_map/launch", glob("launch/*.launch.py")),
        ("share/omni_semantic_map/config", glob("config/*")),
    ],
    entry_points={"console_scripts": ["semantic_map=omni_semantic_map.cli:main",
        "semantic_trial=omni_semantic_map.live_node:main",
        "semantic_categories=omni_semantic_map.category_node:main",
        "semantic_preflight=omni_semantic_map.preflight:main"]},
)
