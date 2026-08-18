from glob import glob
import os

from setuptools import find_packages, setup


package_name = "omni_slam_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Omni maintainers",
    maintainer_email="yan@example.com",
    description="Process supervision and versioned map assets for omni_slam.",
    license="Apache-2.0",
    tests_require=["pytest"],
    scripts=["scripts/omni_slam_ctl"],
    entry_points={
        "console_scripts": [
            "omni_slam_manager = omni_slam_manager.manager_node:main",
        ],
    },
)
