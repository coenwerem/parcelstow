"""Installation script for the 'parcelstow' python package."""

import os

import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "psutil",
    # realized-contact certificate scoring (mdp/ferrari_canny.py)
    "scipy",
]

# Dependency tiers beyond the Isaac Lab environment (torch, gymnasium,
# and scipy arrive with Isaac Lab). The analysis extra covers the
# no-simulator record reproduction, the diffusion extra covers the
# Diffusion Policy baseline, ACT and DAgger need torch only.
EXTRAS_REQUIRE = {
    "analysis": ["numpy", "scipy", "matplotlib"],
    "diffusion": ["diffusers"],
    "hub": ["huggingface_hub"],
}
EXTRAS_REQUIRE["all"] = sorted({d for ds in EXTRAS_REQUIRE.values() for d in ds})

setup(
    name="parcelstow",
    packages=["parcelstow"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
