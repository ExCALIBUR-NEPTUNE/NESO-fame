#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import setuptools

name = "neso_fame"
root_path = Path(__file__).parent
init_path = root_path.joinpath(name, "__init__.py")
readme_path = root_path.joinpath("README.md")

with readme_path.open("r") as f:
    long_description = f.read()

setuptools.setup(
    name=name,
    author="Crown Copyright",
    description="Field-aligned mesh extrusion for the NEPTUNE project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ExCALIBUR-NEPTUNE/NESO-fame",
    project_urls={
        "Bug Tracker": "https://github.com/ExCALIBUR-NEPTUNE/NESO-fame/issues/",
        "Source Code": "https://github.com/ExCALIBUR-NEPTUNE/NESO-fame",
    },
    packages=setuptools.find_packages(),
    keywords=[
        "mesh",
        "plasma",
        "physics",
    ],
    use_scm_version=True,
    setup_requires=[
        "setuptools>=42",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "hypnotoad",
        "NekPy",
        "vtk",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        (
            "License :: OSI Approved :: "
            "GNU Lesser General Public License v3 or later (LGPLv3+)"
        ),
        "Operating System :: OS Independent",
    ],
)
