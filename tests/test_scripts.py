import os.path
import re

import numpy as np
import pytest
from click.testing import CliRunner

from neso_fame.scripts import simple

FLOAT = r"(-?\d\.\d+e[+-]\d\d)"
VERTICES = re.compile(
    r'<\s*V\s+ID="\d+"\s*>\s*'
    + FLOAT
    + r"\s+"
    + FLOAT
    + r"\s+"
    + FLOAT
    + r"\s*</\s*V\s*>",
    re.I,
)
ZONES = re.compile(r'<\s*F\s+ID="\d+"\s+DOMAIN="D\[\d+\]"\s*/>')
INTERFACES = re.compile(r'<\s*INTERFACE\s+NAME=".*"\s*>')


def test_2d_defaults() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "defaults.xml"
        result = runner.invoke(simple, ["2d", meshfile])
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    vertices = [
        (float(v[1]), float(v[2]), float(v[3])) for v in VERTICES.finditer(output)
    ]
    assert len(vertices) == 44
    assert max(v[0] for v in vertices) == 1.0
    assert min(v[0] for v in vertices) == 0.0
    assert max(v[1] for v in vertices) == 1.0
    assert min(v[1] for v in vertices) == 0.0
    assert all(v[2] == 0.0 for v in vertices)
    assert len(ZONES.findall(output)) == 2
    assert len(INTERFACES.findall(output)) == 1


@pytest.mark.parametrize(
    "x1min,x1max,x2min,x2max",
    [(-1.0, 1.0, 2.0, 5), (0.1, 0.7, -2.0, 0.0), (0.0, 100.0, 0.0, 100.0)],
)
def test_2d_limits(x1min: int, x1max: int, x2min: int, x2max: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "limits.xml"
        result = runner.invoke(
            simple,
            [
                "2d",
                "--x1-extent",
                str(x1min),
                str(x1max),
                "--x2-extent",
                str(x2min),
                str(x2max),
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    vertices = [
        (float(v[1]), float(v[2]), float(v[3])) for v in VERTICES.finditer(output)
    ]
    assert max(v[0] for v in vertices) == x1max
    assert min(v[0] for v in vertices) == x1min
    assert max(v[1] for v in vertices) == x2max
    assert min(v[1] for v in vertices) == x2min
    assert all(v[2] == 0.0 for v in vertices)


@pytest.mark.parametrize(
    "nx1,nx2,layers", [(4, 8, 4), (1, 8, 8), (2, 12, 0), (3, 5, 1), (7, 16, 4)]
)
def test_2d_resolution(nx1: int, nx2: int, layers: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "resolution.xml"
        result = runner.invoke(
            simple,
            [
                "2d",
                "--nx1",
                str(nx1),
                "--layers",
                str(layers),
                "--nx2",
                str(nx2),
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    if layers == 0:
        actual_layers = nx2
    else:
        actual_layers = layers
    assert (
        len(VERTICES.findall(output))
        == (nx1 + 1) * (nx2 // actual_layers + 1) * actual_layers
    )
    if actual_layers > 1:
        assert len(ZONES.findall(output)) == actual_layers
        assert len(INTERFACES.findall(output)) == actual_layers - 1
    else:
        assert len(ZONES.findall(output)) == 0
        assert len(INTERFACES.findall(output)) == 0


@pytest.mark.parametrize("layers", [0, 1])
def test_2d_periodic(layers: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "periodic.xml"
        result = runner.invoke(
            simple, ["2d", "--periodic", "--layers", str(layers), meshfile]
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    actual_layers = layers or 2
    assert (
        len(VERTICES.findall(output)) == 11 * (2 // actual_layers + 1) * actual_layers
    )
    assert len(ZONES.findall(output)) == actual_layers
    assert len(INTERFACES.findall(output)) == actual_layers


def test_2d_invalid_layers() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "invalid.xml"
        result = runner.invoke(simple, ["2d", "--nx2", "8", "--layers", "3", meshfile])
        assert result.return_value != 0
        assert not os.path.exists(meshfile)


def test_2d_angled_field() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "angled.xml"
        result = runner.invoke(
            simple,
            [
                "2d",
                "--nx1",
                "2",
                "--nx2",
                "1",
                "--x1-extent",
                "0",
                "2",
                "--x2-extent",
                "0",
                "2",
                "--angle",
                str(np.arctan(0.5) * 180 / np.pi),
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    vertices = [
        (float(v[1]), float(v[2]), float(v[3])) for v in VERTICES.finditer(output)
    ]
    assert (2.0, 1.5, 0) in vertices
    assert (0.0, 0.5, 0) in vertices
