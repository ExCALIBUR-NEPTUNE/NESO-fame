import os.path
import re
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from hypnotoad.geqdsk._geqdsk import write as write_geqdsk  # type: ignore

from neso_fame.mesh import StraightLineAcrossField
from neso_fame.scripts import hypnotoad, simple
from tests.conftest import simple_trace
from tests.test_hypnotoad import CONNECTED_DOUBLE_NULL, LOWER_SINGLE_NULL, eqdsk_data

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
TRIANGLES = re.compile(r'<\s*T\s+ID="\d+"\s*>\s*\d+\s+\d+\s+\d+\s*<.\s*T\s*>')
ZONES = re.compile(r'<\s*F\s+ID="\d+"\s+DOMAIN="D\[\d+\]"\s*/>')
INTERFACES = re.compile(r'<\s*INTERFACE\s+NAME=".*"\s*>')
CURVES = re.compile(
    r'<\s*E\s+ID="\d+"\s+EDGEID="\d+"\s+TYPE="\w+"\s+NUMPOINTS="(\d+)"\s*>\s*'
    + FLOAT
    + r"(?:\s+"
    + FLOAT
    + ")*"
    + r"\s*</\s*E\s*>",
    re.I,
)


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
    "nx1,nx2,layers", [(8, 4, 4), (8, 1, 8), (12, 2, 0), (5, 3, 1), (16, 7, 4)]
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
        actual_layers = nx1
    else:
        actual_layers = layers
    assert (
        len(VERTICES.findall(output))
        == (nx2 + 1) * (nx1 // actual_layers + 1) * actual_layers
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
                "1",
                "--nx2",
                "2",
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


def test_3d_defaults() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "defaults.xml"
        result = runner.invoke(simple, ["3d", meshfile])
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    vertices = [
        (float(v[1]), float(v[2]), float(v[3])) for v in VERTICES.finditer(output)
    ]
    assert len(vertices) == 484
    assert max(v[0] for v in vertices) == 1.0
    assert min(v[0] for v in vertices) == 0.0
    assert max(v[1] for v in vertices) == 1.0
    assert min(v[1] for v in vertices) == 0.0
    assert max(v[2] for v in vertices) == 1.0
    assert min(v[2] for v in vertices) == 0.0
    assert len(ZONES.findall(output)) == 2
    assert len(INTERFACES.findall(output)) == 1


@pytest.mark.parametrize(
    "x1min,x1max,x2min,x2max,x3min,x3max",
    [
        (-1.0, 1.0, 2.0, 5, -22.0, -21.0),
        (0.1, 0.7, -2.0, 0.0, 100.12, 234.5),
        (0.0, 100.0, 0.0, 100.0, -1.0, 1.0),
    ],
)
def test_3d_limits(
    x1min: int, x1max: int, x2min: int, x2max: int, x3min: int, x3max: int
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "limits.xml"
        result = runner.invoke(
            simple,
            [
                "3d",
                "--x1-extent",
                str(x1min),
                str(x1max),
                "--x2-extent",
                str(x2min),
                str(x2max),
                "--x3-extent",
                str(x3min),
                str(x3max),
                "--nx2",
                "3",
                "--nx3",
                "3",
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
    assert max(v[2] for v in vertices) == x3max
    assert min(v[2] for v in vertices) == x3min


@pytest.mark.parametrize(
    "nx1,nx2,nx3,layers", [(8, 2, 3, 4), (8, 1, 1, 8), (12, 1, 2, 0), (5, 4, 1, 1)]
)
def test_3d_resolution(nx1: int, nx2: int, nx3: int, layers: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "resolution.xml"
        result = runner.invoke(
            simple,
            [
                "3d",
                "--nx1",
                str(nx1),
                "--layers",
                str(layers),
                "--nx2",
                str(nx2),
                "--nx3",
                str(nx3),
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    if layers == 0:
        actual_layers = nx1
    else:
        actual_layers = layers
    assert (
        len(VERTICES.findall(output))
        == (nx3 + 1) * (nx2 + 1) * (nx1 // actual_layers + 1) * actual_layers
    )
    if actual_layers > 1:
        assert len(ZONES.findall(output)) == actual_layers
        assert len(INTERFACES.findall(output)) == actual_layers - 1
    else:
        assert len(ZONES.findall(output)) == 0
        assert len(INTERFACES.findall(output)) == 0


@pytest.mark.parametrize("layers", [0, 1])
def test_3d_periodic(layers: int) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "periodic.xml"
        result = runner.invoke(
            simple,
            [
                "3d",
                "--periodic",
                "--layers",
                str(layers),
                "--nx2",
                "2",
                "--nx3",
                "2",
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    actual_layers = layers or 2
    assert len(VERTICES.findall(output)) == 9 * (2 // actual_layers + 1) * actual_layers
    assert len(ZONES.findall(output)) == actual_layers
    assert len(INTERFACES.findall(output)) == actual_layers


def test_3d_invalid_layers() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "invalid.xml"
        result = runner.invoke(simple, ["3d", "--nx1", "8", "--layers", "3", meshfile])
        assert result.return_value != 0
        assert not os.path.exists(meshfile)


def test_3d_angled_field() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        meshfile = "angled.xml"
        result = runner.invoke(
            simple,
            [
                "3d",
                "--nx1",
                "1",
                "--nx2",
                "2",
                "--nx3",
                "2",
                "--x1-extent",
                "0",
                "2",
                "--x2-extent",
                "0",
                "2",
                "--x3-extent",
                "0",
                "2",
                "--angle1",
                str(np.arctan(0.5) * 180 / np.pi),
                "--angle2",
                str(np.arctan(0.25) * 180 / np.pi),
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    vertices = [
        (float(v[1]), float(v[2]), float(v[3])) for v in VERTICES.finditer(output)
    ]
    assert (2.0, 1.5, 1.25) in vertices
    assert (0.0, 0.5, 0.75) in vertices


@pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_tokamak_field() -> None:
    runner = CliRunner()
    eqdsk_info = eqdsk_data(15, 15, (1.0, 2.0), (-1.0, 1.0), LOWER_SINGLE_NULL[0])
    hypno_settings = dict(LOWER_SINGLE_NULL[1]) | {
        "refine_atol": 1e-10,
        "follow_perpendicular_rtol": 1e-10,
        "follow_perpendicular_atol": 1e-10,
    }
    with runner.isolated_filesystem():
        meshfile = "tokamak_segment.xml"
        hypnofile = "hypnotoad.yaml"
        eqdsk = "eqdsk.g"
        with open(eqdsk, "w") as f:
            write_geqdsk(eqdsk_info, f)
        with open(hypnofile, "w") as f:
            yaml.dump(hypno_settings, f)
        result = runner.invoke(
            hypnotoad,
            [
                "--n",
                "3",
                "--toroidal_limits",
                "0",
                str(0.001 * np.pi),  # Don't extrude very far to keep run-times quick
                "--order",
                "2",
                "--core",
                "--config",
                hypnofile,
                eqdsk,
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    assert len(VERTICES.findall(output)) == 77 * 6
    assert len(TRIANGLES.findall(output)) == 4 * 6
    assert len(ZONES.findall(output)) == 3
    assert len(INTERFACES.findall(output)) == 2
    for curve in CURVES.finditer(output):
        n = int(curve[1])
        assert n == 3
        assert len(curve.groups()) == 5


# Patch the various hypnotoad interface methods to keep run-times short
@patch(
    "neso_fame.generators.equilibrium_trace",
    lambda _: simple_trace,
)
@patch(
    "neso_fame.element_builder.flux_surface_edge",
    lambda _, north, south: StraightLineAcrossField(north, south),
)
@patch(
    "neso_fame.element_builder.perpendicular_edge",
    lambda _, north, south: StraightLineAcrossField(north, south),
)
@pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_tokamak_periodic() -> None:
    runner = CliRunner()
    eqdsk_info = eqdsk_data(15, 15, (1.0, 2.0), (-1.0, 1.0), CONNECTED_DOUBLE_NULL[0])
    with runner.isolated_filesystem():
        meshfile = "complete_tokamak.xml"
        eqdsk = "eqdsk.g"
        with open(eqdsk, "w") as f:
            write_geqdsk(eqdsk_info, f)
        result = runner.invoke(
            hypnotoad,
            [
                "--n",
                "10",
                "--layers",
                "5",
                "--order",
                "1",
                "--compress",
                eqdsk,
                meshfile,
            ],
        )
        assert result.exit_code == 0
        with open(meshfile, "r") as f:
            output = f.read()
    assert len(TRIANGLES.findall(output)) == 0
    assert len(ZONES.findall(output)) == 5
    assert len(INTERFACES.findall(output)) == 5
    assert len(CURVES.findall(output)) == 0
