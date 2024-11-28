from __future__ import annotations

import itertools
import pathlib
from tempfile import TemporaryDirectory

import meshio  # type: ignore
import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis.strategies import (
    from_type,
    frozensets,
    lists,
    one_of,
    sampled_from,
    text,
)
from pytest import approx, mark

from neso_fame import meshio_writer
from neso_fame.coordinates import (
    Coord,
    CoordinateSystem,
    FrozenCoordSet,
    SliceCoord,
)
from neso_fame.mesh import (
    AcrossFieldCurve,
    FieldAlignedCurve,
    Prism,
    PrismMesh,
    PrismTypes,
    control_points,
    order,
)
from tests.test_nektar_writer import poloidal_corners

from .conftest import (
    across_field_curves,
    prism_meshes,
    simple_field_aligned_curve,
)

element_types = sampled_from(
    list(
        itertools.chain.from_iterable(d.values() for d in meshio_writer._ELEMENT_TYPES)
    )
)


def assert_points_eq(actual: npt.NDArray, expected: SliceCoord | Coord) -> None:
    if isinstance(expected, SliceCoord):
        expected = expected.to_3d_coord(0.0)
    for a, e in zip(actual, expected.to_cartesian()):
        assert a == approx(e, 1e-8, 1e-8, True)


cellsets = frozensets(text())


def line_name(order: int) -> str:
    return f"line{order + 1}" if order > 1 else "line"


def triangle_name(order: int) -> str:
    return f"triangle{(order + 1) * (order + 2) // 2}" if order > 1 else "triangle"


def quad_name(order: int) -> str:
    return f"quad{(order + 1) ** 2}" if order > 1 else "quad"


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(one_of((from_type(SliceCoord), from_type(Coord))), element_types, cellsets)
def test_point(coord: Coord, element: str, csets: frozenset[str]) -> None:
    mesh_data = meshio_writer.MeshioData()
    point_id = mesh_data.point(coord, element, csets)
    meshio = mesh_data.meshio()
    assert meshio.points.shape[0] == 1
    assert_points_eq(meshio.points[point_id], coord)


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(
    lists(from_type(Coord), min_size=2, max_size=2, unique=True).filter(
        lambda x: x[0].to_cartesian() != x[1].to_cartesian()
    ),
    lists(element_types, min_size=2, max_size=2, unique=True),
    lists(cellsets, min_size=2, max_size=2, unique=True),
)
def test_point_caching(
    coords: list[Coord], element: list[str], csets: list[frozenset[str]]
) -> None:
    mesh_data = meshio_writer.MeshioData()
    c1 = mesh_data.point(coords[0], element[0], csets[0])
    assert c1 == mesh_data.point(coords[0], element[0], csets[0])
    assert c1 != mesh_data.point(coords[1], element[0], csets[0])
    assert c1 != mesh_data.point(coords[0], element[0], csets[1])
    assert c1 != mesh_data.point(coords[0], element[1], csets[0])
    assert mesh_data.meshio().points.shape[0] == 4


@given(
    one_of((simple_field_aligned_curve, across_field_curves)),
    cellsets,
)
def test_line(
    curve: FieldAlignedCurve | AcrossFieldCurve, layer: frozenset[str]
) -> None:
    mesh_data = meshio_writer.MeshioData()
    mesh_data.line(curve, layer)
    meshio = mesh_data.meshio()
    cells = meshio.cells_dict
    assert len(cells) == 1
    n = order(curve)
    shape = line_name(n)
    assert shape in cells
    line_points = cells[shape][0]
    assert len(line_points) == n + 1
    points = control_points(curve)
    assert_points_eq(meshio.points[line_points[0]], points[0])
    assert_points_eq(meshio.points[line_points[1]], points[-1])
    for i in range(1, n):
        assert_points_eq(meshio.points[line_points[i + 1]], points[i])


@given(from_type(Prism), cellsets)
def test_poloidal_face(solid: Prism, layer: frozenset[str]) -> None:
    mesh_data = meshio_writer.MeshioData()
    mesh_data.poloidal_face(solid, layer)
    meshio = mesh_data.meshio()
    cells = meshio.cells_dict
    assert len(cells) == 1
    n = order(solid)
    if solid.shape == PrismTypes.TRIANGULAR:
        shape = triangle_name(n)
        n = 3
    else:
        shape = quad_name(n)
        n = 4
    assert shape in cells
    corners = cells[shape][0][:n]
    expected = FrozenCoordSet(c.to_cartesian() for c in poloidal_corners(solid))
    actual = FrozenCoordSet(
        Coord(*meshio.points[i], CoordinateSystem.CARTESIAN)  # type: ignore
        for i in corners
    )
    assert actual == expected


@settings(deadline=None, report_multiple_bugs=False)
@given(prism_meshes)
def test_poloidal_elements(mesh: PrismMesh) -> None:
    meshio = meshio_writer.meshio_poloidal_elements(mesh)
    cells = meshio.cells_dict
    n = order(next(iter(mesh)))
    lines = line_name(n)
    quads = quad_name(n)
    tris = triangle_name(n)
    empty = np.empty((0, 4))
    quad_elems = cells.get(quads, empty)
    triangle_elems = cells.get(tris, empty)
    assert (
        len(mesh.reference_layer.reference_elements)
        == quad_elems.shape[0] + triangle_elems.shape[0]
    )
    element_corners = frozenset(
        tuple(quad_elems[i, :4]) for i in range(quad_elems.shape[0])
    ) | frozenset(tuple(triangle_elems[i, :3]) for i in range(triangle_elems.shape[0]))
    expected_elements = frozenset(
        FrozenCoordSet(c.to_cartesian() for c in poloidal_corners(element))
        for element in mesh.reference_layer
    )
    actual_elements = frozenset(
        FrozenCoordSet(
            Coord(*meshio.points[i], CoordinateSystem.CARTESIAN)  # type: ignore
            for i in element
        )
        for element in element_corners
    )
    assert expected_elements == actual_elements
    expected_bounds = frozenset(
        FrozenCoordSet(
            [
                b.nodes.start_points[0].to_3d_coord(0.0).to_cartesian(),
                b.nodes.start_points[-1].to_3d_coord(0.0).to_cartesian(),
            ]
        )
        for b in itertools.chain.from_iterable(mesh.reference_layer.boundaries())
    )
    actual_bounds = frozenset(
        FrozenCoordSet(
            Coord(*meshio.points[i], CoordinateSystem.CARTESIAN)  # type: ignore
            for i in cells[lines][i, :2]
        )
        for i in range(cells[lines].shape[0])
    )
    assert expected_bounds == actual_bounds


@settings(deadline=None)
@given(prism_meshes)
def test_write_poloidal_elements(mesh: PrismMesh) -> None:
    with TemporaryDirectory() as tmp_path:
        msh_file = pathlib.Path(tmp_path) / "output.msh"
        meshio_writer.write_poloidal_mesh(mesh, str(msh_file), "gmsh")
        data = meshio.read(msh_file, "gmsh")
    cells = data.cells_dict
    n = order(next(iter(mesh)))
    q = quad_name(n)
    t = triangle_name(n)
    s = line_name(n)
    assert q in cells or t in cells
    assert s in cells
    assert sum(len(b) for b in mesh.reference_layer.bounds) == cells[s].shape[0]
    assert len(mesh.reference_layer.reference_elements) == (
        cells[q].shape[0] if q in cells else 0
    ) + (cells[t].shape[0] if t in cells else 0)