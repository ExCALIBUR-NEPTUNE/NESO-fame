import itertools
from collections.abc import Iterable, Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import basic_indices
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    integers,
    just,
    lists,
    one_of,
    sampled_from,
    shared,
    slices,
)

from neso_fame import coordinates, mesh
from neso_fame.offset import (
    Offset,
)

from .conftest import (
    _hex_mesh_arguments,
    _quad_mesh_elements,
    common_coords,
    common_slice_coords,
    compatible_coords_and_alignments,
    corners_to_poloidal_quad,
    flat_sided_hex,
    linear_traces,
    mesh_arguments,
    non_nans,
    orders,
    prism_mesh_layer_no_divisions,
    quad_mesh_layer_no_divisions,
    shared_coordinate_systems,
    subdivideable_hex,
    subdivideable_mesh_arguments,
    subdivideable_quad,
    unbroadcastable_shape,
    whole_numbers,
)

divisions = shared(integers(-5, 10), key=5545)
pos_divisions = shared(integers(1, 10), key=5546)


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    integers(1, 5),
)
def test_subdivideable_field_aligned_positions(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, field, alignments, order, num_divisions
    )
    assert data.start_points is starts
    assert np.isclose(data.x3[-1] * 2, dx3)
    assert data.x3.shape == (order * num_divisions + 1,)
    assert field is data.trace
    assert alignments is data.alignments
    assert data.subdivision == 0
    assert data.num_divisions == 1
    np.broadcast(*data.start_points, alignments, data._computed[..., 0])
    assert data._x1.shape[-1] == len(data.x3)
    assert data._x2.shape[-1] == len(data.x3)


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
)
def test_field_aligned_positions(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.field_aligned_positions(
        starts, dx3, field, alignments, order, subdivision, num_divisions
    )
    assert data.start_points is starts
    assert np.isclose(data.x3[-1] * 2, dx3)
    assert data.x3.shape == (order * num_divisions + 1,)
    assert field is data.trace
    assert alignments is data.alignments
    assert data.subdivision == subdivision
    assert data.num_divisions == num_divisions
    np.broadcast(*data.start_points, data._computed[..., 0])
    assert data._x1.shape[-1] == len(data.x3)
    assert data._x2.shape[-1] == len(data.x3)


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(-10, 0),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
)
def test_field_aligned_positions_bad_order(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    with pytest.raises(ValueError, match=r".*order.*"):
        mesh.field_aligned_positions(
            starts, dx3, field, alignments, order, subdivision, num_divisions
        )


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    integers(-10, 0),
)
def test_field_aligned_positions_bad_divisions(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    with pytest.raises(ValueError, match=r".*num_divisions.*"):
        mesh.field_aligned_positions(
            starts, dx3, field, alignments, order, 0, num_divisions
        )


@given(
    from_type(coordinates.SliceCoords),
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
    lists(integers(1, 10), min_size=1, max_size=5),
)
def test_field_aligned_positions_bad_alignment_dim(
    starts: coordinates.SliceCoords,
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
    extra_dims: list[int],
) -> None:
    alignments = np.ones(starts.shape + tuple(extra_dims))
    with pytest.raises(ValueError, match=r".*higher dimension.*"):
        mesh.field_aligned_positions(
            starts, dx3, field, alignments, order, subdivision, num_divisions
        )
    alignments = np.ones(tuple(extra_dims) + np.broadcast(*starts).shape)
    with pytest.raises(ValueError, match=r".*higher dimension.*"):
        mesh.field_aligned_positions(
            starts, dx3, field, alignments, order, subdivision, num_divisions
        )


shared_slice_coords = shared(
    from_type(coordinates.SliceCoords).filter(
        lambda x: len(x.shape) > 0 and 1 not in x.shape
    ),
    key=1111,
)


@given(
    shared_slice_coords,
    linear_traces,
    floats(0.1, 10.0),
    shared_slice_coords.flatmap(lambda x: unbroadcastable_shape(x.shape)).map(np.ones),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
)
def test_field_aligned_positions_bad_alignment_shape(
    starts: coordinates.SliceCoords,
    field: mesh.FieldTrace,
    dx3: float,
    alignments: npt.NDArray,
    order: int,
    subdivision: int,
    num_divisions: int,
) -> None:
    with pytest.raises(ValueError, match=r".*broadcast-compatible.*"):
        mesh.field_aligned_positions(
            starts, dx3, field, alignments, order, subdivision, num_divisions
        )


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    lists(integers(1, 4), min_size=1, max_size=4),
)
def test_field_aligned_positions_subdivide(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    division_sizes: list[int],
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, field, alignments, int(np.prod(division_sizes)), 1
    )
    base = 0
    num_divs = 1
    for n in division_sizes:
        base *= n
        num_divs *= n
        for m, array in enumerate(data.subdivide(n)):
            assert array.start_points is data.start_points
            assert array.x3 is data.x3
            assert array.trace is data.trace
            assert array.alignments is data.alignments
            assert array.subdivision == base + m
            assert array.num_divisions == num_divs
            assert array._x1 is data._x1
            assert array._x2 is data._x2
            assert array._computed is data._computed
        data = array
        base += m


def test_field_aligned_positions_bad_subdivide() -> None:
    starts = coordinates.SliceCoords(
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        coordinates.CoordinateSystem.CARTESIAN,
    )
    alignments = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    data = mesh.field_aligned_positions(starts, 2.0, sample_trace, alignments, 4, 0, 1)
    with pytest.raises(ValueError, match=r"Can not subdivide.*"):
        next(data.subdivide(3))


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    integers(1, 5),
    integers(-10, 1),
)
def test_field_aligned_positions_subdivide_self(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
    m: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, field, alignments, order, num_divisions
    )
    divisions = list(data.subdivide(m))
    assert len(divisions) == 1
    assert divisions[0] is data


shared_coords_alignments = shared(compatible_coords_and_alignments, key=22222)


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
)
def test_field_aligned_positions_poloidal_shape(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.field_aligned_positions(
        starts, dx3, field, alignments, order, subdivision, num_divisions
    )
    assert len(data.poloidal_shape) == len(starts.shape)
    assert (
        data.poloidal_shape
        == tuple(
            map(
                max,
                itertools.zip_longest(
                    starts.shape[::-1], alignments.shape[::-1], fillvalue=0
                ),
            )
        )[::-1]
    )


@settings(report_multiple_bugs=False)
@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
)
def test_field_aligned_positions_shape(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.field_aligned_positions(
        starts, dx3, field, alignments, order, subdivision, num_divisions
    )
    assert data.shape == data.coords.shape


@given(
    shared_coords_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions.flatmap(lambda x: integers(0, x - 1)),
    pos_divisions,
    shared_coords_alignments.flatmap(
        lambda x: basic_indices(x[0].shape)
        if len(x[0].shape) > 2
        else one_of((integers(0, len(x[0]) - 1), slices(len(x[0]))))
        if len(x[0].shape) == 1
        else just(())
    ),
)
def test_field_aligned_positions_getitem(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    subdivision: int,
    num_divisions: int,
    idx: Any,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.field_aligned_positions(
        starts, dx3, field, alignments, order, subdivision, num_divisions
    )
    result = data[idx]
    assert result.start_points is not data.start_points
    assert (
        result.start_points.x1.base is data.start_points.x1
        or not isinstance(result.start_points.x1, np.ndarray)
        or not isinstance(data.start_points.x1, np.ndarray)
    )
    assert (
        result.start_points.x2.base is data.start_points.x2
        or not isinstance(result.start_points.x1, np.ndarray)
        or not isinstance(data.start_points.x2, np.ndarray)
    )
    assert result.x3 is data.x3
    assert result.trace is data.trace
    assert result.alignments is not data.alignments
    assert result.alignments.base is data.alignments or not isinstance(
        result.start_points.x1, np.ndarray
    )
    assert result.subdivision == data.subdivision
    assert result.num_divisions == data.num_divisions
    assert result._x1 is not data._x1
    assert result._x1.base is data._x1
    assert result._x2 is not data._x2
    assert result._x2.base is data._x2
    assert result._computed is not data._computed
    assert result._computed.base is data._computed


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions,
)
def test_field_aligned_positions_coords_shape(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, field, alignments, order, num_divisions
    )
    coords = data.coords
    assert coords.shape == data.poloidal_shape + (order * num_divisions + 1,)
    for div in data.subdivide(num_divisions):
        assert div.coords.shape == data.poloidal_shape + (order + 1,)


def sample_trace(
    start: coordinates.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[coordinates.SliceCoords, npt.NDArray]:
    x3 = np.asarray(x3)
    return coordinates.SliceCoords(
        (1 - start_weight) * x3 + start.x1,
        2 * (1 - start_weight) * x3 + start.x2,
        start.system,
    ), np.sqrt(3 * (1 - start_weight)) * x3


@given(
    from_type(coordinates.SliceCoords),
    floats(0.1, 10.0),
    integers(1, 10),
)
def test_field_aligned_positions_coords_fully_aligned(
    starts: coordinates.SliceCoords,
    dx3: float,
    order: int,
) -> None:
    data = mesh.field_aligned_positions(
        starts, dx3, sample_trace, np.asarray(1.0), order, 0, 1
    )
    coords = data.coords
    for idx in itertools.product(*map(range, starts.shape)):
        c1, c2 = starts[idx]
        np.testing.assert_allclose(coords.x1[idx], c1 + data.x3, 1e-8, 1e-8)
        np.testing.assert_allclose(coords.x2[idx], c2 + 2 * data.x3, 1e-8, 1e-8)


@given(
    from_type(coordinates.SliceCoords),
    floats(0.1, 10.0),
    integers(1, 10),
)
def test_field_aligned_positions_coords_partially_aligned(
    starts: coordinates.SliceCoords,
    dx3: float,
    order: int,
) -> None:
    data = mesh.field_aligned_positions(
        starts, dx3, sample_trace, np.asarray(0.5), order, 0, 1
    )
    coords = data.coords
    for idx in itertools.product(*map(range, starts.shape)):
        c1, c2 = starts[idx]
        np.testing.assert_allclose(coords.x1[idx], c1 + 0.5 * data.x3, 1e-8, 1e-8)
        np.testing.assert_allclose(coords.x2[idx], c2 + data.x3, 1e-8, 1e-8)


@given(
    from_type(coordinates.SliceCoords),
    floats(0.1, 10.0),
    integers(1, 10),
)
def test_field_aligned_positions_coords_unaligned(
    starts: coordinates.SliceCoords,
    dx3: float,
    order: int,
) -> None:
    data = mesh.field_aligned_positions(
        starts, dx3, sample_trace, np.asarray(0.0), order, 0, 1
    )
    coords = data.coords
    for idx in itertools.product(*map(range, starts.shape)):
        c1, c2 = starts[idx]
        np.testing.assert_allclose(coords.x1[idx], c1, 1e-8, 1e-8)
        np.testing.assert_allclose(coords.x2[idx], c2, 1e-8, 1e-8)


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    pos_divisions,
)
def test_field_aligned_positions_coords_caching(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    trace = MagicMock()
    trace.side_effect = field
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, trace, alignments, order, num_divisions
    )
    trace.assert_not_called()
    coords = data.coords
    assert trace.call_count == len(starts)
    trace.reset_mock()
    coords2 = data.coords
    assert coords2 is coords
    trace.assert_not_called()
    for d in data.subdivide(num_divisions):
        assert d.trace is trace
        coords = d.coords
        if num_divisions > 1:
            assert coords is not coords2
        trace.assert_not_called()


@given(
    compatible_coords_and_alignments,
    linear_traces,
    floats(0.1, 10.0),
    integers(1, 10),
    integers(2, 6),
)
def test_field_aligned_positions_subdivided_coords(
    coords_alignemnts: tuple[coordinates.SliceCoords, npt.NDArray],
    field: mesh.FieldTrace,
    dx3: float,
    order: int,
    num_divisions: int,
) -> None:
    starts, alignments = coords_alignemnts
    trace = MagicMock()
    trace.side_effect = field
    data = mesh.subdividable_field_aligned_positions(
        starts, dx3, trace, alignments, order, num_divisions
    )
    coords = [d.coords for d in data.subdivide(num_divisions)]
    assert trace.call_count == len(starts)
    x1 = np.concatenate(
        [c.x1[..., 1:] if i != 0 else c.x1 for i, c in enumerate(coords)], axis=-1
    )
    if np.any(x1 != data.coords.x1):
        breakpoint()
    assert np.all(x1 == data.coords.x1)
    x2 = np.concatenate(
        [c.x2[..., 1:] if i != 0 else c.x2 for i, c in enumerate(coords)], axis=-1
    )
    assert np.all(x2 == data.coords.x2)
    x3 = np.concatenate(
        [c.x3[..., 1:] if i != 0 else c.x3 for i, c in enumerate(coords)], axis=-1
    )
    assert np.all(x3 == data.coords.x3)


def test_field_aligned_positions_slice_caching() -> None:
    starts = coordinates.SliceCoords(
        np.array([0.0, 0.5, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        coordinates.CoordinateSystem.CARTESIAN,
    )
    alignments = np.array([1.0, 1.0, 0.0])
    trace = MagicMock()
    trace.side_effect = sample_trace
    data = mesh.field_aligned_positions(starts, 2.0, trace, alignments, 4)
    near = data[0]
    expected_x1 = np.array(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [-0.5, 0.0, 0.5, 1.0, 1.5],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    expected_x2 = np.array(
        [
            [-3.0, -2.0, -1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    expected_x3 = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    near_coords = near.coords
    trace.assert_called_once()
    np.testing.assert_allclose(near_coords.x1, expected_x1[0, :], 1e-8, 1e-8)
    np.testing.assert_allclose(near_coords.x2, expected_x2[0, :], 1e-8, 1e-8)
    np.testing.assert_allclose(near_coords.x3, expected_x3[0, :], 1e-8, 1e-8)
    coords = data.coords
    assert trace.call_count == 3
    np.testing.assert_allclose(coords.x1, expected_x1, 1e-8, 1e-8)
    np.testing.assert_allclose(coords.x2, expected_x2, 1e-8, 1e-8)
    np.testing.assert_allclose(coords.x3, expected_x3, 1e-8, 1e-8)


def example_trace(
    start: coordinates.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[coordinates.SliceCoords, npt.NDArray]:
    a = -2.0
    b = -0.5
    c = 1 - start_weight
    return (
        coordinates.SliceCoords(
            c * np.asarray(x3) * a + start_weight * start.x1,
            c * np.asarray(x3) * b + start_weight * start.x2,
            coordinates.CoordinateSystem.CARTESIAN,
        ),
        np.sqrt(1.0 + c * c * (a * a + b * b)) * np.asarray(x3),
    )


def test_straight_line_across_field() -> None:
    line = mesh.control_points(
        mesh.straight_line_across_field(
            mesh.SliceCoord(1.0, 1.0, coordinates.CoordinateSystem.CARTESIAN),
            mesh.SliceCoord(2.0, 0.0, coordinates.CoordinateSystem.CARTESIAN),
            4,
        )
    )
    np.testing.assert_allclose(line.x1, np.linspace(1, 2, 5), 1e-10, 1e-10)
    np.testing.assert_allclose(line.x2, np.linspace(1, 0, 5), 1e-10, 1e-10)


def test_straight_line_across_field_mismatched_coords() -> None:
    with pytest.raises(ValueError):
        mesh.straight_line_across_field(
            mesh.SliceCoord(1.0, 1.0, coordinates.CoordinateSystem.CARTESIAN),
            mesh.SliceCoord(1.0, 1.0, coordinates.CoordinateSystem.CYLINDRICAL),
            3,
        )


def test_bad_straight_line() -> None:
    with pytest.raises(ValueError):
        mesh.straight_line(
            coordinates.Coord(0.0, 0.0, 0.0, coordinates.CoordinateSystem.CARTESIAN),
            coordinates.Coord(1.0, 2.0, 3.0, coordinates.CoordinateSystem.CYLINDRICAL),
            4,
        )


line_termini = sampled_from(coordinates.CoordinateSystem).flatmap(
    lambda c: lists(
        builds(mesh.Coord, whole_numbers, whole_numbers, whole_numbers, just(c)),
        min_size=2,
        max_size=2,
    )
)


@given(line_termini, orders, integers(0, 10), integers(1, 10))
def test_straight_line_between_termini(
    termini: list[coordinates.Coord], order: int, i: int, j: int
) -> None:
    line = mesh.straight_line(termini[0], termini[1], order)
    positions = mesh.control_points(line)
    for position in positions.iter_points():
        for p, n, s in zip(position, termini[0], termini[1]):
            xs = sorted([p, n, s])
            assert p == xs[1]


@given(from_type(mesh.Quad))
def test_quad_iter(q: mesh.Quad) -> None:
    sides = list(q)
    assert coordinates.FrozenCoordSet(
        sides[0].coords.iter_points()
    ) == coordinates.FrozenCoordSet(q.north.coords.iter_points())
    assert coordinates.FrozenCoordSet(
        sides[1].coords.iter_points()
    ) == coordinates.FrozenCoordSet(q.south.coords.iter_points())


@given(from_type(mesh.Quad), floats(0.0, 1.0))
def test_quad_north(q: mesh.Quad, s: float) -> None:
    actual = q.north.coords
    expected = q.nodes.coords.get[0, :]
    np.testing.assert_allclose(actual.x1, expected.x1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.x2, expected.x2, rtol=1e-12, atol=1e-12)


@given(from_type(mesh.Quad), floats(0.0, 1.0))
def test_quad_south(q: mesh.Quad, s: float) -> None:
    actual = q.south.coords
    expected = q.nodes.coords.get[-1, :]
    np.testing.assert_allclose(actual.x1, expected.x1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.x2, expected.x2, rtol=1e-12, atol=1e-12)


@settings(deadline=None)
@given(from_type(mesh.Quad))
def test_quad_near_edge(q: mesh.Quad) -> None:
    actual = coordinates.FrozenCoordSet(q.near.iter_points())
    expected = coordinates.FrozenCoordSet({q.north.coords[0], q.south.coords[0]})
    assert expected <= actual
    x3 = np.array([a.x3 for a in actual])
    np.testing.assert_allclose(x3, x3[0], 1e-8, 1e-8)


@settings(deadline=None)
@given(from_type(mesh.Quad))
def test_quad_far_edge(q: mesh.Quad) -> None:
    actual = coordinates.FrozenCoordSet(q.far.iter_points())
    expected = coordinates.FrozenCoordSet({q.north.coords[-1], q.south.coords[-1]})
    assert expected <= actual
    x3 = np.array([a.x3 for a in actual])
    np.testing.assert_allclose(x3, x3[0], 1e-8, 1e-8)


@given(from_type(mesh.Quad))
def test_quad_near_far_corners(q: mesh.Quad) -> None:
    expected = coordinates.FrozenCoordSet(q.corners())
    actual = coordinates.FrozenCoordSet([q.far[0], q.far[-1], q.near[0], q.near[-1]])
    assert expected == actual


@given(from_type(mesh.Quad))
def test_quad_corners(q: mesh.Quad) -> None:
    corners = list(q.corners())
    assert corners[0].approx_eq(q.north.coords[0])
    assert corners[1].approx_eq(q.south.coords[0])
    assert corners[2].approx_eq(q.north.coords[-1])
    assert corners[3].approx_eq(q.south.coords[-1])


@given(from_type(mesh.Quad))
def test_quad_control_points_spacing(q: mesh.Quad) -> None:
    # Check spacing in the direction along the field lines (holds true
    # even for quads that aren't field-aligned)
    cp = mesh.control_points(q)
    d_diff = cp.x3[:, 1:] - cp.x3[:, :-1]
    np.testing.assert_allclose(d_diff[0, 0], d_diff, rtol=1e-10, atol=1e-10)


@given(from_type(mesh.Quad), whole_numbers)
def test_quad_offset(q: mesh.Quad, x: float) -> None:
    actual = mesh.control_points(Offset(q, x))
    expected = mesh.control_points(q).offset(x)
    np.testing.assert_allclose(actual.x1, expected.x1, atol=1e-12)
    np.testing.assert_allclose(actual.x2, expected.x2, atol=1e-12)
    np.testing.assert_allclose(actual.x3, expected.x3, atol=1e-12)
    assert actual.system == expected.system


@given(divisions.flatmap(subdivideable_quad), divisions)
def test_quad_subdivision_len(quad: mesh.Quad, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = quad.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@settings(deadline=None)
@given(divisions.flatmap(subdivideable_quad), divisions)
def test_quad_subdivision(quad: mesh.Quad, divisions: int) -> None:
    divisions_iter = quad.subdivide(divisions)
    quad_corners = list(quad.corners())
    first = next(divisions_iter)
    corners = list(first.corners())
    assert corners[0].approx_eq(quad_corners[0])
    assert corners[1].approx_eq(quad_corners[1])
    prev = corners
    for quad in divisions_iter:
        corners = list(quad.corners())
        assert corners[0].approx_eq(prev[2])
        assert corners[1].approx_eq(prev[3])
        prev = corners
    assert prev[2].approx_eq(quad_corners[2])
    assert prev[3].approx_eq(quad_corners[3])


@given(from_type(mesh.Quad))
def test_quad_make_flat(q: mesh.Quad) -> None:
    flat = q.make_flat_quad()
    for corner, flat_corner in zip(q.corners(), flat.corners()):
        assert corner.approx_eq(flat_corner)
    assert corner.system == flat_corner.system


@given(from_type(mesh.Quad), integers(1, 5))
def test_quad_make_flat_idempotent(q: mesh.Quad, n: int) -> None:
    new_quad = q.make_flat_quad()
    expected = coordinates.FrozenCoordSet(mesh.control_points(new_quad).iter_points())
    for _ in range(n):
        new_quad = new_quad.make_flat_quad()
        assert (
            coordinates.FrozenCoordSet(mesh.control_points(new_quad).iter_points())
            == expected
        )


@given(from_type(mesh.Prism))
def test_prism_near_edge(h: mesh.Prism) -> None:
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable(
            s.nodes.coords.get[..., 0].iter_points() for s in h
        )
    )
    actual = coordinates.FrozenCoordSet(h.near.corners())
    assert actual <= expected
    x3 = np.array([a.x3 for a in actual])
    np.testing.assert_allclose(x3, x3[0], 1e-8, 1e-8)


@given(from_type(mesh.UnalignedShape))
def test_unaligned_shape(shape: mesh.UnalignedShape) -> None:
    actual = coordinates.FrozenCoordSet(shape.corners())
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((c[0], c[-1]) for c in shape)
    )
    assert actual == expected
    if shape.shape == mesh.PrismTypes.TRIANGULAR:
        assert len(actual) == 3
    else:
        assert shape.shape == mesh.PrismTypes.RECTANGULAR
        assert len(actual) == 4


@given(flat_sided_hex)
def test_prism_far_edge(h: mesh.Prism) -> None:
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable(
            s.nodes.coords.get[..., -1].iter_points() for s in h
        )
    )
    actual = coordinates.FrozenCoordSet(h.far.corners())
    assert actual <= expected
    x3 = np.array([a.x3 for a in actual])
    np.testing.assert_allclose(x3, x3[0], 1e-8, 1e-8)


@given(from_type(mesh.Prism))
def test_prism_near_far_corners(h: mesh.Prism) -> None:
    expected = coordinates.FrozenCoordSet(h.corners())
    actual = coordinates.FrozenCoordSet(h.near.corners()) | coordinates.FrozenCoordSet(
        h.far.corners()
    )
    assert expected == actual


@given(flat_sided_hex)
def test_hex_corners(h: mesh.Prism) -> None:
    corners = coordinates.FrozenCoordSet(h.corners())
    assert corners <= coordinates.FrozenCoordSet(mesh.control_points(h).iter_points())
    assert len(corners) == 8


@given(from_type(mesh.Prism))
def test_prism_iter(h: mesh.Prism) -> None:
    sides = list(h)
    if h.shape == mesh.PrismTypes.TRIANGULAR:
        assert len(sides) == 3
    else:
        assert h.shape == mesh.PrismTypes.RECTANGULAR
        assert len(sides) == 4
    cp = coordinates.FrozenCoordSet(mesh.control_points(h).iter_points())
    for side in sides:
        assert coordinates.FrozenCoordSet(mesh.control_points(side).iter_points()) < cp


@given(from_type(mesh.Prism), whole_numbers)
def test_prism_offset(h: mesh.Prism, x: float) -> None:
    actual = coordinates.FrozenCoordSet(Offset(h, x).corners())
    expected = coordinates.FrozenCoordSet(c.offset(x) for c in h.corners())
    assert actual == expected


@given(divisions.flatmap(subdivideable_hex), divisions)
def test_prism_subdivision_len(h: mesh.Prism, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = h.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(divisions.flatmap(subdivideable_hex), divisions)
def test_prism_subdivision(h: mesh.Prism, divisions: int) -> None:
    divisions_iter = h.subdivide(divisions)
    prism_corners = list(h.corners())
    first = next(divisions_iter)
    corners = list(first.corners())
    n = 3 if h.shape == mesh.PrismTypes.TRIANGULAR else 4
    for i in range(n):
        corners[i].approx_eq(prism_corners[i])
    prev = corners
    for h in divisions_iter:
        corners = list(h.corners())
        for i in range(n):
            corners[i].approx_eq(prev[i + n])
        prev = corners
    for i in range(n, 2 * n):
        corners[i].approx_eq(prism_corners[i])


@given(from_type(mesh.Prism))
def test_prism_make_flat(p: mesh.Prism) -> None:
    flat = p.make_flat_faces()
    corners = coordinates.FrozenCoordSet(p.corners())
    flat_corners = coordinates.FrozenCoordSet(flat.corners())
    assert corners == flat_corners


@settings(deadline=None)
@given(from_type(mesh.Prism), integers(1, 10))
def test_prism_make_flat_idempotent(p: mesh.Prism, n: int) -> None:
    new_prism = flat_prism = p.make_flat_faces()
    expected = coordinates.FrozenCoordSet(mesh.control_points(flat_prism).iter_points())
    for _ in range(n):
        new_prism = new_prism.make_flat_faces()
        assert (
            coordinates.FrozenCoordSet(mesh.control_points(new_prism).iter_points())
            == expected
        )


@given(mesh_arguments)
def test_mesh_layer_elements_no_offset(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
) -> None:
    layer = mesh.MeshLayer(*args)
    for actual, expected in zip(layer, args[0]):
        assert actual is expected
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        assert actual_bound == expected_bound


def get_corners(
    shape: mesh.Prism
    | mesh.UnalignedShape
    | mesh.Quad
    | mesh.Curve
    | mesh.FieldAlignedCurve,
) -> Iterator[coordinates.Coord]:
    if isinstance(shape, (mesh.Prism, mesh.Quad, mesh.UnalignedShape)):
        yield from shape.corners()
    elif isinstance(shape, coordinates.Coords):
        yield shape[0]
        yield shape[-1]
    else:
        yield shape.coords[0]
        yield shape.coords[-1]


@settings(deadline=None)
@given(mesh_arguments, non_nans())
def test_mesh_layer_elements_with_offset(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]], offset: float
) -> None:
    layer = Offset(mesh.MeshLayer(*args), offset)
    for actual, expected in zip(layer, args[0]):
        actual_corners = coordinates.FrozenCoordSet(actual.corners())
        expected_corners = coordinates.FrozenCoordSet(
            Offset(expected, offset).corners()
        )
        assert actual_corners == expected_corners
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        actual_elems = frozenset(
            coordinates.FrozenCoordSet(get_corners(elem)) for elem in actual_bound
        )
        expected_elems = frozenset(
            coordinates.FrozenCoordSet(get_corners(Offset(elem, offset)))
            for elem in expected_bound
        )
        assert actual_elems == expected_elems


@settings(deadline=None)
@given(integers(1, 5).flatmap(subdivideable_mesh_arguments), divisions)
def test_mesh_layer_elements_with_subdivisions(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]], subdivisions: int
) -> None:
    layer = mesh.MeshLayer(*args, subdivisions=subdivisions)
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable(
            (
                x.corners()
                for x in itertools.chain.from_iterable(
                    (x.subdivide(subdivisions) for x in args[0])
                )
            )
        )
    )
    actual = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((x.corners() for x in layer))
    )
    assert expected == actual
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        expected_corners = coordinates.FrozenCoordSet(
            itertools.chain.from_iterable(
                (
                    get_corners(x)
                    for x in itertools.chain.from_iterable(
                        (
                            cast(Iterable[mesh.B], x.subdivide(subdivisions))
                            for x in expected_bound
                        )
                    )
                )
            )
        )
        actual_corners = coordinates.FrozenCoordSet(
            itertools.chain.from_iterable((get_corners(x) for x in actual_bound))
        )
        assert expected_corners == actual_corners


def evaluate_element(
    element: mesh.Quad | mesh.Prism, i: int
) -> Iterator[coordinates.Coord]:
    coords = element.nodes.coords.get[..., i]
    if isinstance(element, mesh.Quad):
        yield coords[0]
        yield coords[-1]
    else:
        yield coords[0, 0]
        yield coords[0, -1]
        yield coords[-1, 0]
        if element.shape == mesh.PrismTypes.RECTANGULAR:
            yield coords[-1, -1]


@settings(deadline=None)
@given(integers(1, 5).flatmap(subdivideable_mesh_arguments), whole_numbers, divisions)
def test_mesh_layer_near_faces(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
    offset: float,
    subdivisions: int,
) -> None:
    layer = Offset(mesh.MeshLayer(*args, subdivisions), offset)
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable(
            (evaluate_element(Offset(x, offset), 0) for x in args[0])
        ),
    )
    actual = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((get_corners(x) for x in layer.near_faces())),
    )
    assert expected == actual


@settings(deadline=None)
@given(integers(1, 5).flatmap(subdivideable_mesh_arguments), whole_numbers, integers(1, 5))
def test_mesh_layer_far_faces(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
    offset: float,
    subdivisions: int,
) -> None:
    layer = Offset(mesh.MeshLayer(*args, subdivisions), offset)
    expected = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable(
            (evaluate_element(Offset(x, offset), -1) for x in args[0])
        ),
    )
    actual = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((get_corners(x) for x in layer.far_faces())),
    )
    assert expected == actual


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_faces_in_elements(layer: mesh.MeshLayer) -> None:
    element_corners = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((get_corners(x) for x in layer)),
    )
    near_face_corners = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((get_corners(x) for x in layer.near_faces())),
    )
    far_face_corners = coordinates.FrozenCoordSet(
        itertools.chain.from_iterable((get_corners(x) for x in layer.far_faces())),
    )
    assert near_face_corners < element_corners
    assert far_face_corners < element_corners


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_len(layer: mesh.MeshLayer) -> None:
    layer_iter = iter(layer)
    for _ in range(len(layer)):
        next(layer_iter)
    with pytest.raises(StopIteration):
        next(layer_iter)


quad_mesh_elements = _quad_mesh_elements(
    5,
    1.0,
    1.0,
    10.0,
    ((0.0, 0.0), (1.0, 0.0)),
    4,
    coordinates.CoordinateSystem.CARTESIAN,
    False,
    False,
)
hex_mesh_elements = cast(
    tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]],
    _hex_mesh_arguments(
        3,
        1.0,
        1.0,
        10.0,
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        3,
        3,
        coordinates.CoordinateSystem.CARTESIAN,
        False,
    ),
)[0]


@pytest.mark.parametrize(
    "elements",
    [(quad_mesh_elements,), (hex_mesh_elements,)],
)
def test_mesh_layer_element_type(elements: list[mesh.Quad] | list[mesh.Prism]) -> None:
    layer = mesh.MeshLayer(elements, [])  # type: ignore
    element = next(iter(elements))
    assert layer.element_type is type(element)


@given(quad_mesh_layer_no_divisions)
def test_mesh_layer_quads_for_quads(
    layer: mesh.MeshLayer[mesh.Quad, mesh.FieldAlignedCurve, mesh.Curve],
) -> None:
    assert all(q1 is q2 for q1, q2 in zip(layer, layer.quads()))


@given(prism_mesh_layer_no_divisions)
def test_mesh_layer_quads_for_prisms(
    layer: mesh.MeshLayer[mesh.Prism, mesh.Quad, mesh.UnalignedShape],
) -> None:
    all_quads = list(itertools.chain.from_iterable(p for p in layer))
    returned_quads = list(layer.quads())
    assert len(all_quads) == len(returned_quads)
    assert frozenset(
        coordinates.FrozenCoordSet(q.corners()) for q in all_quads
    ) == frozenset(coordinates.FrozenCoordSet(q.corners()) for q in returned_quads)


@given(from_type(mesh.GenericMesh))
def test_mesh_iter_layers(m: mesh.GenericMesh) -> None:
    for layer, offset in zip(m.layers(), m.offsets):
        assert (
            layer.get_underlying_object().reference_elements
            is m.reference_layer.reference_elements
        )
        assert layer.x3_offset == offset
        assert layer.subdivisions == m.reference_layer.subdivisions


@given(from_type(mesh.GenericMesh))
def test_mesh_len(m: mesh.GenericMesh) -> None:
    mesh_iter = iter(m)
    for _ in range(len(m)):
        next(mesh_iter)
    with pytest.raises(StopIteration):
        next(mesh_iter)


shared_orders = shared(integers(1, 10), key=123456)
coords = builds(
    coordinates.Coord,
    whole_numbers,
    whole_numbers,
    whole_numbers,
    just(coordinates.CoordinateSystem.CARTESIAN),
)
lines_across_field = builds(
    mesh.straight_line_across_field,
    common_slice_coords,
    common_slice_coords,
    shared_orders,
)
geometries = one_of(
    (
        lines_across_field,
        builds(mesh.straight_line, common_coords, common_coords, shared_orders),
        builds(
            mesh.field_aligned_positions,
            lines_across_field,
            floats(1e-3, 1e3),
            linear_traces,
            floats(0.0, 1.0).map(np.array),
            shared_orders,
            pos_divisions.flatmap(lambda x: integers(0, x - 1)),
            pos_divisions,
        ).map(mesh.Quad),
        builds(
            mesh.Prism,
            sampled_from(tuple(mesh.PrismTypes)[::-1]),
            builds(
                mesh.field_aligned_positions,
                builds(
                    corners_to_poloidal_quad,
                    shared_orders,
                    lists(common_slice_coords.map(tuple), min_size=4, max_size=4).map(
                        tuple
                    ),
                    shared_coordinate_systems,
                ),
                floats(1e-3, 1e3),
                linear_traces,
                floats(0.0, 1.0).map(np.array),
                shared_orders,
                pos_divisions.flatmap(lambda x: integers(0, x - 1)),
                pos_divisions,
            ),
        ),
    )
)


@given(shared_orders, geometries)
def test_order(
    n: int, geom: mesh.AcrossFieldCurve | mesh.Curve | mesh.Quad | mesh.Prism
) -> None:
    assert n == mesh.order(geom)
