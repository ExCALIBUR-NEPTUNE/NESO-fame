from __future__ import annotations

import itertools
import operator
import os
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import reduce
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Type, TypeGuard, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    builds,
    floats,
    from_type,
    integers,
    frozensets,
    text,
    just,
    lists,
    one_of,
    sampled_from,
    shared,
)
from NekPy import LibUtilities as LU
from NekPy import SpatialDomains as SD
from pytest import approx, mark

from neso_fame import meshio_writer
from neso_fame.coordinates import (
    Coord,
    CoordinateSystem,
    Coords,
    FrozenCoordSet,
    SliceCoord,
)
from neso_fame.mesh import (
    B,
    C,
    E,
    EndShape,
    FieldAlignedCurve,
    FieldTracer,
    GenericMesh,
    Mesh,
    MeshLayer,
    NormalisedCurve,
    Prism,
    PrismMesh,
    PrismMeshLayer,
    Quad,
    QuadMesh,
    QuadMeshLayer,
    Segment,
    StraightLineAcrossField,
    control_points,
)
from neso_fame.offset import Offset

from .conftest import (
    flat_sided_hex,
    flat_sided_prism,
    linear_field_trace,
    non_nans,
    prism_meshes,
    quad_meshes,
    simple_trace,
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


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(one_of((from_type(SliceCoord), from_type(Coord))), element_types, cellsets)
def test_nektar_point(coord: Coord, element: str, csets: frozenset[str]) -> None:
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
def test_nektar_point_caching(
    coords: list[Coord], element: list[str], csets: list[frozenset[str]]
) -> None:
    mesh_data = meshio_writer.MeshioData()
    c1 = mesh_data.point(coords[0], element[0], csets[0])
    assert c1 == mesh_data.point(coords[0], element[0], csets[0])
    assert c1 != mesh_data.point(coords[1], element[0], csets[0])
    assert c1 != mesh_data.point(coords[0], element[0], csets[1])
    assert c1 != mesh_data.point(coords[0], element[1], csets[0])
    assert mesh_data.meshio().points.shape[0] == 4
