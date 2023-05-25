import itertools
import operator
import numpy as np

from neso_fame.edge_filter import NodeStatus, classify_node_position
from neso_fame.mesh import CoordinateSystem, NormalisedFieldLine, Coords


def test_2d_edges() -> None:
    a1 = 0.2
    a3 = 1.0
    n = 10

    def make_field_line(x1_start: float) -> NormalisedFieldLine:
        return lambda s: Coords(
            a1 * np.asarray(s) + x1_start - 0.5 * a1,
            np.array(0.0),
            a3 * np.asarray(s) - 0.5 * a3,
            CoordinateSystem.Cartesian,
        )

    line_centres = np.linspace(0.0, 1.0, n + 1)
    lines = [make_field_line(x) for x in line_centres]
    bounds = (0.01, 0.99)
    connectivity = [
        [i + 1] if i == 0 else [i - 1] if i == n else [i - 1, i + 1]
        for i in range(n + 1)
    ]
    skin_nodes = [True] + [False] * (n - 1) + [True]

    near_end = line_centres - 0.5 * a1
    far_end = line_centres + 0.5 * a1

    actual_status, actual_skin = classify_node_position(
        lines, bounds, connectivity, skin_nodes
    )
    expected = np.logical_and(
        np.logical_and(
            np.logical_and(near_end >= bounds[0], near_end <= bounds[1]),
            far_end >= bounds[0],
        ),
        far_end <= bounds[1],
    )

    for status, in_domain in zip(actual_status, expected):
        if in_domain:
            assert status == NodeStatus.UNKNOWN or status == NodeStatus.INTERNAL
        else:
            assert status == NodeStatus.EXTERNAL

    skin_indices = [
        i if l else i + 1
        for i, (l, r) in enumerate(itertools.pairwise(expected))
        if l != r
    ]
    assert all(
        is_skin if i in skin_indices else not is_skin
        for i, is_skin in enumerate(actual_skin)
    )


def test_2d_curved_edges() -> None:
    n = 10
    def make_field_line(x1_start: float) -> NormalisedFieldLine:
        """Create filed lines corresponding to semi-circles."""
        return lambda s: Coords(
            0.5 * np.sin(np.asarray(s) * np.pi) + x1_start,
            np.array(0.0),
            -0.5 * np.cos(np.asarray(s) * np.pi),
            CoordinateSystem.Cartesian,
        )

    line_centres = np.linspace(0.0, 3.0, n + 1)
    lines = [make_field_line(x) for x in line_centres]
    bounds = (0.01, 2.99)
    connectivity = [
        [i + 1] if i == 0 else [i - 1] if i == n else [i - 1, i + 1]
        for i in range(n + 1)
    ]
    skin_nodes = [True] + [False] * (n - 1) + [True]

    line_min = line_centres
    line_max = line_centres + 0.5

    actual_status, actual_skin = classify_node_position(
        lines, bounds, connectivity, skin_nodes
    )
    expected = np.logical_and(
        np.logical_and(
            np.logical_and(line_min >= bounds[0], line_min <= bounds[1]),
            line_max >= bounds[0],
        ),
        line_max <= bounds[1],
    )

    for status, in_domain in zip(actual_status, expected):
        if in_domain:
            assert status == NodeStatus.UNKNOWN or status == NodeStatus.INTERNAL
        else:
            assert status == NodeStatus.EXTERNAL

    skin_indices = [
        i if l else i + 1
        for i, (l, r) in enumerate(itertools.pairwise(expected))
        if l != r
    ]
    assert all(
        is_skin if i in skin_indices else not is_skin
        for i, is_skin in enumerate(actual_skin)
    )
