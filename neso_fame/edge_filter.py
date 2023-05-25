from __future__ import annotations

from collections.abc import Sequence, Iterable
from enum import Enum
import itertools
from typing import Optional

import numpy as np
from scipy.interpolate import lagrange
from scipy.optimize import minimize_scalar

from .mesh import NormalisedFieldLine, Curve, Coords, CoordinateSystem, CartesianCoordinates


class NodeStatus(Enum):
    UNKNOWN = 1
    EXTERNAL = 2
    INTERNAL = 3


Connectivity = Sequence[Sequence[int]]

TOLERANCE = 1e-8


def make_lagrange_interpolation(
    norm_line: Curve, order=1
) -> Curve[CartesianCoordinates]:
    s = np.linspace(0.0, 1.0, order + 1)
    coords = norm_line.control_points(order).to_cartesian()
    interpolators = [lagrange(s, coord) for coord in coords]
    return Curve(
        lambda s: Coords(
            interpolators[0](s),
            interpolators[1](s),
            interpolators[2](s),
            CoordinateSystem.Cartesian,
        )
    )


def _is_line_in_domain(field_line: NormalisedFieldLine, bounds: Sequence[float], order: int) -> bool:
    # FIXME: This onlye works for 2D
    x_start = field_line(0.).x1
    x_end = field_line(1.).x1
    # Check if either end of the line falls outside the domain
    if not (bounds[0] < x_start < bounds[1]) or not (bounds[0] <  x_end < bounds[1]):
        return False
    # Maybe I should make the Lagrange version here? If any nodes oute of domain then return. Otherwise, check at mid-point between each node. Mid-point not guaranteed to be closest approach, though, especially in 3D.
    upper_bound_sol = minimize_scalar(lambda s: (field_line(s).x1 - bounds[0]) ** 2, bracket=(0., 1.), method="Brent")
    lower_bound_sol = minimize_scalar(lambda s: (field_line(s).x1 - bounds[1]) ** 2, bracket=(0., 1.), method="Brent")
    if not upper_bound_sol.success or not lower_bound_sol.success:
        raise RuntimeError("Failed to converge on closest point of approach to boundaries!")
    # Check if the minimum occurs within the search area
    if 0. < upper_bound_sol.x < 1. or 0. < lower_bound_sol.x < 1.:
        # Check whether the minimum corresponds to intersecting the boundary
        return upper_bound_sol.fun > TOLERANCE and lower_bound_sol.fun > TOLERANCE
    else:
        return True


def _is_skin_node(
    node_status: NodeStatus, connections: Iterable[int], statuses: Sequence[NodeStatus]
) -> bool:
    return node_status != NodeStatus.EXTERNAL and any(
        statuses[i] == NodeStatus.EXTERNAL for i in connections
    )


def classify_node_position(
    lines: Sequence[Curve],
    bounds,
    connectivity: Connectivity,
    order: int,
    skin_nodes: Sequence[bool],
    status: Optional[Sequence[NodeStatus]] = None,
) -> tuple[Sequence[NodeStatus], Sequence[bool]]:
    def check_is_in_domain(
        line: NormalisedFieldLine, is_skin: bool, status: NodeStatus
    ) -> tuple[NodeStatus, bool]:
        if is_skin and status == NodeStatus.UNKNOWN:
            if _is_line_in_domain(line, bounds, order):
                return NodeStatus.INTERNAL, False
            else:
                return NodeStatus.EXTERNAL, True
        else:
            return status, False

    if status is None:
        status = [NodeStatus.UNKNOWN] * len(lines)

    updated_status, newly_external = zip(
        *itertools.starmap(check_is_in_domain, zip(lines, skin_nodes, status))
    )
    updated_skin = list(
        map(
            lambda x: _is_skin_node(x[0], x[1], updated_status),
            zip(updated_status, connectivity),
        )
    )
    if any(newly_external):
        return classify_node_position(
            lines, bounds, connectivity, updated_skin, updated_status
        )
    else:
        return updated_status, updated_skin
