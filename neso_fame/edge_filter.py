from __future__ import annotations

from collections.abc import Sequence, Iterable
from enum import Enum
import itertools
from typing import Optional

from .mesh import NormalisedFieldLine


class NodeStatus(Enum):
    UNKNOWN = 1
    EXTERNAL = 2
    INTERNAL = 3


Connectivity = Sequence[Sequence[int]]


def is_line_in_domain(field_line: NormalisedFieldLine, bounds: Sequence[float]) -> bool:
    # FIXME: This is just a dumb stub. It doesn't even work properly for 2-D, let alone 3-D.
    location = field_line(0.0)
    x1 = float(location.x1)
    return x1 >= bounds[0] and x1 <= bounds[1]


def is_skin_node(
    node_status: NodeStatus, connections: Iterable[int], statuses: Sequence[NodeStatus]
) -> bool:
    return node_status != NodeStatus.EXTERNAL and any(
        statuses[i] == NodeStatus.EXTERNAL for i in connections
    )


def classify_node_position(
    lines: Sequence[NormalisedFieldLine],
    bounds,
    connectivity: Connectivity,
    skin_nodes: Sequence[bool],
    status: Optional[Sequence[NodeStatus]] = None,
) -> tuple[Sequence[NodeStatus], Sequence[bool]]:
    def check_is_in_domain(
        line: NormalisedFieldLine, is_skin: bool, status: NodeStatus
    ) -> tuple[NodeStatus, bool]:
        if is_skin and status == NodeStatus.UNKNOWN:
            if is_line_in_domain(line, bounds):
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
            lambda x: is_skin_node(x[0], x[1], updated_status),
            zip(updated_status, connectivity),
        )
    )
    if any(newly_external):
        return classify_node_position(
            lines, bounds, connectivity, updated_skin, updated_status
        )
    else:
        return updated_status, updated_skin
