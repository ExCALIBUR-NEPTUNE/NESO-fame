"""Functions for building meshes from the command line.

"""

import click
import numpy as np

from neso_fame.fields import straight_field
from neso_fame.generators import field_aligned_2d, field_aligned_3d
from neso_fame.mesh import CoordinateSystem, SliceCoords
from neso_fame.nektar_writer import write_nektar


@click.group()
def simple() -> None:
    """Create simple Cartesian meshes in either 2D or 3D."""


POSITIVE = click.IntRange(1)
NONNEGATIVE = click.IntRange(0)


def _validate_layers(layers: int, nx: int) -> int:
    """Check that the specified direction can be evenly divided into the
    specified number of layers.
    """
    if layers == 0:
        return nx
    if nx % layers == 0:
        return layers
    raise click.BadParameter(
        f"Can not divide {nx} elements evenly into {layers} layers"
    )


@simple.command("2d")
@click.option(
    "--nx1",
    "--x1-resolution",
    type=POSITIVE,
    help="The number of elements in the x1 direction.",
    default=2,
    show_default=True,
)
@click.option(
    "--x1-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x2 direction.",
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--nx2",
    "--x2-resolution",
    type=POSITIVE,
    help="The number of elements in the x2 direction.",
    default=10,
    show_default=True,
)
@click.option(
    "--x2-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x2 direction.",
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--layers",
    type=NONNEGATIVE,
    help=(
        "The number of non-conformal layers that exist in the x1-direction. "
        "Use 0 to indicate it should be the same as the x1-resolution."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--angle",
    type=float,
    help=(
        "The angle the magnentic field is skewed away from the x1-axis "
        "towards the x2-axis, in degrees."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--periodic",
    is_flag=True,
    default=False,
    help="Use periodic x1 boundaries",
    show_default=True,
)
@click.argument("meshfile", type=click.Path(dir_okay=False, writable=True))
def simple_2d(
    nx1: int,
    x1_extent: tuple[float, float],
    nx2: int,
    x2_extent: tuple[float, float],
    layers: int,
    angle: float,
    periodic: bool,
    meshfile: str,
) -> None:
    """Generate a simple 2D Cartesian mesh aligned to straight field
    lines. These field lines can be at an angle to the x1 direction
    (although 90 degrees is singular and will cause the script to
    fail). The mesh will be written to MESHFILE in the Nektar++
    uncompressed XML format.

    """
    layers = _validate_layers(layers, nx1)
    # In order to keep a right-handed co-ordinate system, the
    # conversion from Cartesian2d to standard 3D cartesian rotates the
    # axes and makes x1 negative.
    m = field_aligned_2d(
        SliceCoords(
            np.linspace(-x2_extent[1], -x2_extent[0], nx2 + 1),
            np.zeros(nx2 + 1),
            CoordinateSystem.CARTESIAN2D,
        ),
        straight_field(-angle * np.pi / 180.0),
        x1_extent,
        layers,
        2,
        subdivisions=nx1 // layers,
    )
    write_nektar(m, 1, meshfile, 2, layers > 1 or periodic, periodic)


@simple.command("3d")
@click.option(
    "--nx1",
    "--x1-resolution",
    type=POSITIVE,
    help="The number of elements in the x1 direction.",
    default=10,
    show_default=True,
)
@click.option(
    "--x1-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x1 direction.",
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--nx2",
    "--x2-resolution",
    type=POSITIVE,
    help="The number of elements in the x2 direction.",
    default=10,
    show_default=True,
)
@click.option(
    "--x2-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x2 direction.",
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--nx3",
    "--x3-resolution",
    type=POSITIVE,
    help="The number of elements in the x3 direction.",
    default=2,
    show_default=True,
)
@click.option(
    "--x3-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x4 direction.",
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--layers",
    type=NONNEGATIVE,
    help=(
        "The number of non-conformal layers that exist in the x3-direction. "
        "Use 0 to indicate it should be the same as the x3-resolution."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--angle1",
    type=float,
    help=(
        "The angle the magnentic field is skewed away from the x3-axis "
        "towards the x1-axis, in degrees."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--angle2",
    type=float,
    help=(
        "The angle the magnentic field is skewed away from the x3-axis "
        "towards the x2-axis, in degrees."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--periodic",
    is_flag=True,
    default=False,
    help="Use periodic x3 boundaries",
    show_default=True,
)
@click.argument("meshfile", type=click.Path(dir_okay=False, writable=True))
def simple_3d(
    nx1: int,
    x1_extent: tuple[float, float],
    nx2: int,
    x2_extent: tuple[float, float],
    nx3: int,
    x3_extent: tuple[float, float],
    layers: int,
    angle1: float,
    angle2: float,
    periodic: bool,
    meshfile: str,
) -> None:
    """Generate a simple 3D Cartesian mesh aligned to straight field
    lines. These field lines can be at an angle relative to the x3
    direction (although 90 degrees is singular and will cause the
    script to fail). The mesh will be written to MESHFILE in the
    Nektar++ uncompressed XML format.

    """
    layers = _validate_layers(layers, nx3)
    x1, x2 = np.meshgrid(
        np.linspace(x1_extent[0], x1_extent[1], nx1 + 1),
        np.linspace(x2_extent[0], x2_extent[1], nx2 + 1),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x1,
        x2,
        CoordinateSystem.CARTESIAN,
    )
    elements = [
        ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
        for i in range(nx2)
        for j in range(nx1)
    ]
    field = straight_field(angle1 * np.pi / 180.0, angle2 * np.pi / 180.0)

    m = field_aligned_3d(
        starts, field, elements, x3_extent, layers, 2, nx3 // layers
    )
    write_nektar(m, 1, meshfile, 3, layers > 1 or periodic, periodic)


if __name__ == "__main__":
    simple()