"""Functions for building meshes from the command line.

"""

import click
import numpy as np

from neso_fame.fields import straight_field
from neso_fame.generators import field_aligned_2d
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
    help="The number of elements in the x1 direction.",
    default=2,
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
        "The number of non-conformal layers that exist in the x2-direction. "
        "Use 0 to indicate it should be the same as the x2-resolution."
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
    help="Use periodic x2 boundaries",
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
    fail). The mesh will be written MESHFILE in the Nektar++
    uncompressed XML format.

    """
    layers = _validate_layers(layers, nx2)
    # In order to keep a right-handed co-ordinate system, the
    # conversion from Cartesian2d to standard 3D cartesian rotates the
    # axes and makes x1 negative.
    m = field_aligned_2d(
        SliceCoords(
            np.linspace(-x2_extent[1], -x2_extent[0], nx1 + 1),
            np.zeros(nx1 + 1),
            CoordinateSystem.CARTESIAN2D,
        ),
        straight_field(-angle * np.pi / 180.0),
        x1_extent,
        layers,
        2,
        subdivisions=nx2 // layers,
    )
    write_nektar(m, 1, meshfile, 2, layers > 1 or periodic, periodic)


@simple.command("3d")
def simple_3d() -> None:
    """Generate a simple 3D Cartesian mesh aligned to straight
    field lines. These field lines can be at an angle to the x3
    direction (although 90 degrees is singular and will cause the
    script to fail).

    """


if __name__ == "__main__":
    simple()
