"""Functions for building meshes from the command line."""

from io import StringIO, TextIOBase
from sys import argv
from typing import Optional

import click
import numpy as np
import yaml
from hypnotoad import Mesh as HypnoMesh  # type: ignore

from neso_fame.fields import straight_field
from neso_fame.generators import field_aligned_2d, field_aligned_3d, hypnotoad_mesh
from neso_fame.hypnotoad_interface import eqdsk_equilibrium
from neso_fame.mesh import CoordinateSystem, SliceCoords
from neso_fame.nektar_writer import (
    nektar_3d_element,
    write_nektar,
    write_poloidal_mesh,
)


@click.group()
def simple() -> None:
    """Create simple Cartesian meshes in either 2D or 3D."""


POSITIVE = click.IntRange(1)
NONNEGATIVE = click.IntRange(0)


def _validate_layers(layers: int, nx: int) -> int:
    """Get the number of layers to divide the mesh into.

    If the number of elements in the x3-direction can not be evenly
    divided into this many layers, an exception will be raised.
    """
    if layers == 0:
        return nx
    if nx % layers == 0:
        return layers
    raise click.BadParameter(
        f"Can not divide {nx} elements evenly into {layers} layers"
    )


def _mesh_provenance() -> str:
    return f"<!-- This mesh was generated using NESO-fame:\n    {' '.join(argv)}\n-->"


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
@click.option(
    "--compress",
    is_flag=True,
    default=False,
    help="Use the compressed XML format for the mesh",
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
    compress: bool,
    meshfile: str,
) -> None:
    """Generate a simple 2D Cartesian mesh aligned to straight field lines.

    These field lines can be at an angle to the x1 direction
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
            np.linspace(x2_extent[0], x2_extent[1], nx2 + 1),
            np.zeros(nx2 + 1),
            CoordinateSystem.CARTESIAN2D,
        ),
        straight_field(-angle * np.pi / 180.0),
        (-x1_extent[1], -x1_extent[0]),
        layers,
        2,
        subdivisions=nx1 // layers,
    )
    write_nektar(m, 1, meshfile, 2, layers > 1 or periodic, periodic, compress)
    with open(meshfile, "a") as f:
        f.write(_mesh_provenance())


@simple.command("3d")
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
    default=10,
    show_default=True,
)
@click.option(
    "--x3-extent",
    nargs=2,
    type=float,
    help="The limits of the domain in the x3 direction.",
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
    "--angle1",
    type=float,
    help=(
        "The angle the magnentic field is skewed away from the x1-axis "
        "towards the x2-axis, in degrees."
    ),
    default=0,
    show_default=True,
)
@click.option(
    "--angle2",
    type=float,
    help=(
        "The angle the magnentic field is skewed away from the x1-axis "
        "towards the x3-axis, in degrees."
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
@click.option(
    "--compress",
    is_flag=True,
    default=False,
    help="Use the compressed XML format for the mesh",
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
    compress: bool,
    meshfile: str,
) -> None:
    """Generate a simple 3D Cartesian mesh aligned to straight field lines.

    These field lines can be at an angle relative to the x1
    direction (although 90 degrees is singular and will cause the
    script to fail). The mesh will be written to MESHFILE in the
    Nektar++ uncompressed XML format.

    """
    # Coordinates will be rotated so that the magnetic field aligns
    # with the x1-axis, rather than the x3-axis
    layers = _validate_layers(layers, nx1)
    x3, x1 = np.meshgrid(
        np.linspace(x2_extent[0], x2_extent[1], nx2 + 1),
        np.linspace(x3_extent[0], x3_extent[1], nx3 + 1),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x3,
        x1,
        CoordinateSystem.CARTESIAN_ROTATED,
    )
    elements = [
        ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
        for i in range(nx3)
        for j in range(nx2)
    ]
    field = straight_field(-angle1 * np.pi / 180.0, -angle2 * np.pi / 180.0)

    m = field_aligned_3d(
        starts,
        field,
        elements,
        (-x1_extent[1], -x1_extent[0]),
        layers,
        2,
        nx1 // layers,
    )
    write_nektar(m, 1, meshfile, 3, layers > 1 or periodic, periodic, compress)
    with open(meshfile, "a") as f:
        f.write(_mesh_provenance())


@click.command("hypnotoad")
@click.option(
    "--n",
    "--toroidal-resolution",
    type=POSITIVE,
    help="The number of elements in the toroidal direction.",
    default=8,
    show_default=True,
)
@click.option(
    "--toroidal-limits",
    nargs=2,
    type=float,
    help="The limits of the domain in the x3 direction.",
    default=(0.0, 2 * np.pi),
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
    "--max-ratio",
    type=click.FloatRange(0.0, min_open=True),
    help=(
        "The maximum ratio to allow between the length of the perpendicular"
        "and field-aligned edges of an element. If an element exceeds this"
        "ratio, it will be merged with an adjacent one. Note that"
        "only elements radiating away from an X-point are checked and"
        "some may be left in order to maintain a conformal mesh."
    ),
    default=10.0,
)
@click.option(
    "--core/--no-core",
    is_flag=True,
    default=True,
    help="Whether to fill in the core region of the tokamak with prisms.",
)
@click.option(
    "--mesh-to-wall/--no-mesh-to-wall",
    is_flag=True,
    default=True,
    help="Whether to fill the region between the edge of the hypnotoad "
    "mesh and the tokamak wall.",
)
@click.option(
    "--wall-resolution",
    type=click.FloatRange(0.0, min_open=True),
    default=None,
    help="The target size of elements on the tokamak compared to those in the "
    "outermost layer of hte hypnotoad mesh. If not specified, use the wall "
    "elements specified in the GEQDSK file (which may vary significantly in size).",
)
@click.option(
    "--wall-angle-threshold",
    type=click.FloatRange(0.0, np.pi),
    default=np.pi / 8,
    help="If adjusting the resolution of the tokamak wall, any vertices with an "
    "angle above this threshold will be preserved as sharp corners. Angles below "
    "it will be smoothed out.",
)
@click.option(
    "--min-wall-distance",
    type=click.FloatRange(0.0),
    default=0.025,
    help="The minimum distance to leave between hypnotoad mesh elements and the tokamak wall.",
)
@click.option(
    "--alignment-steps",
    type=NONNEGATIVE,
    default=1,
    help="How quickly to transition between a field-aligned and unaligned mesh "
    "near the tokamak wall.",
)
@click.option(
    "--order",
    type=POSITIVE,
    help="The order of accuracy to use to describe curved elements.",
    default=3,
    show_default=True,
)
@click.option(
    "--compress",
    is_flag=True,
    default=False,
    help="Use the compressed XML format for the mesh",
)
@click.option(
    "--config",
    type=click.File(),
    default=StringIO(""),
    help="YAML file with settings used to generate the 2D hypnotoad mesh.",
)
@click.option(
    "--full/--poloidal",
    is_flag=True,
    default=True,
    help="Whether to output the full 3D mesh or just the poloidal cross-section.",
)
@click.argument("geqdsk", type=click.Path(exists=True, dir_okay=False))
@click.argument("meshfile", type=click.Path(dir_okay=False, writable=True))
def hypnotoad(
    n: int,
    toroidal_limits: tuple[float, float],
    layers: int,
    max_ratio: float,
    core: bool,
    mesh_to_wall: bool,
    wall_resolution: Optional[float],
    wall_angle_threshold: float,
    min_wall_distance: float,
    alignment_steps: int,
    order: int,
    compress: bool,
    full: bool,
    geqdsk: str,
    config: TextIOBase,
    meshfile: str,
) -> None:
    """Generate a 3D mesh from a GEQDSK file.

    This is done by first generating a 2D mesh using hypnotoad and
    then following each node in that mesh along the magnetic field
    lines. The mesh will be written to MESHFILE in the Nektar++
    uncompressed XML format. Note that only orthogonal meshes are
    allowed.

    """
    layers = _validate_layers(layers, n)
    options = yaml.safe_load(config)
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise RuntimeError("Hypnotoad YAML file must contain a dictionary.")
    print(f"Reading G-EQDSK from {geqdsk} and constructing equilibrium...")
    eq = eqdsk_equilibrium(geqdsk, options)
    print("Building 2D poloidal mesh...")
    hypno_mesh = HypnoMesh(eq, options)
    print("Extruding 2D mesh along magnetic field lines...")
    mesh = hypnotoad_mesh(
        hypno_mesh,
        toroidal_limits,
        layers,
        21,
        n // layers,
        max_ratio,
        core,
        mesh_to_wall,
        mesh_to_wall,
        min_wall_distance,
        wall_resolution,
        wall_angle_threshold,
        alignment_steps,
        lambda x: next(iter(nektar_3d_element(x, order, 3, -1)[0])).IsValid(),
    )
    periodic = toroidal_limits[0] % (2 * np.pi) == toroidal_limits[1] % (2 * np.pi)
    print("Converting mesh to Nektar++ format and writing to disk...")
    if full:
        # FIXME: This seems to be really really slow when I mesh all the way
        # to the wall. It needs profiling, as I suspect it is
        # repeating quite a lot of work or something.
        write_nektar(mesh, order, meshfile, 3, True, periodic, compress)
    else:
        write_poloidal_mesh(mesh, order, meshfile, compress)
    with open(meshfile, "a") as f:
        f.write(_mesh_provenance())
