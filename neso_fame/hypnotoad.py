"""Functions for representing magnetic fields using GEQDSK data read
in using hypnotoad.

"""

from pathlib import Path
from typing import Any

from hypnotoad.cases.tokamak import TokamakEquilibrium, read_geqdsk
from hypnotoad.core.equilibrium import PsiContour
from hypnotoad.core.mesh import followPerpendicular
import numpy as np
import numpy.typing as npt
from hypnotoad import Equilibrium, Point2D  # type: ignore
from scipy.integrate import solve_ivp
import yaml

from .mesh import AcrossFieldCurve, CoordinateSystem, FieldTrace, SliceCoord, SliceCoords


def equilibrium_trace(equilibrium: Equilibrium) -> FieldTrace:
    """Return a field trace corresponding to the hypnotoad equilibrium
    object.

    """

    def d_dphi(_: float, y: npt.NDArray) -> npt.NDArray:
        R = y[0]
        Z = y[1]
        R_over_B_t = R / equilibrium.Bzeta(R, Z)
        dR_dphi = equilibrium.Bp_R(R, Z) * R_over_B_t
        dZ_dphi = equilibrium.Bp_Z(R, Z) * R_over_B_t
        return np.asarray(
            [dR_dphi, dZ_dphi, np.sqrt(dR_dphi * dR_dphi + dZ_dphi * dZ_dphi + 1)]
        )

    def trace(start: SliceCoord, phi: npt.ArrayLike) -> tuple[SliceCoords, npt.NDArray]:
        if start.system != CoordinateSystem.CYLINDRICAL:
            raise ValueError("`start` must use a cylindrical coordinate system")
        phi = np.asarray(phi)
        shape = phi.shape
        flat_phi = np.ravel(phi)
        n = len(flat_phi)
        sort_order = np.argsort(flat_phi)
        phi_sorted = flat_phi[sort_order]
        neg = phi_sorted < 0
        pos = phi_sorted > 0
        phi_neg = phi_sorted[neg][::-1]
        phi_pos = phi_sorted[pos]
        R = np.empty(n)
        Z = np.empty(n)
        dist = np.empty(n)
        R_tmp = np.full(n, start.x1)
        Z_tmp = np.full(n, start.x2)
        dist_tmp = np.zeros(n)
        if len(phi_neg) > 0:
            result = solve_ivp(
                d_dphi,
                (0.0, phi_neg[-1]),
                [start.x1, start.x2, 0.0],
                method="DOP853",
                t_eval=phi_neg,
                rtol=1e-10,
                atol=1e-12,
                dense_output=True,
            )
            if not result.success:
                raise RuntimeError("Failed to integrate along field line")
            R_tmp[neg] = result.y[0, ::-1]
            Z_tmp[neg] = result.y[1, ::-1]
            # FIXME: Should I use FineContour.getDistance instead?
            dist_tmp[neg] = result.y[2, ::-1]
        if len(phi_pos) > 0:
            result = solve_ivp(
                d_dphi,
                (0.0, phi_pos[-1]),
                [start.x1, start.x2, 0.0],
                method="DOP853",
                t_eval=phi_pos,
                rtol=1e-10,
                atol=1e-12,
                dense_output=True,
            )
            if not result.success:
                raise RuntimeError("Failed to integrate along field line")
            R_tmp[pos] = result.y[0]
            Z_tmp[pos] = result.y[1]
            dist_tmp[pos] = result.y[2]
        R[sort_order] = R_tmp
        Z[sort_order] = Z_tmp
        dist[sort_order] = dist_tmp
        # FIXME: Should I refine the points using one of the
        # PsiContour methods?
        # https://hypnotoad.readthedocs.io/en/latest/point-refinement.html
        #
        # Probably should, but will require having a psi contour
        # available; think I'll need to do some clever sort of caching
        # to handle that efficiently
        return SliceCoords(
            R.reshape(shape), Z.reshape(shape), CoordinateSystem.CYLINDRICAL
        ), dist.reshape(shape)

    return trace


def eqdsk_equilibrium(eqdsk: str | Path, options: dict[str, Any] = {}) -> TokamakEquilibrium:
    """Reads in GEQDSK file and creates a hypnotoad
    TokamakEquilibrium object.

    Parameters
    ----------
    eqdsk
        The filename or path of the EQDSK file to read in.
    options
        Configurations for hypnotoad, see `Tokamak Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/options.html>`_
        and `Nonorthogonal Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/nonorthogonal-options.html>`_.

    """
    possible_options = (
        [opt for opt in TokamakEquilibrium.user_options_factory.defaults]
        + [
            opt
            for opt in TokamakEquilibrium.nonorthogonal_options_factory.defaults
        ]
    )
    unused_options = [opt for opt in options if opt not in possible_options]
    if unused_options != []:
        raise ValueError(
            f"There are options that are not used: {unused_options}"
        )
    with open(eqdsk, 'r') as f:
        return read_geqdsk(f, options, options)


def perpendicular_edge(eq: Equilibrium, north: SliceCoord, south: SliceCoord) -> AcrossFieldCurve:
    """Returns a line traveling at right angles to the magnetic field
    connecting the two points on the poloidal plane. If this is not
    possible, raise an exception.

    Parameters
    ----------
    eq
        A hypnotaod Equilibrium object describing the magnetic field
    north
        The starting point of the line
    south
        The end point of the line

    """
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")
    psi_start = eq.psi(north.x1, north.x2)
    psi_end = eq.psi(south.x1, south.x2)
    start = Point2D(north.x1, north.x2)

    def func(s: npt.ArrayLike) -> SliceCoords:
        psis = psi_start + (psi_end - psi_start) * np.asarray(s)
        points = followPerpendicular(None, start, psi_start, f_R=eq.f_R, f_Z=eq.f_Z, psivals=psis, rtol=1e-10, atol=1e-12)
        return SliceCoords(np.array(p.R for p in points), np.array(p.Z for p in points), CoordinateSystem.CYLINDRICAL)

    # Check end point agrees with what we expect, within
    # tolerance. Otherwise our edges won't properly join up later.
    if func(1.).to_coord() != south:
        raise RuntimeError(
            "No line perpendicular to the magnetic field connects start and end "
            f"points {north} and {south}"
        )
    return func


def flux_surface_edge(eq: Equilibrium, contour: PsiContour, north: SliceCoord, south: SliceCoord) -> AcrossFieldCurve:
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")
    fine_contour = contour.get_fine_contour()
    start = fine_contour.getDistance(Point2D(north.x1, north.x2))
    end = fine_contour.getDistance(Point2D(south.x1, south.x2))
    interp = fine_contour.interpFunction()

    @np.vectorize
    def wrapped_interp(dist: float) -> tuple[float, float]:
        point = interp(dist)
        try:
            tangent_point = interp(dist + 0.05 * (end - start))
            tangent = tangent_point - point
        except ValueError:
            tangent_point = interp(dist - 0.05 * (end - start))
            tangent = point - tangent_point
        return tuple(contour.refinePoint(point, tangent, psi=eq.psi))

    def func(s: npt.ArrayLike) -> SliceCoords:
        dists = start + (end - start) * np.asarray(s)
        R, Z = wrapped_interp(dists)
        return SliceCoords(R, Z, CoordinateSystem.CYLINDRICAL)

    return func
