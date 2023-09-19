"""Functions for extruding hypnotoad meshes.

"""

import numpy as np
import numpy.typing as npt
from hypnotoad import Equilibrium  # type: ignore
from scipy.integrate import solve_ivp

from .mesh import CoordinateSystem, FieldTrace, SliceCoord, SliceCoords


def equilibrium_trace(equilibrium: Equilibrium) -> FieldTrace:
    """Return a field trace corresponding to the hypnotoad equilibrium
    object.

    """

    def d_dphi(t: float, y: npt.NDArray) -> npt.NDArray:
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
        return SliceCoords(
            R.reshape(shape), Z.reshape(shape), CoordinateSystem.CYLINDRICAL
        ), dist.reshape(shape)

    return trace
