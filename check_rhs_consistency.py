"""Check that the transformed RHS reproduces the direct physical RHS."""

from __future__ import annotations

import argparse

import numpy as np

from direct_ivp_2d import Params, build_domain, khi_initial_arrays_local
from poisson_projection import global_l2
from transformed_ivp_2d import TransformedRHS, transformed_from_physical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--dealias", type=float, default=3 / 2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = Params(nx=args.nx, ny=args.ny, dt=args.dt)
    coords, dist, xbasis, ybasis, x, y = build_domain(params)
    rhs_eval = TransformedRHS(params, coords, dist, (xbasis, ybasis), dealias=args.dealias)

    # Build a smooth KHI state, transform it, then compare the induced physical
    # time derivatives against the direct Navier--Stokes right-hand side.
    s0, ux0, uy0 = khi_initial_arrays_local(params, x, y)
    state = transformed_from_physical(params, rhs_eval, s0, ux0, uy0)
    rec = rhs_eval.reconstruct(state)
    transformed_rhs = rhs_eval.rhs(state)
    ops = rhs_eval.ops

    # Convert Theta_t, Xi_t, and Psi_t back into tau_t, chi_t, and s_t.
    tau_t = transformed_rhs["Theta"] / state["Theta"]
    chi_t = transformed_rhs["Xi"] / state["Xi"]
    s_t_trans = (
        transformed_rhs["Psi"] / state["Psi"] - (1 - 2 * params.alpha) * tau_t
    ) / params.alpha
    ux_t_trans = -2 * params.mu_c * ops.dx(tau_t) + 2 * params.mu_s * ops.dy(chi_t)
    uy_t_trans = -2 * params.mu_c * ops.dy(tau_t) - 2 * params.mu_s * ops.dx(chi_t)

    s = rec["s"]
    ux = rec["ux"]
    uy = rec["uy"]
    s_x, s_y = ops.dx(s), ops.dy(s)
    ux_x, ux_y = ops.dx(ux), ops.dy(ux)
    uy_x, uy_y = ops.dx(uy), ops.dy(uy)
    div_u = ux_x + uy_y

    s_t_dir = -ux * s_x - uy * s_y - div_u
    ux_t_dir = (
        -ux * ux_x
        - uy * ux_y
        - params.c_s**2 * s_x
        + params.nu * ops.lap(ux)
        + params.bulk * ops.dx(div_u)
    )
    uy_t_dir = (
        -ux * uy_x
        - uy * uy_y
        - params.c_s**2 * s_y
        + params.nu * ops.lap(uy)
        + params.bulk * ops.dy(div_u)
    )

    for name, transformed, direct in (
        ("s_t", s_t_trans, s_t_dir),
        ("ux_t", ux_t_trans, ux_t_dir),
        ("uy_t", uy_t_trans, uy_t_dir),
    ):
        err = global_l2(dist, transformed - direct)
        ref = global_l2(dist, direct)
        if dist.comm.rank == 0:
            print(f"{name}: abs={err:.6e} rel={err / max(ref, 1e-14):.6e} ref={ref:.6e}")

    if dist.comm.rank == 0:
        print(
            "poisson residuals: "
            f"Phi={transformed_rhs['poisson_phi_l2']:.6e} "
            f"Psi={transformed_rhs['poisson_psi_l2']:.6e}"
        )


if __name__ == "__main__":
    main()
