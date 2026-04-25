"""Lockstep KHI comparison between direct and transformed v2 solvers.

The direct reference system is advanced with a Dedalus IVP.  The transformed
system is advanced with the Dedalus/LBVP RHS implemented in
``transformed_ivp_2d.py``.  Both systems are initialized from the same
projected transform state so the initial reconstructed fields agree to
roundoff.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

from direct_ivp_2d import Params, build_domain, khi_initial_arrays_local
from poisson_projection import global_l2
from transformed_ivp_2d import TransformedRHS, rk4_step, transformed_from_physical


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nu", type=float, default=5e-4)
    parser.add_argument("--zeta", type=float, default=5e-2)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--mu-s", type=float, default=1.0)
    parser.add_argument("--khi-u0", type=float, default=1.0)
    parser.add_argument("--khi-perturb", type=float, default=0.5)
    parser.add_argument("--khi-width-fraction", type=float, default=3e-2)
    parser.add_argument("--khi-perturb-width-fraction", type=float, default=6e-2)
    parser.add_argument("--khi-density-contrast", type=float, default=0.0)
    parser.add_argument("--khi-mode", type=int, default=2)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--t-end", type=float, default=0.02)
    parser.add_argument("--diag-dt", type=float, default=0.005)
    parser.add_argument("--snapshot-dt", type=float, default=0.01)
    parser.add_argument("--dealias", type=float, default=3 / 2)
    parser.add_argument("--output", default="output/khi_v2_comparison.h5")
    parser.add_argument(
        "--mpi-smoke",
        action="store_true",
        help="Run distributed diagnostics only; skip HDF5 snapshots.",
    )
    return parser.parse_args()


def build_direct_ivp(params: Params, coords, dist, bases):
    xbasis, ybasis = bases
    s = dist.Field(name="s", bases=(xbasis, ybasis))
    ux = dist.Field(name="ux", bases=(xbasis, ybasis))
    uy = dist.Field(name="uy", bases=(xbasis, ybasis))

    dx = lambda a: d3.Differentiate(a, coords["x"])
    dy = lambda a: d3.Differentiate(a, coords["y"])
    lap = lambda a: d3.Laplacian(a)
    divu = dx(ux) + dy(uy)
    cs2 = params.c_s**2
    nu = params.nu
    bulk = params.bulk

    problem = d3.IVP([s, ux, uy], namespace=locals())
    problem.add_equation("dt(s) = - ux*dx(s) - uy*dy(s) - divu")
    problem.add_equation(
        "dt(ux) - nu*lap(ux) - bulk*dx(divu) = "
        "- ux*dx(ux) - uy*dy(ux) - cs2*dx(s)"
    )
    problem.add_equation(
        "dt(uy) - nu*lap(uy) - bulk*dy(divu) = "
        "- ux*dx(uy) - uy*dy(uy) - cs2*dy(s)"
    )
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = params.t_end
    return solver, s, ux, uy


def set_direct_fields(s_field, ux_field, uy_field, rec: dict[str, Array]) -> None:
    s_field.change_scales(1)
    ux_field.change_scales(1)
    uy_field.change_scales(1)
    s_field["g"] = rec["s"]
    ux_field["g"] = rec["ux"]
    uy_field["g"] = rec["uy"]


def direct_arrays(s_field, ux_field, uy_field) -> dict[str, Array]:
    s_field.change_scales(1)
    ux_field.change_scales(1)
    uy_field.change_scales(1)
    s = np.asarray(s_field["g"]).copy()
    ux = np.asarray(ux_field["g"]).copy()
    uy = np.asarray(uy_field["g"]).copy()
    return {"s": s, "rho": np.exp(s), "ux": ux, "uy": uy}


def rel_l2(dist, a: Array, b: Array, floor: float = 1.0e-14) -> float:
    return global_l2(dist, np.asarray(a) - np.asarray(b)) / max(global_l2(dist, b), floor)


def min_global(dist, data: Array) -> float:
    return dist.comm.allreduce(float(np.min(np.asarray(data))), op=MPI.MIN)


def diagnostics(
    params: Params,
    rhs_eval: TransformedRHS,
    t: float,
    direct: dict[str, Array],
    transformed_state: dict[str, Array],
    wall_elapsed: float,
) -> dict[str, float]:
    rec = rhs_eval.reconstruct(transformed_state)
    dist = rhs_eval.dist
    u_err = np.sqrt((rec["ux"] - direct["ux"]) ** 2 + (rec["uy"] - direct["uy"]) ** 2)
    u_ref = np.sqrt(direct["ux"] ** 2 + direct["uy"] ** 2)
    return {
        "t": float(t),
        "wall_elapsed": float(wall_elapsed),
        "rho_rel_l2": rel_l2(dist, rec["rho"], direct["rho"]),
        "s_rel_l2": rel_l2(dist, rec["s"], direct["s"]),
        "u_rel_l2": global_l2(dist, u_err) / max(global_l2(dist, u_ref), 1.0e-14),
        "ux_rel_l2": rel_l2(dist, rec["ux"], direct["ux"]),
        "uy_rel_l2": rel_l2(dist, rec["uy"], direct["uy"]),
        "rho_direct_min": min_global(dist, direct["rho"]),
        "rho_reconstructed_min": min_global(dist, rec["rho"]),
        "theta_min": min_global(dist, transformed_state["Theta"]),
        "xi_min": min_global(dist, transformed_state["Xi"]),
        "psi_min": min_global(dist, transformed_state["Psi"]),
        "u_direct_rms": global_l2(dist, u_ref),
        "u_reconstructed_rms": global_l2(dist, np.sqrt(rec["ux"] ** 2 + rec["uy"] ** 2)),
    }


def append_row(store: dict[str, list], row: dict[str, float]) -> None:
    for key, value in row.items():
        store.setdefault(key, []).append(float(value))


def append_snapshot(
    store: dict[str, list],
    rhs_eval: TransformedRHS,
    t: float,
    direct: dict[str, Array],
    transformed_state: dict[str, Array],
) -> None:
    rec = rhs_eval.reconstruct(transformed_state)
    values = {
        "rho_direct": direct["rho"],
        "rho_reconstructed": rec["rho"],
        "ux_direct": direct["ux"],
        "ux_reconstructed": rec["ux"],
        "uy_direct": direct["uy"],
        "uy_reconstructed": rec["uy"],
        "Theta": transformed_state["Theta"],
        "Xi": transformed_state["Xi"],
        "Psi": transformed_state["Psi"],
        "tau": rec["tau"],
        "chi": rec["chi"],
    }
    if rhs_eval.dist.comm.rank == 0:
        store.setdefault("t", []).append(float(t))
    for key, value in values.items():
        gathered = gather_grid(rhs_eval, value)
        if rhs_eval.dist.comm.rank == 0:
            store.setdefault(key, []).append(gathered)


def gather_grid(rhs_eval: TransformedRHS, data: Array) -> Array | None:
    field = rhs_eval.ops.field(data, name="snapshot_gather")
    field.change_scales(1)
    gathered = field.gather_data(root=0)
    if gathered is None:
        return None
    return np.asarray(gathered).copy()


def write_hdf5(
    path: str | Path,
    params: Params,
    diags: dict[str, list],
    snaps: dict[str, list],
    extra_attrs: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        pgrp = h5.create_group("params")
        for key, value in asdict(params).items():
            pgrp.attrs[key] = value
        for key, value in (extra_attrs or {}).items():
            pgrp.attrs[key] = value
        dgrp = h5.create_group("diagnostics")
        for key, values in diags.items():
            dgrp.create_dataset(key, data=np.asarray(values))
        sgrp = h5.create_group("snapshots")
        for key, values in snaps.items():
            sgrp.create_dataset(key, data=np.asarray(values))


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    args = parse_args()
    params = Params(
        nx=args.nx,
        ny=args.ny,
        nu=args.nu,
        zeta=args.zeta,
        alpha=args.alpha,
        mu_s=args.mu_s,
        khi_u0=args.khi_u0,
        khi_perturb=args.khi_perturb,
        khi_width_fraction=args.khi_width_fraction,
        khi_perturb_width_fraction=args.khi_perturb_width_fraction,
        khi_density_contrast=args.khi_density_contrast,
        khi_mode=args.khi_mode,
        dt=args.dt,
        t_end=args.t_end,
        snapshot_dt=args.snapshot_dt,
        diag_dt=args.diag_dt,
        output=args.output,
        mpi_smoke=args.mpi_smoke,
    )
    params.validate()

    coords, dist, xbasis, ybasis, x, y = build_domain(params)
    comm = dist.comm
    rank = comm.rank

    direct_solver, s_field, ux_field, uy_field = build_direct_ivp(
        params, coords, dist, (xbasis, ybasis)
    )
    rhs_eval = TransformedRHS(params, coords, dist, (xbasis, ybasis), dealias=args.dealias)

    s0, ux0, uy0 = khi_initial_arrays_local(params, x, y)
    transformed_state = transformed_from_physical(params, rhs_eval, s0, ux0, uy0)
    set_direct_fields(s_field, ux_field, uy_field, rhs_eval.reconstruct(transformed_state))

    diags: dict[str, list] = {}
    snaps: dict[str, list] = {}
    next_diag = 0.0
    next_snapshot = 0.0
    t0_wall = time.perf_counter()

    while direct_solver.sim_time < params.t_end - 1.0e-14:
        t = float(direct_solver.sim_time)
        if t >= next_diag - 0.5 * params.dt:
            direct = direct_arrays(s_field, ux_field, uy_field)
            row = diagnostics(
                params,
                rhs_eval,
                t,
                direct,
                transformed_state,
                time.perf_counter() - t0_wall,
            )
            append_row(diags, row)
            if rank == 0:
                print(
                    f"step={direct_solver.iteration:06d} t={t:.6e} "
                    f"rho_err={row['rho_rel_l2']:.3e} u_err={row['u_rel_l2']:.3e} "
                    f"rho_min=({row['rho_reconstructed_min']:.3e},{row['rho_direct_min']:.3e})"
                )
            next_diag += params.diag_dt
        if (not params.mpi_smoke) and t >= next_snapshot - 0.5 * params.dt:
            append_snapshot(snaps, rhs_eval, t, direct_arrays(s_field, ux_field, uy_field), transformed_state)
            next_snapshot += params.snapshot_dt

        step_dt = min(params.dt, params.t_end - float(direct_solver.sim_time))
        direct_solver.step(step_dt)
        transformed_state = rk4_step(rhs_eval, transformed_state, step_dt)

        direct = direct_arrays(s_field, ux_field, uy_field)
        state_values = [
            direct["rho"],
            direct["ux"],
            direct["uy"],
            transformed_state["Theta"],
            transformed_state["Xi"],
            transformed_state["Psi"],
        ]
        if any(not np.all(np.isfinite(value)) for value in state_values):
            raise FloatingPointError(f"non-finite comparison state at step {direct_solver.iteration}")
        if min_global(dist, direct["rho"]) <= 0 or min_global(dist, transformed_state["Theta"]) <= 0:
            raise FloatingPointError(f"non-positive comparison state at step {direct_solver.iteration}")

    final_t = float(direct_solver.sim_time)
    direct = direct_arrays(s_field, ux_field, uy_field)
    final_row = diagnostics(
        params,
        rhs_eval,
        final_t,
        direct,
        transformed_state,
        time.perf_counter() - t0_wall,
    )
    if not diags.get("t") or diags["t"][-1] < final_t - 0.5 * params.dt:
        append_row(diags, final_row)
    if (not params.mpi_smoke) and (
        not snaps.get("t") or snaps["t"][-1] < final_t - 0.5 * params.dt
    ):
        append_snapshot(snaps, rhs_eval, final_t, direct, transformed_state)

    wall = time.perf_counter() - t0_wall
    steps = max(direct_solver.iteration, 1)
    if rank == 0:
        print(
            f"completed steps={direct_solver.iteration} wall={wall:.3f}s "
            f"step_time={wall / steps:.6f}s rho_err={final_row['rho_rel_l2']:.3e} "
            f"u_err={final_row['u_rel_l2']:.3e}"
        )

    if params.mpi_smoke:
        if rank == 0:
            print("comparison MPI smoke run completed")
        return

    if rank == 0:
        write_hdf5(params.output, params, diags, snaps, {"dealias": args.dealias})
        print(f"wrote {params.output}")


if __name__ == "__main__":
    main()
