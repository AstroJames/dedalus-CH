"""MPI-safe transformed 2D solver using Dedalus fields and LBVP projections.

This is not a symbolic ``d3.IVP`` because the transformed RHS requires
nonlinear Poisson projections at every RK stage. It is nevertheless a Dedalus
solver path: derivatives are Dedalus operators on distributed fields, and all
periodic Poisson solves use the Dedalus LBVP helper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

from direct_ivp_2d import Params, build_domain, khi_initial_arrays_local
from poisson_projection import ZeroMeanPoissonSolver, global_l2


Array = np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhs-smoke", default=None)
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
    parser.add_argument("--khi-mode", type=int, default=2)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--t-end", type=float, default=0.01)
    parser.add_argument("--diag-dt", type=float, default=0.0025)
    parser.add_argument("--snapshot-dt", type=float, default=0.005)
    parser.add_argument("--dealias", type=float, default=3 / 2)
    parser.add_argument("--output", default="output/khi_transformed_rk.h5")
    parser.add_argument("--mpi-smoke", action="store_true")
    return parser.parse_args()


def params_from_hdf5(h5: h5py.File) -> Params:
    """Reconstruct a ``Params`` instance from an output file's attributes."""
    attrs = h5["params"].attrs
    kwargs = {}
    for field in Params.__dataclass_fields__:
        if field in attrs:
            value = attrs[field]
            if isinstance(value, np.generic):
                value = value.item()
            kwargs[field] = value
    return Params(**kwargs)


class FieldOps:
    """Dedalus operator helpers returning local grid arrays at requested scale.

    The transformed RHS mixes physical-space nonlinear products with spectral
    derivatives. These wrappers keep the scale conversion explicit, which helps
    avoid silently differentiating arrays at the wrong dealiasing scale.
    """

    def __init__(self, coords, dist, bases, dealias: float = 3 / 2):
        self.coords = coords
        self.dist = dist
        self.bases = bases
        self.dealias = dealias
        self.tmp = dist.Field(name="tmp", bases=bases)

    def field(self, data: Array, name: str = "tmp", scale: float = 1):
        """Wrap a local grid array in a Dedalus field at the requested scale."""
        f = self.dist.Field(name=name, bases=self.bases)
        f.change_scales(scale)
        f["g"] = np.asarray(data)
        return f

    def set_tmp(self, data: Array, scale: float = 1):
        self.tmp.change_scales(scale)
        self.tmp["g"] = np.asarray(data)
        return self.tmp

    def eval_array(self, expr, scale: float = 1) -> Array:
        """Evaluate a Dedalus operator expression and return local grid data."""
        out = expr.evaluate()
        out.change_scales(scale)
        return np.asarray(out["g"]).copy()

    def to_scale(self, data: Array, input_scale: float = 1, output_scale: float = 1) -> Array:
        if input_scale == output_scale:
            return np.asarray(data).copy()
        f = self.field(data, name="scale_convert", scale=input_scale)
        f.change_scales(output_scale)
        return np.asarray(f["g"]).copy()

    def dx(self, data: Array, scale: float = 1) -> Array:
        return self.eval_array(d3.Differentiate(self.set_tmp(data, scale), self.coords["x"]), scale)

    def dy(self, data: Array, scale: float = 1) -> Array:
        return self.eval_array(d3.Differentiate(self.set_tmp(data, scale), self.coords["y"]), scale)

    def lap(self, data: Array, scale: float = 1) -> Array:
        return self.eval_array(d3.Laplacian(self.set_tmp(data, scale)), scale)

    def div(self, fx: Array, fy: Array, scale: float = 1) -> Array:
        fx_f = self.field(fx, name="fx", scale=scale)
        fy_f = self.field(fy, name="fy", scale=scale)
        return self.eval_array(
            d3.Differentiate(fx_f, self.coords["x"]) + d3.Differentiate(fy_f, self.coords["y"]),
            scale,
        )

    def curl_scalar(self, fx: Array, fy: Array, scale: float = 1) -> Array:
        fx_f = self.field(fx, name="fx", scale=scale)
        fy_f = self.field(fy, name="fy", scale=scale)
        return self.eval_array(
            d3.Differentiate(fy_f, self.coords["x"]) - d3.Differentiate(fx_f, self.coords["y"]),
            scale,
        )


class TransformedRHS:
    """Evaluate the transformed ``Theta, Xi, Psi`` right-hand side."""

    def __init__(self, params: Params, coords, dist, bases, dealias: float = 3 / 2):
        self.params = params
        self.ops = FieldOps(coords, dist, bases, dealias=dealias)
        self.poisson = ZeroMeanPoissonSolver(coords, dist, bases, name="transformed")

    @property
    def dist(self):
        return self.ops.dist

    def reconstruct(
        self,
        state: dict[str, Array],
        input_scale: float = 1,
        output_scale: float = 1,
    ) -> dict[str, Array]:
        """Map transformed variables back to ``rho`` and velocity.

        The reconstruction uses ``tau=log(Theta)`` and ``chi=log(Xi)`` with
        the potential split
        ``u=(-2 mu_c grad tau) + 2 mu_s curl_perp chi``.
        """
        p = self.params
        theta = self.ops.to_scale(state["Theta"], input_scale, output_scale)
        xi = self.ops.to_scale(state["Xi"], input_scale, output_scale)
        psi = self.ops.to_scale(state["Psi"], input_scale, output_scale)
        tau = np.log(theta)
        chi = np.log(xi)
        s = (np.log(psi) - (1 - 2 * p.alpha) * tau) / p.alpha
        tau_x, tau_y = self.ops.dx(tau, output_scale), self.ops.dy(tau, output_scale)
        chi_x, chi_y = self.ops.dx(chi, output_scale), self.ops.dy(chi, output_scale)
        ux = -2 * p.mu_c * tau_x + 2 * p.mu_s * chi_y
        uy = -2 * p.mu_c * tau_y - 2 * p.mu_s * chi_x
        omega = -2 * p.mu_s * self.ops.lap(chi, output_scale)
        return {
            "tau": tau,
            "chi": chi,
            "s": s,
            "rho": np.exp(s),
            "ux": ux,
            "uy": uy,
            "omega": omega,
        }

    def forcing_potentials(self, rec: dict[str, Array], scale: float = 1) -> dict[str, Array]:
        """Project the nonlinear vortex force into scalar potentials.

        In two dimensions ``-u x omega`` is represented as a planar force.
        Its divergence and scalar curl define the compressive and solenoidal
        forcing potentials through zero-mean Poisson solves.
        """
        fx = -rec["uy"] * rec["omega"]
        fy = rec["ux"] * rec["omega"]
        div_f = self.ops.div(fx, fy, scale)
        skew_div_f = self.ops.curl_scalar(fx, fy, scale)
        div_solve = self.ops.to_scale(div_f, scale, 1)
        skew_div_solve = self.ops.to_scale(skew_div_f, scale, 1)
        phi = self.poisson.solve(div_solve)
        psi = self.poisson.solve(-skew_div_solve)
        return {
            "Phi_F": self.ops.to_scale(phi.data, 1, scale),
            "Psi_F": self.ops.to_scale(psi.data, 1, scale),
            "poisson_phi_l2": phi.residual_l2,
            "poisson_psi_l2": psi.residual_l2,
        }

    def rhs(self, state: dict[str, Array]) -> dict[str, Array]:
        """Return one transformed RHS evaluation on local grid arrays."""
        p = self.params
        alpha = p.alpha
        mu_c = p.mu_c
        mu_s = p.mu_s
        d_alpha = mu_c / (1 - 2 * alpha)
        scale = self.ops.dealias
        # Nonlinear products are formed on the dealiasing grid, then converted
        # back to the stored grid before returning to the RK driver.
        state_work = {key: self.ops.to_scale(state[key], 1, scale) for key in ("Theta", "Xi", "Psi")}
        rec = self.reconstruct(state_work, input_scale=scale, output_scale=scale)
        pots = self.forcing_potentials(rec, scale)

        tau, chi, s = rec["tau"], rec["chi"], rec["s"]
        ux, uy = rec["ux"], rec["uy"]
        tau_x, tau_y = self.ops.dx(tau, scale), self.ops.dy(tau, scale)
        chi_x, chi_y = self.ops.dx(chi, scale), self.ops.dy(chi, scale)
        s_x, s_y = self.ops.dx(s, scale), self.ops.dy(s, scale)

        grad_chi_sq = chi_x**2 + chi_y**2
        grad_s_sq = s_x**2 + s_y**2
        grad_s_dot_tau = s_x * tau_x + s_y * tau_y
        u_dot_tau = ux * tau_x + uy * tau_y

        v_theta = (
            -2 * mu_s * (tau_x * chi_y - tau_y * chi_x)
            + (mu_s**2 / mu_c) * grad_chi_sq
            + (p.c_s**2 / (2 * mu_c)) * s
            + pots["Phi_F"] / (2 * mu_c)
        )
        v_xi = -pots["Psi_F"] / (2 * mu_s) - p.nu * grad_chi_sq
        u_alpha = (
            -(mu_c * alpha / (1 - 2 * alpha)) * self.ops.lap(s, scale)
            - (mu_c * alpha**2 / (1 - 2 * alpha)) * grad_s_sq
            - 2 * mu_c * alpha * grad_s_dot_tau
            + (1 - 2 * alpha) * (v_theta + u_dot_tau)
        )

        psi_x, psi_y = self.ops.dx(state_work["Psi"], scale), self.ops.dy(state_work["Psi"], scale)
        out_work = {
            "Theta": mu_c * self.ops.lap(state_work["Theta"], scale) + v_theta * state_work["Theta"],
            "Xi": p.nu * self.ops.lap(state_work["Xi"], scale) + v_xi * state_work["Xi"],
            "Psi": d_alpha * self.ops.lap(state_work["Psi"], scale)
            - (ux * psi_x + uy * psi_y)
            + u_alpha * state_work["Psi"],
            "poisson_phi_l2": pots["poisson_phi_l2"],
            "poisson_psi_l2": pots["poisson_psi_l2"],
        }
        out = {
            key: self.ops.to_scale(value, scale, 1) if key in ("Theta", "Xi", "Psi") else value
            for key, value in out_work.items()
        }
        return out


def transformed_from_physical(params: Params, rhs_eval: TransformedRHS, s: Array, ux: Array, uy: Array):
    """Initialize transformed variables from physical logarithmic density and velocity."""
    ops = rhs_eval.ops
    divu = ops.div(ux, uy)
    omega = ops.curl_scalar(ux, uy)
    tau = rhs_eval.poisson.solve(-divu / (2 * params.mu_c)).data
    chi = rhs_eval.poisson.solve(-omega / (2 * params.mu_s)).data
    theta = np.exp(tau)
    xi = np.exp(chi)
    psi = np.exp(s) ** params.alpha * theta ** (1 - 2 * params.alpha)
    return {"Theta": theta, "Xi": xi, "Psi": psi}


def rk4_step(rhs_eval: TransformedRHS, state: dict[str, Array], dt: float) -> dict[str, Array]:
    """Advance the transformed variables by one explicit RK4 step."""
    keys = ["Theta", "Xi", "Psi"]

    def combine(base, inc, scale):
        return {key: base[key] + scale * inc[key] for key in keys}

    k1 = rhs_eval.rhs(state)
    k2 = rhs_eval.rhs(combine(state, k1, 0.5 * dt))
    k3 = rhs_eval.rhs(combine(state, k2, 0.5 * dt))
    k4 = rhs_eval.rhs(combine(state, k3, dt))
    return {
        key: state[key] + (dt / 6) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        for key in keys
    }


def positivity(rhs_eval: TransformedRHS, state: dict[str, Array]) -> dict[str, float]:
    """Return global positivity checks for logarithmic transform variables."""
    dist = rhs_eval.dist
    vals = {
        "theta_min": float(np.min(state["Theta"])),
        "xi_min": float(np.min(state["Xi"])),
        "psi_min": float(np.min(state["Psi"])),
    }
    return {key: dist.comm.allreduce(value, op=MPI.MIN) for key, value in vals.items()}


def append_diag(diags: dict[str, list], rhs_eval: TransformedRHS, t: float, state: dict[str, Array]) -> None:
    """Append scalar diagnostics for a transformed-only run."""
    rec = rhs_eval.reconstruct(state)
    mins = positivity(rhs_eval, state)
    vals = {
        "t": t,
        **mins,
        "rho_min": rhs_eval.dist.comm.allreduce(float(np.min(rec["rho"])), op=MPI.MIN),
        "u_rms": global_l2(rhs_eval.dist, np.sqrt(rec["ux"] ** 2 + rec["uy"] ** 2)),
        "omega_rms": global_l2(rhs_eval.dist, rec["omega"]),
    }
    for key, value in vals.items():
        diags.setdefault(key, []).append(float(value))


def append_snapshot(snaps: dict[str, list], rhs_eval: TransformedRHS, t: float, state: dict[str, Array]) -> None:
    """Append reconstructed physical and transformed fields to a local snapshot buffer."""
    rec = rhs_eval.reconstruct(state)
    vals = {
        "t": t,
        "Theta": state["Theta"],
        "Xi": state["Xi"],
        "Psi": state["Psi"],
        "rho": rec["rho"],
        "ux": rec["ux"],
        "uy": rec["uy"],
        "omega": rec["omega"],
        "tau": rec["tau"],
        "chi": rec["chi"],
    }
    for key, value in vals.items():
        snaps.setdefault(key, []).append(np.asarray(value).copy() if key != "t" else float(value))


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


def rhs_smoke(path: str) -> None:
    """Evaluate the transformed RHS on the final snapshot of a direct run."""
    with h5py.File(path, "r") as h5:
        params = params_from_hdf5(h5)
        s = h5["snapshots/s"][-1]
        ux = h5["snapshots/ux"][-1]
        uy = h5["snapshots/uy"][-1]

    coords, dist, xbasis, ybasis, *_ = build_domain(params)
    rhs_eval = TransformedRHS(params, coords, dist, (xbasis, ybasis), dealias=args.dealias)
    state = transformed_from_physical(params, rhs_eval, s, ux, uy)
    out = rhs_eval.rhs(state)
    if dist.comm.rank == 0:
        print("transformed RHS smoke")
        print(f"Theta_rhs_l2={global_l2(dist, out['Theta']):.6e}")
        print(f"Xi_rhs_l2={global_l2(dist, out['Xi']):.6e}")
        print(f"Psi_rhs_l2={global_l2(dist, out['Psi']):.6e}")
        print(f"Phi_poisson_l2={out['poisson_phi_l2']:.6e}")
        print(f"Psi_poisson_l2={out['poisson_psi_l2']:.6e}")


def run_transformed(args: argparse.Namespace) -> None:
    """Run the transformed solver as a standalone IVP."""
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
        khi_mode=args.khi_mode,
        dt=args.dt,
        t_end=args.t_end,
        diag_dt=args.diag_dt,
        snapshot_dt=args.snapshot_dt,
        output=args.output,
    )
    params.validate()
    coords, dist, xbasis, ybasis, x, y = build_domain(params)
    rhs_eval = TransformedRHS(params, coords, dist, (xbasis, ybasis))
    s0, ux0, uy0 = khi_initial_arrays_local(params, x, y)
    state = transformed_from_physical(params, rhs_eval, s0, ux0, uy0)

    diags: dict[str, list] = {}
    snaps: dict[str, list] = {}
    t = 0.0
    step = 0
    next_diag = 0.0
    next_snapshot = 0.0
    rank = dist.comm.rank

    while t < params.t_end - 0.5 * params.dt:
        if t >= next_diag - 0.5 * params.dt:
            append_diag(diags, rhs_eval, t, state)
            latest = {key: values[-1] for key, values in diags.items()}
            if rank == 0:
                print(
                    f"step={step:06d} t={t:.6e} rho_min={latest['rho_min']:.3e} "
                    f"theta_min={latest['theta_min']:.3e} u_rms={latest['u_rms']:.3e}"
                )
            next_diag += params.diag_dt
        if (not args.mpi_smoke) and t >= next_snapshot - 0.5 * params.dt:
            append_snapshot(snaps, rhs_eval, t, state)
            next_snapshot += params.snapshot_dt

        state = rk4_step(rhs_eval, state, params.dt)
        step += 1
        t += params.dt
        mins = positivity(rhs_eval, state)
        if any(not np.all(np.isfinite(value)) for value in state.values()):
            raise FloatingPointError(f"non-finite transformed state at step {step}")
        if min(mins.values()) <= 0:
            raise FloatingPointError(f"non-positive transformed state at step {step}: {mins}")

    append_diag(diags, rhs_eval, t, state)
    if not args.mpi_smoke:
        append_snapshot(snaps, rhs_eval, t, state)
        if rank == 0:
            write_hdf5(params.output, params, diags, snaps, {"dealias": args.dealias})
            print(f"wrote {params.output}")
    elif rank == 0:
        print("transformed MPI smoke run completed")


def main() -> None:
    args = parse_args()
    if args.rhs_smoke:
        rhs_smoke(args.rhs_smoke)
    else:
        run_transformed(args)


if __name__ == "__main__":
    main()
