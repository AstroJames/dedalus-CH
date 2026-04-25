"""Full-Dedalus direct 2D compressible-NS Kelvin-Helmholtz run.

This is the first v2 target: evolve the direct compressible Navier-Stokes
reference system with Dedalus's IVP machinery, then post-process each saved
snapshot into the transformed variables for visual comparison.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

from poisson_projection import ZeroMeanPoissonSolver, global_l2


Array = np.ndarray


@dataclass
class Params:
    """Numerical and physical parameters shared by all validation drivers."""

    nx: int = 128
    ny: int = 128
    lx: float = 2 * np.pi
    ly: float = 2 * np.pi
    c_s: float = 1.0
    nu: float = 5.0e-4
    zeta: float = 5.0e-2
    alpha: float = 0.25
    mu_s: float = 1.0
    rho0: float = 1.0
    khi_u0: float = 1.0
    khi_perturb: float = 0.5
    khi_width_fraction: float = 3.0e-2
    khi_perturb_width_fraction: float = 6.0e-2
    khi_density_contrast: float = 0.0
    khi_mode: int = 2
    dt: float = 5.0e-4
    t_end: float = 2.0
    snapshot_dt: float = 0.5
    diag_dt: float = 0.25
    output: str = "output/khi_direct_ivp.h5"
    mpi_smoke: bool = False

    @property
    def mu_c(self) -> float:
        """Longitudinal viscosity coefficient in the compressive potential."""
        return self.zeta + 2 * self.nu

    @property
    def bulk(self) -> float:
        """Coefficient multiplying grad div(u) in the direct velocity equation."""
        return self.zeta + self.nu

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must be at least 8")
        if self.dt <= 0 or self.t_end <= 0:
            raise ValueError("dt and t_end must be positive")
        if self.snapshot_dt <= 0 or self.diag_dt <= 0:
            raise ValueError("snapshot_dt and diag_dt must be positive")
        if self.khi_width_fraction <= 0 or self.khi_perturb_width_fraction <= 0:
            raise ValueError("KHI width fractions must be positive")
        if abs(self.khi_density_contrast) >= 1:
            raise ValueError("abs(khi_density_contrast) must be < 1")
        if self.khi_mode < 1:
            raise ValueError("khi_mode must be at least 1")
        if np.isclose(self.alpha, 0) or np.isclose(self.alpha, 0.5):
            raise ValueError("alpha must be nonzero and not equal to 1/2")
        if self.mu_s <= 0 or self.mu_c <= 0:
            raise ValueError("mu_s and mu_c must be positive")


class SpectralPost:
    """Single-rank periodic FFT helpers for diagnostics and post-processing.

    These helpers are used only on gathered arrays. The production evolution
    and MPI-safe projections use Dedalus fields and LBVPs instead.
    """

    def __init__(self, params: Params):
        self.params = params
        self.kx = 2 * np.pi * np.fft.fftfreq(params.nx, d=params.lx / params.nx)
        self.ky = 2 * np.pi * np.fft.fftfreq(params.ny, d=params.ly / params.ny)
        self.k2 = self.kx[:, None] ** 2 + self.ky[None, :] ** 2

    def dx(self, data: Array) -> Array:
        return np.fft.ifftn(1j * self.kx[:, None] * np.fft.fftn(data)).real

    def dy(self, data: Array) -> Array:
        return np.fft.ifftn(1j * self.ky[None, :] * np.fft.fftn(data)).real

    def lap(self, data: Array) -> Array:
        return np.fft.ifftn(-self.k2 * np.fft.fftn(data)).real

    def div(self, ux: Array, uy: Array) -> Array:
        return self.dx(ux) + self.dy(uy)

    def omega(self, ux: Array, uy: Array) -> Array:
        return self.dx(uy) - self.dy(ux)

    def poisson_zero_mean(self, source: Array) -> Array:
        source_hat = np.fft.fftn(source)
        out_hat = np.zeros_like(source_hat)
        mask = self.k2 != 0
        out_hat[mask] = -source_hat[mask] / self.k2[mask]
        return np.fft.ifftn(out_hat).real

    def l2(self, data: Array) -> float:
        return float(np.sqrt(np.mean(np.asarray(data) ** 2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
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
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--snapshot-dt", type=float, default=0.5)
    parser.add_argument("--diag-dt", type=float, default=0.25)
    parser.add_argument("--output", default="output/khi_direct_ivp.h5")
    parser.add_argument(
        "--mpi-smoke",
        action="store_true",
        help="Run only MPI-safe IVP stepping diagnostics; skip post-processed snapshots.",
    )
    return parser.parse_args()


def build_domain(params: Params):
    """Create the 2D periodic Dedalus domain used by all solvers."""
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=params.nx, bounds=(0, params.lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=params.ny, bounds=(0, params.ly), dealias=3 / 2)
    x, y = dist.local_grids(xbasis, ybasis)
    return coords, dist, xbasis, ybasis, np.asarray(x + 0 * y), np.asarray(0 * x + y)


def khi_initial_arrays(params: Params, x: Array, y: Array) -> tuple[Array, Array, Array]:
    """Return a divergence-free single-rank Kelvin--Helmholtz initial state.

    The base shear is built through a streamfunction so that the validation
    starts from a clean Helmholtz decomposition. This routine relies on global
    FFTs and is therefore used only for single-rank direct runs.
    """
    post = SpectralPost(params)
    y1 = 0.25 * params.ly
    y2 = 0.75 * params.ly
    shear_width = params.khi_width_fraction * params.ly
    perturb_width = params.khi_perturb_width_fraction * params.ly
    profile = np.tanh((y - y1) / shear_width) - np.tanh((y - y2) / shear_width) - 1
    envelope = np.exp(-((y - y1) / perturb_width) ** 2) + np.exp(
        -((y - y2) / perturb_width) ** 2
    )

    kx = 2 * np.pi * params.khi_mode / params.lx
    rho = params.rho0 * (1 + params.khi_density_contrast * profile)

    ux_base = params.khi_u0 * profile
    ux_base = ux_base - np.mean(ux_base)
    base_stream_hat = np.fft.fft(ux_base, axis=1)
    stream_hat = np.zeros_like(base_stream_hat, dtype=np.complex128)
    ky = post.ky[None, :]
    mask = ky != 0
    stream_hat[:, mask[0]] = base_stream_hat[:, mask[0]] / (1j * ky[:, mask[0]])
    base_stream = np.fft.ifft(stream_hat, axis=1).real

    perturb_stream = (params.khi_perturb / kx) * np.cos(kx * x) * envelope
    stream = base_stream + perturb_stream
    ux = post.dy(stream)
    uy = -post.dx(stream)
    return np.log(rho), ux, uy


def khi_initial_arrays_local(params: Params, x: Array, y: Array) -> tuple[Array, Array, Array]:
    """MPI-safe local KHI initializer for IVP smoke tests.

    This avoids global FFT projections. It is only used by ``--mpi-smoke`` to
    verify parallel Dedalus stepping; the single-rank validation initializer
    remains the divergence-free projected version above.
    """

    y1 = 0.25 * params.ly
    y2 = 0.75 * params.ly
    shear_width = params.khi_width_fraction * params.ly
    perturb_width = params.khi_perturb_width_fraction * params.ly
    profile = np.tanh((y - y1) / shear_width) - np.tanh((y - y2) / shear_width) - 1
    envelope = np.exp(-((y - y1) / perturb_width) ** 2) + np.exp(
        -((y - y2) / perturb_width) ** 2
    )
    kx = 2 * np.pi * params.khi_mode / params.lx
    rho = params.rho0 * (1 + params.khi_density_contrast * profile) + 0 * x
    ux = params.khi_u0 * profile + 0 * x
    uy = params.khi_perturb * np.sin(kx * x) * envelope
    return np.log(rho), ux, uy


def transformed_projection(
    params: Params,
    post: SpectralPost,
    poisson: ZeroMeanPoissonSolver,
    s: Array,
    ux: Array,
    uy: Array,
):
    """Project physical fields into the Cole--Hopf transformed variables.

    The velocity is split into compressive and solenoidal potentials,
    ``tau`` and ``chi``, by solving two zero-mean Poisson problems. The
    density-carrying scalar ``Psi`` is then built from ``rho`` and ``Theta``.
    """
    rho = np.exp(s)
    divu = post.div(ux, uy)
    omega = post.omega(ux, uy)
    tau_result = poisson.solve(-divu / (2 * params.mu_c))
    chi_result = poisson.solve(-omega / (2 * params.mu_s))
    tau = tau_result.data
    chi = chi_result.data
    theta = np.exp(tau)
    xi = np.exp(chi)
    psi = rho**params.alpha * theta ** (1 - 2 * params.alpha)
    return {
        "rho": rho,
        "omega": omega,
        "tau": tau,
        "chi": chi,
        "Theta": theta,
        "Xi": xi,
        "Psi": psi,
        "poisson_tau_l2": tau_result.residual_l2,
        "poisson_chi_l2": chi_result.residual_l2,
    }


def append_snapshot(
    store: dict[str, list],
    params: Params,
    post: SpectralPost,
    poisson: ZeroMeanPoissonSolver,
    solver,
    s,
    ux,
    uy,
) -> None:
    """Store a full physical/transformed snapshot in the in-memory HDF5 buffer."""
    s.change_scales(1)
    ux.change_scales(1)
    uy.change_scales(1)
    s_g = np.asarray(s["g"]).copy()
    ux_g = np.asarray(ux["g"]).copy()
    uy_g = np.asarray(uy["g"]).copy()
    trans = transformed_projection(params, post, poisson, s_g, ux_g, uy_g)
    values = {
        "t": solver.sim_time,
        "s": s_g,
        "rho": trans["rho"],
        "ux": ux_g,
        "uy": uy_g,
        "omega": trans["omega"],
        "tau": trans["tau"],
        "chi": trans["chi"],
        "Theta": trans["Theta"],
        "Xi": trans["Xi"],
        "Psi": trans["Psi"],
    }
    for key, value in values.items():
        store.setdefault(key, []).append(value)


def append_diag(
    store: dict[str, list],
    params: Params,
    post: SpectralPost,
    poisson: ZeroMeanPoissonSolver,
    solver,
    s,
    ux,
    uy,
) -> None:
    """Append lightweight scalar diagnostics for the direct run."""
    s.change_scales(1)
    ux.change_scales(1)
    uy.change_scales(1)
    s_g = np.asarray(s["g"])
    ux_g = np.asarray(ux["g"])
    uy_g = np.asarray(uy["g"])
    trans = transformed_projection(params, post, poisson, s_g, ux_g, uy_g)
    values = {
        "t": solver.sim_time,
        "rho_min": float(np.min(trans["rho"])),
        "rho_max": float(np.max(trans["rho"])),
        "u_rms": global_l2(poisson.dist, np.sqrt(ux_g**2 + uy_g**2)),
        "omega_rms": global_l2(poisson.dist, trans["omega"]),
        "theta_min": float(np.min(trans["Theta"])),
        "xi_min": float(np.min(trans["Xi"])),
        "psi_min": float(np.min(trans["Psi"])),
        "poisson_tau_l2": trans["poisson_tau_l2"],
        "poisson_chi_l2": trans["poisson_chi_l2"],
    }
    for key, value in values.items():
        store.setdefault(key, []).append(float(value))


def write_hdf5(path: str | Path, params: Params, diagnostics: dict[str, list], snapshots: dict[str, list]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        pgrp = h5.create_group("params")
        for key, value in asdict(params).items():
            pgrp.attrs[key] = value
        dgrp = h5.create_group("diagnostics")
        for key, values in diagnostics.items():
            dgrp.create_dataset(key, data=np.asarray(values))
        sgrp = h5.create_group("snapshots")
        for key, values in snapshots.items():
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
    s = dist.Field(name="s", bases=(xbasis, ybasis))
    ux = dist.Field(name="ux", bases=(xbasis, ybasis))
    uy = dist.Field(name="uy", bases=(xbasis, ybasis))

    if params.mpi_smoke:
        s0, ux0, uy0 = khi_initial_arrays_local(params, x, y)
    else:
        s0, ux0, uy0 = khi_initial_arrays(params, x, y)
    s["g"] = s0
    ux["g"] = ux0
    uy["g"] = uy0

    dx = lambda a: d3.Differentiate(a, coords["x"])
    dy = lambda a: d3.Differentiate(a, coords["y"])
    lap = lambda a: d3.Laplacian(a)
    divu = dx(ux) + dy(uy)
    cs2 = params.c_s**2
    nu = params.nu
    bulk = params.bulk

    # Direct isothermal compressible Navier--Stokes system in logarithmic
    # density. Dedalus keeps the viscous linear terms on the left-hand side
    # and evaluates the advection/pressure terms explicitly through RK443.
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

    comm = dist.comm
    rank = comm.rank
    post = SpectralPost(params)
    poisson = ZeroMeanPoissonSolver(coords, dist, (xbasis, ybasis), name="projection")
    diagnostics: dict[str, list] = {}
    snapshots: dict[str, list] = {}
    next_diag = 0.0
    next_snapshot = 0.0

    while solver.proceed:
        if params.mpi_smoke:
            if solver.sim_time >= next_diag - 0.5 * params.dt:
                s.change_scales(1)
                ux.change_scales(1)
                uy.change_scales(1)
                local_rho_min = float(np.min(np.exp(s["g"])))
                local_u2_mean = float(np.mean(ux["g"] ** 2 + uy["g"] ** 2))
                rho_min = comm.allreduce(local_rho_min, op=MPI.MIN)
                u2_mean = comm.allreduce(local_u2_mean, op=MPI.SUM) / comm.size
                if rank == 0:
                    print(
                        f"step={solver.iteration:06d} t={solver.sim_time:.6e} "
                        f"rho_min={rho_min:.3e} u_rms={np.sqrt(u2_mean):.3e}"
                    )
                next_diag += params.diag_dt
        elif solver.sim_time >= next_diag - 0.5 * params.dt:
            append_diag(diagnostics, params, post, poisson, solver, s, ux, uy)
            latest = {key: values[-1] for key, values in diagnostics.items()}
            print(
                f"step={solver.iteration:06d} t={solver.sim_time:.6e} "
                f"rho_min={latest['rho_min']:.3e} u_rms={latest['u_rms']:.3e} "
                f"theta_min={latest['theta_min']:.3e}"
            )
            next_diag += params.diag_dt
        if (not params.mpi_smoke) and solver.sim_time >= next_snapshot - 0.5 * params.dt:
            append_snapshot(snapshots, params, post, poisson, solver, s, ux, uy)
            next_snapshot += params.snapshot_dt

        solver.step(params.dt)
        s.change_scales(1)
        ux.change_scales(1)
        uy.change_scales(1)
        state_values = [s["g"], ux["g"], uy["g"]]
        if any(not np.all(np.isfinite(value)) for value in state_values):
            raise FloatingPointError(f"non-finite state at step {solver.iteration}")
        if np.min(np.exp(s["g"])) <= 0:
            raise FloatingPointError(f"non-positive density at step {solver.iteration}")

    if params.mpi_smoke:
        if rank == 0:
            print("MPI smoke run completed")
        return

    if not snapshots["t"] or snapshots["t"][-1] < params.t_end - 0.5 * params.dt:
        append_snapshot(snapshots, params, post, poisson, solver, s, ux, uy)
    if not diagnostics["t"] or diagnostics["t"][-1] < params.t_end - 0.5 * params.dt:
        append_diag(diagnostics, params, post, poisson, solver, s, ux, uy)

    write_hdf5(params.output, params, diagnostics, snapshots)
    print(f"wrote {params.output}")


if __name__ == "__main__":
    main()
