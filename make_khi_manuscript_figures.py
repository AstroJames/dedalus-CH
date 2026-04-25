"""Create manuscript-ready KHI validation and variable-comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="dedalus/v2/output/khi_v2_compare_128_t10tsh_dealias.h5",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--rc-file", default=None)
    return parser.parse_args()


def apply_figure_rc() -> None:
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def shear_time(params) -> float:
    return float(params["khi_width_fraction"] * params["ly"] / params["khi_u0"])


def fft_vorticity(ux: np.ndarray, uy: np.ndarray, lx: float, ly: float) -> np.ndarray:
    nx, ny = ux.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)[None, :]
    return np.fft.ifftn(1j * kx * np.fft.fftn(uy) - 1j * ky * np.fft.fftn(ux)).real


def robust_limits(values: list[np.ndarray], symmetric: bool = False, q: float = 99.5):
    flat = np.concatenate([np.ravel(v) for v in values])
    if symmetric:
        vmax = float(np.percentile(np.abs(flat), q))
        if vmax == 0:
            vmax = float(np.max(np.abs(flat))) or 1.0
        return -vmax, vmax
    return float(np.percentile(flat, 100 - q)), float(np.percentile(flat, q))


def add_colorbar(fig, image, axis, scientific: bool = False) -> None:
    cbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.012)
    cbar.ax.tick_params(labelsize=6)
    if scientific:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        cbar.formatter = fmt
        cbar.update_ticks()
        cbar.ax.yaxis.get_offset_text().set_size(7)


def image_panel(fig, ax, field, title, lx, ly, cmap, limits, scientific=False) -> None:
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=(0, lx, 0, ly),
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="bilinear",
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    add_colorbar(fig, im, ax, scientific=scientific)


def integral_diagnostics(h5: h5py.File) -> dict[str, np.ndarray]:
    params = h5["params"].attrs
    snaps = h5["snapshots"]
    lx = float(params["lx"])
    ly = float(params["ly"])
    cs = float(params["c_s"])
    tsh = shear_time(params)

    dxdy = lx * ly / snaps["rho_direct"][0].size
    t = snaps["t"][:] / tsh
    out: dict[str, list[float]] = {
        "t": list(t),
        "px_direct": [],
        "py_direct": [],
        "px_reconstructed": [],
        "py_reconstructed": [],
        "energy_direct": [],
        "energy_reconstructed": [],
    }
    for suffix in ("direct", "reconstructed"):
        rho_all = snaps[f"rho_{suffix}"][:]
        ux_all = snaps[f"ux_{suffix}"][:]
        uy_all = snaps[f"uy_{suffix}"][:]
        for rho, ux, uy in zip(rho_all, ux_all, uy_all, strict=True):
            out[f"px_{suffix}"].append(float(np.sum(rho * ux) * dxdy))
            out[f"py_{suffix}"].append(float(np.sum(rho * uy) * dxdy))
            kinetic = 0.5 * rho * (ux**2 + uy**2)
            internal = cs**2 * (rho * np.log(rho) - rho + 1.0)
            out[f"energy_{suffix}"].append(float(np.sum(kinetic + internal) * dxdy))
    return {key: np.asarray(value) for key, value in out.items()}


def plot_integral_diagnostics(h5: h5py.File, output_dir: Path) -> Path:
    vals = integral_diagnostics(h5)
    t = vals["t"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), constrained_layout=False)

    ax = axes[0]
    ax.plot(t, vals["px_direct"], "o-", ms=3, label=r"$P_x$ direct")
    ax.plot(t, vals["px_reconstructed"], "o--", ms=3, label=r"$P_x$ rec.")
    ax.plot(t, vals["py_direct"], "s-", ms=3, label=r"$P_y$ direct")
    ax.plot(t, vals["py_reconstructed"], "s--", ms=3, label=r"$P_y$ rec.")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_title(r"(a) total momentum")
    ax.set_xlabel(r"$t/t_{\rm sh}$")
    ax.set_ylabel(r"$P_i=\int \rho u_i\,dA$")
    ax.grid(True, alpha=0.28)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(frameon=False, ncol=2, loc="best", handlelength=1.4, columnspacing=0.9)

    ax = axes[1]
    e0_direct = vals["energy_direct"][0]
    e0_rec = vals["energy_reconstructed"][0]
    ax.plot(t, vals["energy_direct"] / e0_direct, "o-", ms=3, label="direct")
    ax.plot(t, vals["energy_reconstructed"] / e0_rec, "s--", ms=3, label="rec.")
    ax.set_title(r"(b) total isothermal free energy")
    ax.set_xlabel(r"$t/t_{\rm sh}$")
    ax.set_ylabel(r"$E(t)/E(0)$")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="best", handlelength=1.4)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.28)
    path = output_dir / "khi_integral_diagnostics_128_t10tsh.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_validation(h5: h5py.File, output_dir: Path) -> Path:
    params = h5["params"].attrs
    diag = h5["diagnostics"]
    snaps = h5["snapshots"]
    lx = float(params["lx"])
    ly = float(params["ly"])
    tsh = shear_time(params)
    final = -1
    final_time = float(snaps["t"][final] / tsh)

    ux_rec = snaps["ux_reconstructed"][final]
    uy_rec = snaps["uy_reconstructed"][final]
    ux_dir = snaps["ux_direct"][final]
    uy_dir = snaps["uy_direct"][final]
    omega_rec = fft_vorticity(ux_rec, uy_rec, lx, ly)
    omega_dir = fft_vorticity(ux_dir, uy_dir, lx, ly)
    omega_lim = robust_limits([omega_rec, omega_dir], symmetric=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=False)
    _ = final_time

    ax = axes[0]
    t_diag = diag["t"][:] / tsh
    ax.semilogy(t_diag, diag["rho_rel_l2"][:], "o-", ms=3, label=r"$\rho$")
    ax.semilogy(t_diag, diag["u_rel_l2"][:], "s-", ms=3, label=r"$\bm{u}$")
    ax.set_title(r"(a) error history")
    ax.set_xlabel(r"$t/t_{\rm sh}$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_xlim(0, max(10, float(t_diag[-1])))
    ax.grid(True, alpha=0.28)
    ax.legend(
        frameon=False,
        loc="lower right",
        handlelength=1.2,
        borderpad=0.2,
        labelspacing=0.25,
    )

    image_panel(fig, axes[1], omega_rec, r"(b) reconstructed $\omega$", lx, ly, "RdBu_r", omega_lim)
    image_panel(fig, axes[2], omega_dir, r"(c) direct $\omega$", lx, ly, "RdBu_r", omega_lim)

    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.20, top=0.88, wspace=0.12)
    path = output_dir / "khi_validation_128_t10tsh.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_variable_comparison(h5: h5py.File, output_dir: Path) -> Path:
    params = h5["params"].attrs
    snaps = h5["snapshots"]
    lx = float(params["lx"])
    ly = float(params["ly"])
    tsh = shear_time(params)
    final = -1
    final_time = float(snaps["t"][final] / tsh)

    rho = snaps["rho_reconstructed"][final]
    ux = snaps["ux_reconstructed"][final]
    uy = snaps["uy_reconstructed"][final]
    omega = fft_vorticity(ux, uy, lx, ly)
    tau = snaps["tau"][final]
    chi = snaps["chi"][final]
    log_psi = np.log(snaps["Psi"][final])

    physical_fields = [
        (rho - 1.0, r"(a) $\rho-1$", "RdBu_r", robust_limits([rho - 1.0], symmetric=True)),
        (omega, r"(b) $\omega$", "RdBu_r", robust_limits([omega], symmetric=True)),
        (uy, r"(c) $u_y$", "RdBu_r", robust_limits([uy], symmetric=True)),
    ]
    transformed_fields = [
        (
            tau - np.mean(tau),
            r"(d) $\tau-\langle\tau\rangle$",
            "RdBu_r",
            robust_limits([tau - np.mean(tau)], symmetric=True),
        ),
        (chi, r"(e) $\chi=\ln\Xi$", "RdBu_r", robust_limits([chi], symmetric=True)),
        (
            log_psi - np.mean(log_psi),
            r"(f) $\ln\Psi_\alpha-\langle\ln\Psi_\alpha\rangle$",
            "RdBu_r",
            robust_limits([log_psi - np.mean(log_psi)], symmetric=True),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.65), constrained_layout=False)
    _ = final_time
    for ax, (field, title, cmap, limits) in zip(axes[0], physical_fields, strict=True):
        image_panel(fig, ax, field, title, lx, ly, cmap, limits)
    for ax, (field, title, cmap, limits) in zip(axes[1], transformed_fields, strict=True):
        image_panel(fig, ax, field, title, lx, ly, cmap, limits)

    axes[0, 0].set_ylabel("fluid", fontsize=8, labelpad=8)
    axes[1, 0].set_ylabel("Schr. variables", fontsize=8, labelpad=8)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.93, wspace=0.10, hspace=0.16)
    path = output_dir / "khi_schrodinger_fluid_128_t10tsh.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.rc_file:
        matplotlib.rc_file(args.rc_file)
    apply_figure_rc()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5:
        paths = [
            plot_validation(h5, output_dir),
            plot_variable_comparison(h5, output_dir),
            plot_integral_diagnostics(h5, output_dir),
        ]

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
