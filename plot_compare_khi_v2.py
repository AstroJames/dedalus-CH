"""Plot v2 direct/transformed KHI comparison output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="output/khi_v2_comparison.h5")
    parser.add_argument("--output-dir", default="output/khi_v2_comparison_plots")
    parser.add_argument("--max-panels", type=int, default=6)
    return parser.parse_args()


def select_indices(n: int, max_panels: int) -> np.ndarray:
    """Choose evenly spaced snapshot indices for panel plots."""
    if n <= max_panels:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_panels).round().astype(int))


def robust_limits(values: list[np.ndarray], symmetric: bool = False):
    """Use percentile limits so isolated extrema do not dominate colormaps."""
    flat = np.concatenate([np.ravel(v) for v in values])
    if symmetric:
        vmax = np.percentile(np.abs(flat), 99.5)
        return -vmax, vmax
    return np.percentile(flat, 0.5), np.percentile(flat, 99.5)


def fft_vorticity(ux: np.ndarray, uy: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Compute periodic scalar vorticity from gathered velocity snapshots."""
    nx, ny = ux.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)[None, :]
    return np.fft.ifftn(1j * kx * np.fft.fftn(uy) - 1j * ky * np.fft.fftn(ux)).real


def x_perturbation(field: np.ndarray) -> np.ndarray:
    """Remove the x-average to emphasize roll-up perturbations."""
    return field - np.mean(field, axis=0, keepdims=True)


def add_colorbar(fig, image, axis) -> None:
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def plot_panel_grid(
    out_path: Path,
    times: np.ndarray,
    indices: np.ndarray,
    rows: list[dict[str, np.ndarray]],
    columns: list[tuple[str, str, str, tuple[float, float]]],
    lx: float,
    ly: float,
) -> None:
    fig, axes = plt.subplots(
        len(indices),
        len(columns),
        figsize=(3.2 * len(columns), 2.8 * len(indices)),
        squeeze=False,
    )
    for row, idx in enumerate(indices):
        fields = rows[row]
        for col, (key, title, cmap, limits) in enumerate(columns):
            ax = axes[row, col]
            im = ax.imshow(
                fields[key].T,
                origin="lower",
                extent=(0, lx, 0, ly),
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="bilinear",
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"t={times[idx]:.3f}")
            else:
                ax.set_yticks([])
            ax.set_xticks([])
            add_colorbar(fig, im, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_history(out_path: Path, h5: h5py.File) -> None:
    """Plot direct/reconstructed relative errors from the diagnostics group."""
    diag = h5["diagnostics"]
    t = diag["t"][:]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(t, diag["rho_rel_l2"][:], label=r"$\rho$ relative $L^2$")
    ax.semilogy(t, diag["u_rel_l2"][:], label=r"$u$ relative $L^2$")
    ax.semilogy(t, diag["s_rel_l2"][:], label=r"$s$ relative $L^2$")
    ax.set_xlabel("t")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as h5:
        params = h5["params"].attrs
        lx = float(params["lx"])
        ly = float(params["ly"])
        data = h5["snapshots"]
        times = data["t"][:]
        indices = select_indices(len(times), args.max_panels)

        omega_rec = [
            fft_vorticity(data["ux_reconstructed"][i], data["uy_reconstructed"][i], lx, ly)
            for i in indices
        ]
        omega_dir = [
            fft_vorticity(data["ux_direct"][i], data["uy_direct"][i], lx, ly)
            for i in indices
        ]
        omega_lim = robust_limits(omega_rec + omega_dir, symmetric=True)
        theta_lim = robust_limits([data["Theta"][i] for i in indices])
        xi_lim = robust_limits([data["Xi"][i] for i in indices])
        psi_lim = robust_limits([data["Psi"][i] for i in indices])

        rows = []
        for row, idx in enumerate(indices):
            rows.append(
                {
                    "Theta": data["Theta"][idx],
                    "Xi": data["Xi"][idx],
                    "Psi": data["Psi"][idx],
                    "omega_rec": omega_rec[row],
                    "omega_dir": omega_dir[row],
                }
            )
        plot_panel_grid(
            out_dir / "khi_v2_transformed_vs_direct.png",
            times,
            indices,
            rows,
            [
                ("Theta", "Theta", "viridis", theta_lim),
                ("Xi", "Xi", "magma", xi_lim),
                ("Psi", "Psi", "viridis", psi_lim),
                ("omega_rec", "omega reconstructed", "RdBu_r", omega_lim),
                ("omega_dir", "omega direct IVP", "RdBu_r", omega_lim),
            ],
            lx,
            ly,
        )

        tau_values = [data["tau"][i] for i in indices]
        chi_values = [data["chi"][i] for i in indices]
        log_psi_values = [np.log(data["Psi"][i]) for i in indices]
        omega_rec_pert = [x_perturbation(v) for v in omega_rec]
        omega_dir_pert = [x_perturbation(v) for v in omega_dir]
        omega_pert_lim = robust_limits(omega_rec_pert + omega_dir_pert, symmetric=True)
        log_rows = []
        for row, _idx in enumerate(indices):
            log_rows.append(
                {
                    "tau": tau_values[row],
                    "chi": chi_values[row],
                    "logPsi": log_psi_values[row],
                    "omega_rec_pert": omega_rec_pert[row],
                    "omega_dir_pert": omega_dir_pert[row],
                }
            )
        plot_panel_grid(
            out_dir / "khi_v2_log_transforms_vs_direct_perturbations.png",
            times,
            indices,
            log_rows,
            [
                ("tau", "tau = log Theta", "RdBu_r", robust_limits(tau_values, symmetric=True)),
                ("chi", "chi = log Xi", "RdBu_r", robust_limits(chi_values, symmetric=True)),
                ("logPsi", "log Psi", "RdBu_r", robust_limits(log_psi_values, symmetric=True)),
                ("omega_rec_pert", "omega' reconstructed", "RdBu_r", omega_pert_lim),
                ("omega_dir_pert", "omega' direct IVP", "RdBu_r", omega_pert_lim),
            ],
            lx,
            ly,
        )

        final = int(indices[-1])
        err_fields = [
            (data["rho_reconstructed"][final] - data["rho_direct"][final], "rho reconstructed - direct"),
            (data["ux_reconstructed"][final] - data["ux_direct"][final], "ux reconstructed - direct"),
            (data["uy_reconstructed"][final] - data["uy_direct"][final], "uy reconstructed - direct"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for ax, (field, title) in zip(axes, err_fields, strict=True):
            vmax = np.percentile(np.abs(field), 99.5)
            im = ax.imshow(
                field.T,
                origin="lower",
                extent=(0, lx, 0, ly),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            add_colorbar(fig, im, ax)
        fig.tight_layout()
        fig.savefig(out_dir / "khi_v2_final_errors.png", dpi=180)
        plt.close(fig)

        plot_error_history(out_dir / "khi_v2_error_history.png", h5)

    print(f"wrote KHI comparison plots to {out_dir}")


if __name__ == "__main__":
    main()
