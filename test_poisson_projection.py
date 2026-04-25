"""Smoke test the Dedalus zero-mean periodic Poisson helper."""

from __future__ import annotations

import argparse

import numpy as np
import dedalus.public as d3

from poisson_projection import ZeroMeanPoissonSolver, global_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=args.nx, bounds=(0, 2 * np.pi), dealias=1)
    ybasis = d3.RealFourier(coords["y"], size=args.ny, bounds=(0, 2 * np.pi), dealias=1)
    x, y = dist.local_grids(xbasis, ybasis)
    source = dist.Field(name="source", bases=(xbasis, ybasis))

    # The exact solution of lap(phi)=-2 sin(x) cos(y) is sin(x) cos(y),
    # with zero mean on the periodic domain.
    expected = np.sin(x) * np.cos(y)
    source["g"] = -2 * expected
    solver = ZeroMeanPoissonSolver(coords, dist, (xbasis, ybasis), name="test")
    result = solver.solve(source)
    err = global_l2(dist, result.data - expected)
    if dist.comm.rank == 0:
        print(f"poisson_l2_error={err:.3e} residual_l2={result.residual_l2:.3e}")


if __name__ == "__main__":
    main()
