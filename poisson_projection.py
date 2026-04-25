"""Dedalus-native periodic zero-mean Poisson projection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import dedalus.public as d3
from mpi4py import MPI


Array = np.ndarray


@dataclass
class PoissonResult:
    field: object
    data: Array
    residual_l2: float


class ZeroMeanPoissonSolver:
    """Solve ``lap(phi) = source`` with ``integ(phi)=0`` on a periodic domain.

    The Lagrange multiplier ``tau`` removes the periodic Laplacian nullspace.
    This class is MPI-safe when used with Dedalus-distributed fields. Returned
    ``data`` is the local grid data on the calling rank.
    """

    def __init__(self, coords, dist, bases, name: str = "poisson"):
        self.coords = coords
        self.dist = dist
        self.bases = bases
        self.source = dist.Field(name=f"{name}_source", bases=bases)
        self.phi = dist.Field(name=f"{name}_phi", bases=bases)
        self.tau = dist.Field(name=f"{name}_tau")
        self.lap = lambda a: d3.Laplacian(a)
        self.integ = lambda a: d3.Integrate(a)
        problem = d3.LBVP([self.phi, self.tau], namespace=vars(self))
        problem.add_equation("lap(phi) + tau = source")
        problem.add_equation("integ(phi) = 0")
        self.solver = problem.build_solver()

    def solve(self, source) -> PoissonResult:
        self.source.change_scales(1)
        self.phi.change_scales(1)
        if hasattr(source, "change_scales"):
            source.change_scales(1)
            self.source["g"] = source["g"]
        else:
            self.source["g"] = np.asarray(source)
        self.solver.solve()
        self.phi.change_scales(1)
        data = np.asarray(self.phi["g"]).copy()
        residual = (self.lap(self.phi) - self.source).evaluate()
        residual.change_scales(1)
        residual_l2 = global_l2(self.dist, residual["g"])
        self.phi.change_scales(1)
        return PoissonResult(self.phi, data, residual_l2)


def global_l2(dist, local_data: Array) -> float:
    local_sum = float(np.sum(np.asarray(local_data) ** 2))
    local_count = int(np.asarray(local_data).size)
    total_sum = dist.comm.allreduce(local_sum, op=MPI.SUM)
    total_count = dist.comm.allreduce(local_count, op=MPI.SUM)
    return float(np.sqrt(total_sum / max(total_count, 1)))
