from __future__ import annotations

from typing import Optional, Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from src.domain.type import Matrix
from src.application.ports.outbound.constraint import Constraint
from src.application.ports.outbound.controller import Controller
from src.application.ports.outbound.controller_models import AncillaryControlLaw
from src.infrastructure.adapters.outbound.controllers.mpc.mpc_core import MpcCore
from src.infrastructure.adapters.outbound.utils import MatrixOps


class TightenedTubeSmpc(Controller):
    def __init__(
        self,
        N: int,
        N_tilde: int,
        A: Matrix,
        B: Matrix,
        G: Matrix,
        Q: Matrix,
        R: Matrix,
        K: Matrix,
        Sigma: Matrix,
        epsilon: float,
        umin: float,
        umax: float,
        wmin: float,
        wmax: float,
        constraints: Optional[Sequence[Constraint]] = None,
        ancillary_law: Optional[AncillaryControlLaw] = None,
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")
        self.N_tilde = int(N_tilde)
        self.K = K
        self.G = G
        self.umin = umin
        self.umax = umax
        self.wmin = wmin
        self.wmax = wmax
        self.ancillary_law = ancillary_law 

        A_list = MatrixOps.to_list(A, self.N, "A")
        B_list = MatrixOps.to_list(B, self.N, "B")
        Q_list = MatrixOps.to_list(Q, self.N, "Q")
        R_list = MatrixOps.to_list(R, self.N, "R")
        Acl_list = [A_list[i] - B_list[i] @ K for i in range(self.N)]

        A0 = sp.csc_matrix(A_list[0])
        B0 = sp.csc_matrix(B_list[0])
        n = int(A0.shape[0])
        if A0.shape != (n, n):
            raise ValueError(f"A[0] must be (n,n), got {A0.shape}")
        if B0.shape[0] != n:
            raise ValueError(f"B[0] must have n rows, got {B0.shape}")
        m = int(B0.shape[1])
        self.n, self.m = n, m
        self.constraints = constraints
        self._solve_times: list[float] = []
        self.avg_solve_time: float = float("nan")

        self.z, self.x0_param, obj, cons = MpcCore.build_problem_params(
            N=N,
            n=n,
            m=m,
            Q_list=Q_list,
            R_list=R_list,
            A_list=A_list,
            B_list=B_list,
            Qn=None,
            x_min=None,
            x_max=None,
            u_min=None,
            u_max=None,
        )
        for constraint in self.constraints:
            cons += constraint.build(self.z, self.x0_param)

        self.problem_sparse = cp.Problem(obj, cons)

    def initialize(self) -> None:
        self._solve_times.clear()
        self.avg_solve_time = float("nan")

    def compute(self, y_k):
        self.x0_param.value = np.asarray(y_k, dtype=float).reshape(self.n)
        self.problem_sparse.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=False,
        )
        solve_time = getattr(self.problem_sparse, "_solve_time", None)
        if solve_time is not None:
            self._solve_times.append(float(solve_time))
            self.avg_solve_time = float(np.mean(self._solve_times))
        if self.problem_sparse.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC failed: {self.problem_sparse.status}")
        z = self.z.value
        if z is None:
            raise RuntimeError("Solver returned no solution.")
        nx = self.N * self.n
        v0 = z[nx:nx + self.m].reshape(self.m, 1)
        return self.ancillary_law(v0, y_k) if self.ancillary_law is not None else v0
