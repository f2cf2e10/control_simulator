from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from src.infrastructure.adapters.outbound.utils import MatrixOps
from src.infrastructure.adapters.outbound.controllers.mpc_core import MpcCore
from src.application.ports.outbound.controller import Controller
from src.domain.type import Matrix, MatrixOrSeq


class NominalMpc(Controller):
    """
    Linear MPC (sparse QP), offline matrices + single online parameter (x0).

    Decision vector:
        z = [x1, x2, ..., xN, u0, u1, ..., u_{N-1}]

    Dynamics:
        x_{k+1} = A_k x_k + B_k u_k
    encoded as:
        Aeq z = E x0
    where Aeq and E are constant; only x0 updates online.
    """

    def __init__(
        self,
        N: int,
        A: MatrixOrSeq,
        B: MatrixOrSeq,
        Q: MatrixOrSeq,
        R: MatrixOrSeq,
        Qn: Optional[Matrix] = None,  # terminal cost
        x_min: Optional[np.ndarray] = None,
        x_max: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")

        # Convert to per-step lists (length N)
        A_list = MatrixOps.to_list(A, self.N, "A")
        B_list = MatrixOps.to_list(B, self.N, "B")
        Q_list = MatrixOps.to_list(Q, self.N, "Q")
        R_list = MatrixOps.to_list(R, self.N, "R")

        # Dimensions from step 0
        A0 = sp.csc_matrix(A_list[0])
        B0 = sp.csc_matrix(B_list[0])

        n = int(A0.shape[0])
        if A0.shape != (n, n):
            raise ValueError(f"A[0] must be (n,n), got {A0.shape}")
        if B0.shape[0] != n:
            raise ValueError(f"B[0] must have n rows, got {B0.shape}")
        m = int(B0.shape[1])
        self.n, self.m = n, m

        if (x_min is None) != (x_max is None):
            raise ValueError("Provide both x_min and x_max, or neither.")
        if (u_min is None) != (u_max is None):
            raise ValueError("Provide both u_min and u_max, or neither.")

        Qn = sp.csc_matrix(Qn) if Qn is not None else None

        self._core = MpcCore(
            N=N,
            n=n,
            m=m,
            Q_list=Q_list,
            R_list=R_list,
            A_list=A_list,
            B_list=B_list,
            Qn=Qn,
            x_min=x_min,
            x_max=x_max,
            u_min=u_min,
            u_max=u_max,
        )
        self.problem_sparse = self._core.problem_sparse
        self.z = self._core.z
        self.x0_param = self._core.x0_param

    def _build_problem_offline(self, N, n, m, Q_list, R_list, A_list, B_list, Qn, x_min, x_max, u_min, u_max):
        # Compatibility path for subclasses that still override/compose via this method.
        obj, cons = self._build_problem_params(
            N, n, m, Q_list, R_list, A_list, B_list, Qn, x_min, x_max, u_min, u_max)
        self.problem_sparse = cp.Problem(obj, cons)

    def _build_problem_params(self, N, n, m, Q_list, R_list, A_list, B_list, Qn, x_min, x_max, u_min, u_max) -> None:
        # Compatibility path for subclasses: creates self.z and self.x0_param.
        self.z, self.x0_param, obj, cons = MpcCore.build_problem_params(
            N=N,
            n=n,
            m=m,
            Q_list=Q_list,
            R_list=R_list,
            A_list=A_list,
            B_list=B_list,
            Qn=Qn,
            x_min=x_min,
            x_max=x_max,
            u_min=u_min,
            u_max=u_max,
        )
        return obj, cons

    def initialize(self) -> None:
        pass

    def compute(self, y_k: Matrix) -> Matrix:
        if hasattr(self, "_core"):
            return self._core.compute_first_input(y_k)

        # Compatibility path (e.g., subclasses assembling their own problem).
        self.x0_param.value = np.asarray(y_k, dtype=float).reshape(self.n)
        self.problem_sparse.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=True,
        )
        if self.problem_sparse.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC failed: {self.problem_sparse.status}")
        z = self.z.value
        if z is None:
            raise RuntimeError("Solver returned no solution.")
        nx = self.N * self.n
        u0 = z[nx: nx + self.m]
        return u0.reshape(self.m, 1)
