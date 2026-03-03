from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from src.infrastructure.adapters.outbound.utils import MatrixOps
from src.infrastructure.adapters.outbound.controllers.mpc.mpc_core import MpcCore
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

    def initialize(self) -> None:
        pass

    def compute(self, y_k: Matrix) -> Matrix:
        return self._core.compute_first_input(y_k)
