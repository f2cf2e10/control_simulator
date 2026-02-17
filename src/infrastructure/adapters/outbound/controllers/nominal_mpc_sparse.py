from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from src.infrastructure.adapters.outbound.utils import MatrixOps
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
        self.A_list = MatrixOps.to_list(A, self.N, "A")
        self.B_list = MatrixOps.to_list(B, self.N, "B")
        self.Q_list = MatrixOps.to_list(Q, self.N, "Q")
        self.R_list = MatrixOps.to_list(R, self.N, "R")

        # Dimensions from step 0
        A0 = sp.csc_matrix(self.A_list[0])
        B0 = sp.csc_matrix(self.B_list[0])

        n = int(A0.shape[0])
        if A0.shape != (n, n):
            raise ValueError(f"A[0] must be (n,n), got {A0.shape}")
        if B0.shape[0] != n:
            raise ValueError(f"B[0] must have n rows, got {B0.shape}")

        m = int(B0.shape[1])
        self.n, self.m = n, m

        # Store bounds (validate shapes)
        self.x_min = None if x_min is None else np.asarray(
            x_min, dtype=float).reshape(self.n)
        self.x_max = None if x_max is None else np.asarray(
            x_max, dtype=float).reshape(self.n)
        self.u_min = None if u_min is None else np.asarray(
            u_min, dtype=float).reshape(self.m)
        self.u_max = None if u_max is None else np.asarray(
            u_max, dtype=float).reshape(self.m)

        if (self.x_min is None) != (self.x_max is None):
            raise ValueError("Provide both x_min and x_max, or neither.")
        if (self.u_min is None) != (self.u_max is None):
            raise ValueError("Provide both u_min and u_max, or neither.")

        self.Qn = sp.csc_matrix(Qn) if Qn is not None else None

        self._build_problem_offline()

    def _build_problem_offline(self) -> None:
        N, n, m = self.N, self.n, self.m

        nx = N * n
        nu = N * m
        nz = nx + nu

        # ---------------------------
        # OFFLINE Hessian H (cost)
        # Cost on x1..x_{N-1} uses Q_list[0..N-2], terminal uses Qn, controls use R_list[0..N-1]
        # If you already pass Q_list length N and want x1..xN weighted by Q_list, keep that,
        # but most MPC uses terminal Qn for xN.
        # ---------------------------
        Q_blocks = [sp.csc_matrix(Qk) for Qk in self.Q_list]
        R_blocks = [sp.csc_matrix(Rk) for Rk in self.R_list]

        # Replace last state weight with Qn (terminal)
        if len(Q_blocks) != N:
            raise ValueError("Q must expand to a list of length N.")
        if self.Qn is not None:
            Q_blocks[-1] = self.Qn

        self.H = sp.block_diag(Q_blocks + R_blocks, format="csc")

        # ---------------------------
        # OFFLINE equality constraints: Aeq z = E x0
        # Aeq shape: (N*n) x (N*n + N*m)
        # Rows correspond to k=0..N-1:
        #   x1 - B0 u0              = A0 x0
        #   x2 - A1 x1 - B1 u1      = 0
        #   ...
        #   xN - A_{N-1} x_{N-1} - B_{N-1} u_{N-1} = 0
        # ---------------------------
        Aeq = sp.lil_matrix((nx, nz))
        E = sp.lil_matrix((nx, n))  # multiplies x0

        for k in range(N):
            r = k * n

            # + x_{k+1} is always a decision state, stored at block k in [x1..xN]
            Aeq[r: r + n, k * n: (k + 1) * n] = sp.eye(n, format="csc")

            Ak = sp.csc_matrix(self.A_list[k])
            Bk = sp.csc_matrix(self.B_list[k])

            if k == 0:
                # RHS depends on x0:  x1 - B0 u0 = A0 x0
                E[r: r + n, :] = Ak
            else:
                # -A_k x_k (note x_k is decision state at block (k-1))
                Aeq[r: r + n, (k - 1) * n: k * n] = -Ak

            # -B_k u_k
            Aeq[r: r + n, nx + k * m: nx + (k + 1) * m] = -Bk

        self.Aeq = Aeq.tocsc()
        self.E = E.tocsc()

        # Only ONLINE piece:
        self.x0_param = cp.Parameter(n, name="x0")

        # ---------------------------
        # OFFLINE inequality constraints (constant bounds)
        # ---------------------------
        G_blocks = []
        h_blocks = []

        if self.x_min is not None:
            # x1..xN bounds
            Gx = sp.eye(nx, format="csc")

            G_blocks.append(Gx)
            h_blocks.append(np.tile(self.x_max, N))

            G_blocks.append(-Gx)
            h_blocks.append(-np.tile(self.x_min, N))

        if self.u_min is not None:
            # u0..u_{N-1} bounds; embed into full z dimension
            Gu = sp.eye(nu, format="csc")
            Znx = sp.csc_matrix((nx, nx))  # zeros for x-part

            G_u_pos = sp.block_diag([Znx, Gu], format="csc")
            G_u_neg = sp.block_diag([Znx, -Gu], format="csc")

            G_blocks.append(G_u_pos)
            h_blocks.append(np.tile(self.u_max, N))

            G_blocks.append(G_u_neg)
            h_blocks.append(-np.tile(self.u_min, N))

        self.G = sp.vstack(G_blocks, format="csc") if G_blocks else None
        self.h_const = np.concatenate(
            h_blocks) if h_blocks else None  # numeric constant

        # ---------------------------
        # Build CVXPY problem ONCE (offline)
        # ---------------------------
        self.z = cp.Variable(nz, name="z")

        obj = cp.Minimize(0.5 * cp.quad_form(self.z, self.H))

        cons = [self.Aeq @ self.z == self.E @ self.x0_param]

        if self.G is not None:
            # constant bounds, no online update
            cons += [self.G @ self.z <= self.h_const]

        self.problem_sparse = cp.Problem(obj, cons)

    def initialize(self) -> None:
        pass

    def compute(self, y_k: Matrix) -> Matrix:
        # ---- ONLY ONLINE OPERATION ----
        self.x0_param.value = np.asarray(y_k, dtype=float).reshape(self.n)

        self.problem_sparse.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=True,
        )

        if self.problem_sparse.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Sparse MPC failed: {self.problem_sparse.status}")

        z = self.z.value
        if z is None:
            raise RuntimeError("Solver returned no solution.")

        # u0 is the first control block in the control section
        nx = self.N * self.n
        u0 = z[nx: nx + self.m]
        return u0.reshape(self.m, 1)
