from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
import polytope as pc

from src.domain.type import Matrix, MatrixOrSeq
from src.infrastructure.adapters.outbound.utils import MatrixOps
from src.infrastructure.adapters.outbound.controllers.nominal_mpc_sparse import NominalMpc


class TightenedSMpc(NominalMpc):
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
        A: Matrix,
        B: Matrix,
        G: Matrix,
        Q: Matrix,
        R: Matrix,
        K: Matrix,
        Sigma: Matrix,
        Ccbf1: Matrix,
        bcbf1: Matrix,
        Ccbf2: Matrix,
        bcbf2: Matrix,
        gamma: float,
        epsilon: float,
        umin: float,
        umax: float,
        wmin: float,
        wmax: float,
        vmax: float
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")

        # Convert to per-step lists (length N)
        A_list = MatrixOps.to_list(A, self.N, "A")
        B_list = MatrixOps.to_list(B, self.N, "B")
        Q_list = MatrixOps.to_list(Q, self.N, "Q")
        R_list = MatrixOps.to_list(R, self.N, "R")
        Acl = A + B @ K
        GSG = G @ Sigma @ G.T
        S_list = [GSG]
        for i in range(N):
            S_list.append(Acl @ S_list[i] @ Acl.T + GSG)

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
        Ccbf = np.vstack((Ccbf1.T, Ccbf2.T))  # shape (2, n)
        bcbf = np.array([bcbf1, bcbf2])
        self._build_problem_offline(
            N, n, m, Q_list, R_list, K, A_list, B_list, G, Ccbf, bcbf,
            gamma, epsilon, umin, umax, wmin, wmax, vmax)

    def _build_problem_offline(self, N, n, m, Q_list, R_list, K, A_list,
                               B_list, G, Ccbf, bcbf, gamma, epsilon, umin,
                               umax, wmin, wmax, vmax):
        Acl_list = [A_list[i] + B_list[i] @ K for i in range(self.N)]
        obj, cons = self._build_problem_params(
            N, n, m, Q_list, R_list, Acl_list, B_list, None, None, None, None, None)
        cons += self._build_input_tightening(Acl_list,
                                             K, G, umin, umax, wmin, wmax)

        def prob(i): return st.irwinhall(i, loc=i * wmin, scale=(wmax - wmin))
        # P{C'*x_i + b >= (1-gamma)(C'*x_{i-1}+b) | x_{i-1}} >= 1-epsilon
        cons += self._build_chance_constraint((1-gamma)
                                              * Ccbf, -Ccbf, gamma * bcbf, prob, epsilon)
        # P{| vx_i | + | vy_i | <= vmax | x_{i-1} } >= 1-epsilon
        mid = n / 2
        L1 = np.zeros((4, n))
        L1[0, [mid, mid + 1]] = [-1, -1]
        L1[1, [mid, mid + 1]] = [-1, 1]
        L1[2, [mid, mid + 1]] = [1, -1]
        L1[3, [mid, mid + 1]] = [1, 1]
        cons += self._build_chance_constraint(
            np.zeros([4, n]), L1, np.ones([4, 1])*vmax, prob, epsilon)
        # ---------------------------
        # Build CVXPY problem ONCE (offline)
        # ---------------------------
        self.problem_sparse = cp.Problem(obj, cons)

    def _build_input_tightening(self, A_list, K, G, umin, umax, wmin, wmax):
        # ---------------------------
        # Input constraint tightening
        # ---------------------------
        U = pc.box2poly(np.array([[umin, umax]]*self.m))
        W = pc.box2poly(np.array([[wmin, wmax]]*self.m))
        V_list = [U]
        GW = G@W
        E = pc.Polytope(GW, np.zeros(self.m))
        for i in range(1, self.N):
            KE = pc.qhull((K @ pc.extreme(E).T).T)
            V_list.append(U.diff(KE))
            AE = pc.qhull((A_list[i] @ pc.extreme(E).T).T)
            E = AE.union(GW)

        zero_blocks = [sp.csc_matrix(self.m)] * self.N
        V = sp.block_diag(zero_blocks + [V.A for V in V_list], format="csc")
        v_bound = np.hstack([V.b for V in V_list])
        return [V @ self.z <= v_bound]

    def _build_chance_constraint(self, Cprev, Ccurr, b, quantile, epsilon):
        # ---------------------------
        # State constraint tightening
        # ---------------------------
        # AND velocity L1 constraints on states (3, 4), for i = 1..p:
        n = self.n
        m = self.m
        nx = self.N * n
        nz = nx + self.N * m

        d = Cprev.shape[0]
        M = sp.lil_matrix(((self.N - 1) * d, nz))

        for i in range(1, self.N):
            r = (i - 1) * d
            M[r:r+d, (i - 1) * n:i * n] = Cprev
            M[r:r+d, i * n:(i + 1) * n] = Ccurr

        M = M.tocsc()
        rhs = np.zeros(d * (self.N - 1))
        for i in range(1, self.N):
            q = quantile(i).ppf(1.0 - epsilon)
            # reshape because we use a column vector
            rhs[d*(i-1):d*i] = (b - q).reshape(-1)
        return [M @ self.z <= rhs]
