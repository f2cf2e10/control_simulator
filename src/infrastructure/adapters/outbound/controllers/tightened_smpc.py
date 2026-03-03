from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import polytope as pc
import scipy.sparse as sp
import scipy.stats as st

from src.application.ports.outbound.controller import Controller
from src.application.ports.outbound.controller_models import (
    AncillaryControlLaw,
    ChanceConstraintSpec,
)
from src.domain.type import Matrix
from src.infrastructure.adapters.outbound.controllers.mpc_core import MpcCore
from src.infrastructure.adapters.outbound.utils import MatrixOps


class TightenedSmpc(Controller):
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
        vmax: float,
        chance_specs: Optional[Sequence[ChanceConstraintSpec]] = None,
        quantile_provider: Optional[Callable[[int, float], Matrix]] = None,
        velocity_state_indices: Optional[Tuple[int, int]] = None,
        ancillary_law: Optional[AncillaryControlLaw] = None,
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")
        self.N_tilde = int(N_tilde)
        self.K = K
        self.ancillary_law = ancillary_law or AncillaryControlLaw(
            transform=lambda v0, y_k: v0 - self.K @ y_k
        )

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

        Ccbf = np.vstack((Ccbf1.T, Ccbf2.T))
        bcbf = np.array([bcbf1, bcbf2])
        if quantile_provider is None:
            quantile_provider = self._build_irwinhall_quantile_provider(wmin, wmax)
        if chance_specs is None:
            chance_specs = self._build_default_chance_specs(
                Ccbf=Ccbf,
                bcbf=bcbf,
                gamma=gamma,
                vmax=vmax,
                n=n,
                velocity_state_indices=velocity_state_indices,
            )

        # Kept for API compatibility with existing constructor.
        _ = Sigma

        self._build_problem_offline(
            N=N,
            n=n,
            m=m,
            Q_list=Q_list,
            R_list=R_list,
            A_list=Acl_list,
            B_list=B_list,
            G=G,
            umin=umin,
            umax=umax,
            wmin=wmin,
            wmax=wmax,
            epsilon=epsilon,
            chance_specs=chance_specs,
            quantile_provider=quantile_provider,
        )

    def _build_problem_offline(
        self,
        N,
        n,
        m,
        Q_list,
        R_list,
        A_list,
        B_list,
        G,
        umin,
        umax,
        wmin,
        wmax,
        epsilon,
        chance_specs,
        quantile_provider,
    ):
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
        cons += self.build_input_tightening_constraints(
            A_list=A_list, K=self.K, G=G, umin=umin, umax=umax, wmin=wmin, wmax=wmax
        )
        cons += self.build_chance_constraints(
            specs=chance_specs,
            epsilon=epsilon,
            quantile_provider=quantile_provider,
        )
        self.problem_sparse = cp.Problem(obj, cons)

    def build_input_tightening_constraints(self, A_list, K, G, umin, umax, wmin, wmax):
        nx = self.N * self.n
        U = pc.box2poly(np.array([[umin, umax]] * self.m))
        W = pc.box2poly(np.array([[wmin, wmax]] * G.shape[1]))

        def matrix_times_polytope(A, P):
            if pc.is_empty(P):
                return P
            AP = (A @ pc.extreme(P).T).T
            all_zeros = (AP == 0).all(1)
            AP = np.delete(AP, all_zeros, axis=1)
            return pc.qhull(AP)

        GW = matrix_times_polytope(G, W)
        E = GW
        V_list = [U]
        for i in range(1, self.N):
            KE = matrix_times_polytope(K, E)
            V_list.append(U.diff(KE))
            AE = matrix_times_polytope(A_list[i], E)
            E = AE.union(GW)

        Au = sp.block_diag([Vi.A for Vi in V_list], format="csc")
        bu = np.vstack([Vi.b for Vi in V_list]).reshape(-1)
        Zu = sp.csc_matrix((Au.shape[0], nx))
        Gu = sp.hstack([Zu, Au], format="csc")
        return [Gu @ self.z <= bu]

    def build_chance_constraints(self, specs, epsilon, quantile_provider):
        constraints = []
        for spec in specs:
            mean = None
            if spec.mean_selector is not None:
                mean = spec.mean_selector(self.x0_param)
            constraints += self._build_chance_constraint(
                Cprev=spec.Cprev,
                Ccurr=spec.Ccurr,
                b=spec.b,
                quantile_provider=quantile_provider,
                epsilon=epsilon,
                mean=mean,
                step_start=spec.step_start,
                step_stop=spec.step_stop,
            )
        return constraints

    def _build_chance_constraint(
        self,
        Cprev,
        Ccurr,
        b,
        quantile_provider,
        epsilon,
        mean,
        step_start,
        step_stop,
    ):
        n = self.n
        N_eff = self.N - self.N_tilde
        nx = self.N * n
        nz = nx + self.N * self.m

        d = Cprev.shape[0]
        i_start = max(1, int(step_start))
        i_stop = int(step_stop) if step_stop is not None else N_eff
        i_stop = min(i_stop, N_eff)
        if i_stop <= i_start:
            return []

        step_ids = list(range(i_start, i_stop))
        steps = len(step_ids)
        M = sp.lil_matrix((steps * d, nz))
        rhs = np.zeros(steps * d)

        for row_i, i in enumerate(step_ids):
            r = row_i * d
            M[r:r + d, (i - 1) * n:i * n] = Cprev
            M[r:r + d, i * n:(i + 1) * n] = Ccurr
            q = np.asarray(quantile_provider(i, epsilon), dtype=float).reshape(-1)
            if q.size == 1:
                q = np.full(d, q.item())
            if q.size != d:
                raise ValueError(f"quantile length must be {d}, got {q.size}")
            rhs[r:r + d] = np.asarray(b, dtype=float).reshape(-1) - q

        M = M.tocsc()
        if mean is None:
            return [M @ self.z <= rhs]

        mean_len = int(np.prod(mean.shape))
        mean_vec = cp.reshape(mean, (mean_len,), order="F")
        if mean_len != d:
            if d % mean_len != 0:
                raise ValueError(f"mean length must divide d={d}, got {mean_len}")
            mean_vec = cp.hstack([mean_vec] * (d // mean_len))

        mean_stacked = cp.hstack([mean_vec] * steps)
        return [M @ self.z <= rhs + mean_stacked]

    def _build_irwinhall_quantile_provider(self, wmin, wmax):
        def provider(i, epsilon):
            return st.irwinhall(i, loc=i * wmin, scale=(wmax - wmin)).ppf(1.0 - epsilon)

        return provider

    def _build_default_chance_specs(
        self,
        Ccbf,
        bcbf,
        gamma,
        vmax,
        n,
        velocity_state_indices,
    ):
        if velocity_state_indices is None:
            mid = n // 2
            velocity_state_indices = (mid, mid + 1)
        i0, i1 = velocity_state_indices

        L1 = np.zeros((4, n))
        L1[0, [i0, i1]] = [-1, -1]
        L1[1, [i0, i1]] = [-1, 1]
        L1[2, [i0, i1]] = [1, -1]
        L1[3, [i0, i1]] = [1, 1]

        return [
            ChanceConstraintSpec(
                Cprev=(1 - gamma) * Ccbf,
                Ccurr=-Ccbf,
                b=gamma * bcbf,
                mean_selector=lambda x0: x0[0:2],
            ),
            ChanceConstraintSpec(
                Cprev=np.zeros((4, n)),
                Ccurr=L1,
                b=np.ones((4, 1)) * vmax,
                mean_selector=lambda x0: x0[2:4],
            ),
        ]

    def initialize(self) -> None:
        pass

    def compute(self, y_k):
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
        v0 = z[nx:nx + self.m].reshape(self.m, 1)
        y_col = np.asarray(y_k, dtype=float).reshape(self.n, 1)
        return self.ancillary_law(v0, y_col)
