from __future__ import annotations

import itertools
import numpy as np
import scipy.sparse as sp

from src.application.ports.outbound.constraint import Constraint
from src.domain.type import Matrix
from src.infrastructure.adapters.outbound.utils import MatrixOps


class InputTighteningConstraint(Constraint):
    def __init__(
        self,
        N: int,
        A: Matrix,
        B: Matrix,
        K: Matrix,
        G: Matrix,
        umin: float,
        umax: float,
        wmin: float,
        wmax: float,
    ):
        self.N = int(N)
        A_list = MatrixOps.to_list(A, self.N, "A")
        B_list = MatrixOps.to_list(B, self.N, "B")
        self.G_list = MatrixOps.to_list(G, self.N, "G")
        self.Acl_list = [A_list[i] - B_list[i] @ K for i in range(self.N)]
        self.n = int(np.asarray(A_list[0]).shape[0])
        self.m = int(np.asarray(B_list[0]).shape[1])
        self.nw = int(np.asarray(self.G_list[0]).shape[1])
        self.K = K
        self.umin_vec = self._to_dim_vector(umin, self.m, "umin")
        self.umax_vec = self._to_dim_vector(umax, self.m, "umax")
        self.wmin_vec = self._to_dim_vector(wmin, self.nw, "wmin")
        self.wmax_vec = self._to_dim_vector(wmax, self.nw, "wmax")

        if np.any(self.umin_vec > self.umax_vec):
            raise ValueError("Each element of umin must be <= corresponding umax.")
        if np.any(self.wmin_vec > self.wmax_vec):
            raise ValueError("Each element of wmin must be <= corresponding wmax.")

    @staticmethod
    def _to_dim_vector(value: Matrix | float, dim: int, name: str) -> np.ndarray:
        vec = np.asarray(value, dtype=float).reshape(-1)
        if vec.size == 1:
            return np.full(dim, vec.item(), dtype=float)
        if vec.size != dim:
            raise ValueError(f"{name} must be scalar or length {dim}, got {vec.size}.")
        return vec

    @staticmethod
    def _box_vertices(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        dim = int(lb.size)
        bits = np.array(list(itertools.product([0.0, 1.0], repeat=dim)), dtype=float)
        return lb + bits * (ub - lb)

    def _error_vertices_by_stage(self) -> list[np.ndarray]:
        """
        Mirrors robust_setdiff.m:
            E_1 = {0}
            E_{k+1} = Acl_k E_k ⊕ G_k W
        """
        w_vertices = self._box_vertices(self.wmin_vec, self.wmax_vec)  # (2^nw, nw)
        e_k = np.zeros((1, self.n), dtype=float)  # E_1
        stages = [e_k]

        for k in range(self.N):
            acl_k = np.asarray(self.Acl_list[k], dtype=float)
            g_k = np.asarray(self.G_list[k], dtype=float)

            ae = e_k @ acl_k.T
            gw = w_vertices @ g_k.T
            # Candidate vertices of Minkowski sum conv(ae) ⊕ conv(gw)
            e_next = (ae[:, None, :] + gw[None, :, :]).reshape(-1, self.n)
            # Cheap dedup to keep growth manageable.
            e_next = np.unique(np.round(e_next, decimals=12), axis=0)
            stages.append(e_next)
            e_k = e_next

        return stages

    def build(self, z, x0_param):
        _ = x0_param
        nx = self.N * self.n
        # Box constraints U = {u | Au u <= bu}
        # with Au = [I; -I], bu = [umax; -umin].
        au_stage = np.vstack((np.eye(self.m), -np.eye(self.m)))
        bu_stage = np.concatenate((self.umax_vec, -self.umin_vec))

        # Tightening term h_{-K E_k}(a_j) = max_{e in E_k} a_j (-K e)
        support_dirs = -au_stage @ np.asarray(self.K, dtype=float)  # (2m, n)
        e_stages = self._error_vertices_by_stage()

        au_blocks = []
        bu_blocks = []
        for k in range(self.N):
            e_k = e_stages[k]
            support = np.max(e_k @ support_dirs.T, axis=0)  # (2m,)
            au_blocks.append(au_stage)
            bu_blocks.append(bu_stage - support)

        Au = sp.block_diag(au_blocks, format="csc")
        bu = np.concatenate(bu_blocks)
        Zu = sp.csc_matrix((Au.shape[0], nx))
        Gu = sp.hstack([Zu, Au], format="csc")
        return [Gu @ z <= bu]
