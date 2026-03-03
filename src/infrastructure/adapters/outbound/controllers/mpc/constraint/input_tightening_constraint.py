from __future__ import annotations

import numpy as np
import polytope as pc
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
        self.A_list = [A_list[i] - B_list[i] @ K for i in range(self.N)]
        self.n = int(np.asarray(A_list[0]).shape[0])
        self.m = int(np.asarray(B_list[0]).shape[1])
        self.K = K
        self.G = G
        self.umin = umin
        self.umax = umax
        self.wmin = wmin
        self.wmax = wmax

    def build(self, z, x0_param):
        _ = x0_param
        nx = self.N * self.n
        U = pc.box2poly(np.array([[self.umin, self.umax]] * self.m))
        W = pc.box2poly(np.array([[self.wmin, self.wmax]] * self.G.shape[1]))

        def matrix_times_polytope(A, P):
            if pc.is_empty(P):
                return P
            AP = (A @ pc.extreme(P).T).T
            all_zeros = (AP == 0).all(1)
            AP = np.delete(AP, all_zeros, axis=1)
            return pc.qhull(AP)

        GW = matrix_times_polytope(self.G, W)
        E = GW
        V_list = [U]
        for i in range(1, self.N):
            KE = matrix_times_polytope(self.K, E)
            V_list.append(U.diff(KE))
            AE = matrix_times_polytope(self.A_list[i], E)
            E = AE.union(GW)

        Au = sp.block_diag([Vi.A for Vi in V_list], format="csc")
        bu = np.vstack([Vi.b for Vi in V_list]).reshape(-1)
        Zu = sp.csc_matrix((Au.shape[0], nx))
        Gu = sp.hstack([Zu, Au], format="csc")
        return [Gu @ z <= bu]
