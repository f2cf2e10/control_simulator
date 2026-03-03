from __future__ import annotations

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


class MpcCore:
    """
    Reusable sparse linear MPC QP core.

    Decision vector:
        z = [x1, x2, ..., xN, u0, u1, ..., u_{N-1}]
    """

    def __init__(
        self,
        N,
        n,
        m,
        Q_list,
        R_list,
        A_list,
        B_list,
        Qn=None,
        x_min=None,
        x_max=None,
        u_min=None,
        u_max=None,
    ):
        self.N = int(N)
        self.n = int(n)
        self.m = int(m)

        self.z, self.x0_param, obj, cons = self.build_problem_params(
            N=self.N,
            n=self.n,
            m=self.m,
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
        self.problem_sparse = cp.Problem(obj, cons)

    @staticmethod
    def build_problem_params(
        N,
        n,
        m,
        Q_list,
        R_list,
        A_list,
        B_list,
        Qn=None,
        x_min=None,
        x_max=None,
        u_min=None,
        u_max=None,
    ):
        nx = N * n
        nu = N * m
        nz = nx + nu

        Q_blocks = [sp.csc_matrix(Qk) for Qk in Q_list]
        R_blocks = [sp.csc_matrix(Rk) for Rk in R_list]

        if len(Q_blocks) != N:
            raise ValueError("Q must expand to a list of length N.")
        if Qn is not None:
            Q_blocks[-1] = Qn

        H = sp.block_diag(Q_blocks + R_blocks, format="csc")

        Aeq = sp.lil_matrix((nx, nz))
        E = sp.lil_matrix((nx, n))

        for k in range(N):
            r = k * n
            Aeq[r:r + n, k * n:(k + 1) * n] = sp.eye(n, format="csc")

            Ak = sp.csc_matrix(A_list[k])
            Bk = sp.csc_matrix(B_list[k])

            if k == 0:
                E[r:r + n, :] = Ak
            else:
                Aeq[r:r + n, (k - 1) * n:k * n] = -Ak

            Aeq[r:r + n, nx + k * m:nx + (k + 1) * m] = -Bk

        Aeq = Aeq.tocsc()
        E = E.tocsc()

        x0_param = cp.Parameter(n, name="x0")
        z = cp.Variable(nz, name="z")

        G_blocks = []
        h_blocks = []

        if x_min is not None:
            Gx = sp.eye(nx, format="csc")
            Znu = sp.csc_matrix((nx, nu))
            G_x_pos = sp.hstack([Gx, Znu], format="csc")
            G_x_neg = sp.hstack([-Gx, Znu], format="csc")
            G_blocks.append(G_x_pos)
            h_blocks.append(np.tile(x_max, N))
            G_blocks.append(G_x_neg)
            h_blocks.append(-np.tile(x_min, N))

        if u_min is not None:
            Gu = sp.eye(nu, format="csc")
            Znxu = sp.csc_matrix((nu, nx))
            G_u_pos = sp.hstack([Znxu, Gu], format="csc")
            G_u_neg = sp.hstack([Znxu, -Gu], format="csc")
            G_blocks.append(G_u_pos)
            h_blocks.append(np.tile(u_max, N))
            G_blocks.append(G_u_neg)
            h_blocks.append(-np.tile(u_min, N))

        G = sp.vstack(G_blocks, format="csc") if G_blocks else None
        h_const = np.concatenate(h_blocks) if h_blocks else None

        obj = cp.Minimize(0.5 * cp.quad_form(z, H))
        cons = [Aeq @ z == E @ x0_param]
        if G is not None:
            cons += [G @ z <= h_const]

        return z, x0_param, obj, cons

    def compute_first_input(self, y_k):
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
        u0 = z[nx:nx + self.m]
        return u0.reshape(self.m, 1)
