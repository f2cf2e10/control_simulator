from typing import List, Tuple
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from src.infrastructure.adapters.outbound.controllers.lqg.lqg import Lqg
from src.domain.type import Matrix
from src.infrastructure.adapters.outbound.utils import MatrixOps


def num_lower_triangular_elements(n, p):
    if n < p:
        return n * (n + 1) // 2
    else:
        return p * (n - p + 1) + p * (p + 1) // 2


def cumulative_product(A, s, t):
    n = A[0].shape[0]  # assuming A is a list of square matrices
    if s == t:
        return np.eye(n)
    else:
        result = np.eye(n)
        for k in range(s, t):
            result = A[k][:, :] @ result
        return result


def full_sparsity(rows: int, cols: int):
    r = np.repeat(np.arange(rows, dtype=int), cols)
    c = np.tile(np.arange(cols, dtype=int), rows)
    return r, c


def block_diag_sparsity(num_blocks: int, block_rows: int, block_cols: int):
    rows = []
    cols = []
    for b in range(num_blocks):
        row_idx = np.repeat(np.arange(block_rows, dtype=int) + b * block_rows, block_cols)
        col_idx = np.tile(np.arange(block_cols, dtype=int), block_rows) + b * block_cols
        rows.append(row_idx)
        cols.append(col_idx)
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(rows), np.concatenate(cols)


def strict_upper_block_sparsity(num_blocks: int, block_rows: int, block_cols: int):
    rows = []
    cols = []
    for t in range(num_blocks):
        for s in range(t + 1, num_blocks):
            row_idx = np.repeat(np.arange(block_rows, dtype=int) + t * block_rows, block_cols)
            col_idx = np.tile(np.arange(block_cols, dtype=int), block_rows) + s * block_cols
            rows.append(row_idx)
            cols.append(col_idx)
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(rows), np.concatenate(cols)


def has_nonzero_pattern(sparsity):
    return sparsity[0].size > 0


class LqgDro(Lqg):
    def __init__(
        self,
        N: int,
        A: Matrix,
        B: Matrix,
        C: Matrix,
        Q: Matrix,
        R: Matrix,
        Qn: Matrix,
        Sigma: Matrix,
        Gamma: Matrix,
        x0: Matrix,
        P0: Matrix,
        zeta: float,
        use_y0_update: bool = True
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")

        # Always store as lists (control horizon N; measurements 0..N)
        self.A_list = MatrixOps.to_list(A, self.N, "A")
        self.B_list = MatrixOps.to_list(B, self.N, "B")
        self.Q_list = MatrixOps.to_list(Q, self.N, "Q")
        self.R_list = MatrixOps.to_list(R, self.N, "R")
        self.Sigma_list = MatrixOps.to_list(Sigma, self.N, "Sigma")

        self.C_list = MatrixOps.to_list(C, self.N + 1, "C")
        self.Gamma_list = MatrixOps.to_list(Gamma, self.N + 1, "Gamma")

        self.Qn = np.asarray(Qn, dtype=float)
        self.x0 = MatrixOps.to_col(x0)
        self.P0 = np.asarray(P0, dtype=float)
        self.zeta = zeta
        self.use_y0_update = bool(use_y0_update)

        # Dim checks
        A0, B0, C0 = self.A_list[0], self.B_list[0], self.C_list[0]
        n = int(A0.shape[0])
        if A0.shape != (n, n):
            raise ValueError(f"A[0] must be (n,n), got {A0.shape}")
        if B0.shape[0] != n:
            raise ValueError(f"B[0] must have n rows, got {B0.shape}")
        if C0.shape[1] != n:
            raise ValueError(f"C[0] must have n cols, got {C0.shape}")
        if self.P0.shape != (n, n):
            raise ValueError(f"P0 must be (n,n), got {self.P0.shape}")
        if self.Qn.shape != (n, n):
            raise ValueError(f"Qn must be (n,n), got {self.Qn.shape}")

        # Precompute gains
        self.K_list, self.L_list, self.S_list, self.P_post_list, self.Sigma_dro_list, self.Gamma_dro_list = self._design_lqg_dro()

    def _design_lqg_dro(self) -> Tuple[List[Matrix], List[Matrix], List[Matrix], List[Matrix]]:
        """
        Returns:
          K_list: length N, (m,n)
          L_list: length N+1, (n,p)  gain for y_k
          S_list: length N+1, (n,n)
          P_post_list: length N+1, (n,n) posterior cov after incorporating y_k
        """

        N = self.N
        A_list, B_list, C_list = self.A_list, self.B_list, self.C_list
        Q_list, R_list = self.Q_list, self.R_list
        Sigma_dro_0, Sigma_dro_list, Gamma_dro_list = self._design_dro_noise()
        
        n = int(A_list[0].shape[0])
        m = int(B_list[0].shape[1])
        p = int(C_list[0].shape[0])

        # ---- LQR backward Riccati ----
        S_list: List[Matrix] = [
            np.zeros((n, n), dtype=float) for _ in range(N + 1)]
        K_list: List[Matrix] = [
            np.zeros((m, n), dtype=float) for _ in range(N)]
        S_list[N] = self.Qn.copy()

        for k in range(N - 1, -1, -1):
            Ak, Bk = A_list[k], B_list[k]
            Qk, Rk = Q_list[k], R_list[k]
            Sk1 = S_list[k + 1]

            Bbar = Bk.T @ Sk1 @ Bk + Rk
            Kk = np.linalg.solve(Bbar, Bk.T @ Sk1 @ Ak)  # (m,n)
            K_list[k] = Kk

            middle = Sk1 - Sk1 @ Bk @ np.linalg.solve(Bbar, Bk.T @ Sk1)
            S_list[k] = Ak.T @ middle @ Ak + Qk

        # ---- Kalman forward (L_k for y_k) ----
        P_list: List[Matrix] = [np.zeros((n, p), dtype=float)
                                for _ in range(N + 1)]
        L_list: List[Matrix] = [np.zeros((n, p)) for _ in range(N+1)]
        P_list[0] = Sigma_dro_0 

        for k in range(N):
            Ck = C_list[k]
            Gammak = Gamma_dro_list[k]
            Pk = P_list[k]

            Cbar = Ck @ Pk @ Ck.T + Gammak
            Lk = Pk @ Ck.T @ np.linalg.inv(Cbar)
            L_list[k] = Lk

            middle = Pk - Pk @ Ck.T @ np.linalg.solve(Cbar, Ck.dot(Pk))
            if k < N:
                Sigmak = Sigma_dro_list[k]
                Ak = A_list[k]
                P_list[k + 1] = Ak @ middle @ Ak.T + Sigmak

        return K_list, L_list, S_list, P_list, Sigma_dro_list, Gamma_dro_list

    def _design_dro_noise(self):
        # Optimized version using sparse matrices 
        # original naive source: https://github.com/RAO-EPFL/DR-Control/
        A = self.A_list
        B = self.B_list
        R = self.R_list
        C = self.C_list
        Q = self.Q_list
        zeta = self.zeta
        xhat0 = self.P0
        Gamma = self.Gamma_list
        Sigma = self.Sigma_list
        N = self.N
        n = A[0].shape[0]
        m = R[0].shape[0]
        p = Gamma[0].shape[0]

        #### Creating Block Matrices for SDP ####
        R_block = np.zeros([N, N, m, m])
        C_block = np.zeros([N, N + 1, p, n])
        for t in range(N):
            R_block[t, t] = R[t][:, :]
            C_block[t, t] = C[t][:, :]
        Q_block = np.zeros([n * (N + 1), n * (N + 1)])
        for t in range(N):
            Q_block[t * n: t * n + n, t * n: t * n + n] = Q[t][:, :]

        R_block = np.reshape(R_block.transpose(0, 2, 1, 3), (m * N, m * N))
        # Q_block = np.reshape(Q_block.transpose(0, 2, 1, 3), (n * (T + 1), n * (T + 1)))
        C_block = np.reshape(C_block.transpose(
            0, 2, 1, 3), (p * N, n * (N + 1)))

        # initialize H and G as zero matrices
        G = np.zeros((n * (N + 1), n * (N + 1)))
        H = np.zeros((n * (N + 1), m * N))
        for t in range(N + 1):
            for s in range(t + 1):
                # print(GG[t * n : t * n + n, s * n : s * n + n])
                G[t * n: t * n + n, s * n: s * n +
                    n] = cumulative_product(A, s, t)
                if t != s:
                    H[t * n: t * n + n, s * m: s * m + m] = (
                        cumulative_product(A, s + 1, t) @ B[s][:, :]
                    )
        D = np.matmul(C_block, G)
        inv_cons = np.linalg.inv(R_block + H.T @ Q_block @ H)

        ### OPTIMIZATION MODEL ###
        w_var_sparsity = block_diag_sparsity(N + 1, n, n)
        v_var_sparsity = block_diag_sparsity(N, p, p)
        m_var_sparsity = strict_upper_block_sparsity(N, m, p)
        sep_w_sparsity = full_sparsity(n, n)
        sep_v_sparsity = full_sparsity(p, p)

        E = cp.Variable((m * N, m * N), symmetric=True)
        E_x0 = cp.Variable((n, n), symmetric=True)
        W_var = cp.Variable((n * (N + 1), n * (N + 1)), sparsity=w_var_sparsity)
        V_var = cp.Variable((p * N, p * N), sparsity=v_var_sparsity)
        E_w = []
        E_v = []
        W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
        V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
        for t in range(N):
            E_w.append(cp.Variable((n, n), symmetric=True))
            E_v.append(cp.Variable((p, p), symmetric=True))
            W_var_sep.append(cp.Variable((n, n), sparsity=sep_w_sparsity))
            V_var_sep.append(cp.Variable((p, p), sparsity=sep_v_sparsity))
        W_var_sep.append(cp.Variable((n, n), sparsity=sep_w_sparsity))
        if has_nonzero_pattern(m_var_sparsity):
            M_var = cp.Variable((m * N, p * N), sparsity=m_var_sparsity)
        else:
            M_var = cp.Constant(np.zeros((m * N, p * N)))

        cons = []
        for t in range(N):
            for s in range(t + 1):
                cons.append(M_var[t * m: t * m + m, p * s: p * s + p] == 0)

        for t in range(N + 1):
            cons.append(W_var[n * t: n * t + n, n *
                        t: n * t + n] == W_var_sep[t])
            cons.append(W_var_sep[t] == W_var_sep[t].T)
            cons.append(W_var_sep[t] >> 0)

        for t in range(N):
            cons.append(V_var[p * t: p * t + p, p *
                        t: p * t + p] == V_var_sep[t])
            cons.append(V_var_sep[t] == V_var_sep[t].T)
            cons.append(V_var_sep[t] >> 0)
            cons.append(E_v[t] >> 0)
            cons.append(E_w[t] >> 0)

        cons.append(E >> 0)
        cons.append(E_x0 >> 0)

        cons.append(cp.trace(W_var_sep[0] + xhat0 - 2 * E_x0) <= zeta**2)
        cons.append(W_var_sep[0] >> np.min(
            np.linalg.eigvals(xhat0)) * np.eye(n))
        for t in range(N):
            cons.append(
                cp.trace(W_var_sep[t + 1] +
                         Sigma[t][:, :] - 2 * E_w[t]) <= zeta**2
            )
            cons.append(
                cp.trace(V_var_sep[t] + Gamma[t][:, :] - 2 * E_v[t]) <= zeta**2)
            cons.append(
                W_var_sep[t +
                          1] >> np.min(np.linalg.eigvals(Sigma[t][:, :])) * np.eye(n)
            )
            cons.append(
                V_var_sep[t] >> np.min(
                    np.linalg.eigvals(Gamma[t][:, :])) * np.eye(p)
            )
        X0_hat_sqrt = sqrtm(xhat0)
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]),
                               X0_hat_sqrt), E_x0],
                    [E_x0, np.eye(n)],
                ]
            )
            >> 0
        )
        for t in range(N):
            temp = sqrtm(Sigma[t][:, :])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(
                            cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                        [E_w[t], np.eye(n)],
                    ]
                )
                >> 0
            )
            temp = sqrtm(Gamma[t][:, :])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                        [E_v[t], np.eye(p)],
                    ]
                )
                >> 0
            )

        cons.append(
            cp.bmat(
                [
                    [
                        E,
                        cp.matmul(
                            cp.matmul(
                                cp.matmul(cp.matmul(H.T, Q_block), G), W_var), D.T
                        )
                        + M_var / 2,
                    ],
                    [
                        (
                            cp.matmul(
                                cp.matmul(
                                    cp.matmul(cp.matmul(H.T, Q_block), G), W_var),
                                D.T,
                            )
                            + M_var / 2
                        ).T,
                        cp.matmul(cp.matmul(D, W_var), D.T) + V_var,
                    ],
                ]
            )
            >> 0
        )
        obj = -cp.trace(cp.matmul(E, inv_cons)) + cp.trace(
            cp.matmul(cp.matmul(cp.matmul(G.T, Q_block), G), W_var)
        )

        prob = cp.Problem(cp.Maximize(obj), cons)
        prob.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=True,
        )
        
        W_list = [W_var.value[i*n:(i+1)*n, i*n:(i+1)*n] for i in range(N+1)]
        V_list = [V_var.value[i*p:(i+1)*p, i*p:(i+1)*p] for i in range(N)]

        return W_list[0], W_list[1:], V_list


    def _design_dro_noise_with_dense_matrices(self):
        # Code from https://github.com/RAO-EPFL/DR-Control/
        A = self.A_list
        B = self.B_list
        R = self.R_list
        C = self.C_list
        Q = self.Q_list
        zeta = self.zeta
        xhat0 = self.P0
        Gamma = self.Gamma_list
        Sigma = self.Sigma_list
        N = self.N
        n = A[0].shape[0]
        m = R[0].shape[0]
        p = Gamma[0].shape[0]

        #### Creating Block Matrices for SDP ####
        R_block = np.zeros([N, N, m, m])
        C_block = np.zeros([N, N + 1, p, n])
        for t in range(N):
            R_block[t, t] = R[t][:, :]
            C_block[t, t] = C[t][:, :]
        Q_block = np.zeros([n * (N + 1), n * (N + 1)])
        for t in range(N):
            Q_block[t * n: t * n + n, t * n: t * n + n] = Q[t][:, :]

        R_block = np.reshape(R_block.transpose(0, 2, 1, 3), (m * N, m * N))
        # Q_block = np.reshape(Q_block.transpose(0, 2, 1, 3), (n * (T + 1), n * (T + 1)))
        C_block = np.reshape(C_block.transpose(
            0, 2, 1, 3), (p * N, n * (N + 1)))

        # initialize H and G as zero matrices
        G = np.zeros((n * (N + 1), n * (N + 1)))
        H = np.zeros((n * (N + 1), m * N))
        for t in range(N + 1):
            for s in range(t + 1):
                # print(GG[t * n : t * n + n, s * n : s * n + n])
                G[t * n: t * n + n, s * n: s * n +
                    n] = cumulative_product(A, s, t)
                if t != s:
                    H[t * n: t * n + n, s * m: s * m + m] = (
                        cumulative_product(A, s + 1, t) @ B[s][:, :]
                    )
        D = np.matmul(C_block, G)
        inv_cons = np.linalg.inv(R_block + H.T @ Q_block @ H)

        ### OPTIMIZATION MODEL ###
        E = cp.Variable((m * N, m * N), symmetric=True)
        E_x0 = cp.Variable((n, n), symmetric=True)
        W_var = cp.Variable((n * (N + 1), n * (N + 1)))
        V_var = cp.Variable((p * N, p * N))
        E_w = []
        E_v = []
        W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
        V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
        for t in range(N):
            E_w.append(cp.Variable((n, n), symmetric=True))
            E_v.append(cp.Variable((p, p), symmetric=True))
            W_var_sep.append(cp.Variable((n, n), symmetric=True))
            V_var_sep.append(cp.Variable((p, p), symmetric=True))
        W_var_sep.append(cp.Variable((n, n), symmetric=True))
        M_var = cp.Variable((m * N, p * N))
        M_var_sep = []
        num_lower_tri = num_lower_triangular_elements(N, N)
        for k in range(num_lower_tri):
            M_var_sep.append(cp.Variable((m, p)))
        k = 0
        cons = []
        for t in range(N):
            for s in range(t + 1):
                cons.append(M_var[t * m: t * m + m, p *
                            s: p * s + p] == M_var_sep[k])
                cons.append(M_var_sep[k] == np.zeros((m, p)))
                k = k + 1

        for t in range(N + 1):
            cons.append(W_var[n * t: n * t + n, n *
                        t: n * t + n] == W_var_sep[t])
            cons.append(W_var_sep[t] >> 0)

        # Setting the rest of the elements of the matrix to zero
        for i in range(W_var.shape[0]):
            for j in range(W_var.shape[1]):
                # If the element is not in one of the blocks
                if not any(
                    n * t <= i < n * (t + 1) and n * t <= j < n * (t + 1)
                    for t in range(N + 1)
                ):
                    cons.append(W_var[i, j] == 0)

        for t in range(N):
            cons.append(V_var[p * t: p * t + p, p *
                        t: p * t + p] == V_var_sep[t])
            cons.append(V_var_sep[t] >> 0)
            cons.append(E_v[t] >> 0)
            cons.append(E_w[t] >> 0)
        # Setting the rest of the elements of the matrix to zero
        for i in range(V_var.shape[0]):
            for j in range(V_var.shape[1]):
                # If the element is not in one of the blocks
                if not any(
                    p * t <= i < p * (t + 1) and p * t <= j < p * (t + 1)
                    for t in range(N + 1)
                ):
                    cons.append(V_var[i, j] == 0)

        cons.append(E >> 0)
        cons.append(E_x0 >> 0)

        cons.append(cp.trace(W_var_sep[0] + xhat0 - 2 * E_x0) <= zeta**2)
        cons.append(W_var_sep[0] >> np.min(
            np.linalg.eigvals(xhat0)) * np.eye(n))
        for t in range(N):
            cons.append(
                cp.trace(W_var_sep[t + 1] +
                         Sigma[t][:, :] - 2 * E_w[t]) <= zeta**2
            )
            cons.append(
                cp.trace(V_var_sep[t] + Gamma[t][:, :] - 2 * E_v[t]) <= zeta**2)
            cons.append(
                W_var_sep[t +
                          1] >> np.min(np.linalg.eigvals(Sigma[t][:, :])) * np.eye(n)
            )
            cons.append(
                V_var_sep[t] >> np.min(
                    np.linalg.eigvals(Gamma[t][:, :])) * np.eye(p)
            )
        X0_hat_sqrt = sqrtm(xhat0)
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]),
                               X0_hat_sqrt), E_x0],
                    [E_x0, np.eye(n)],
                ]
            )
            >> 0
        )
        for t in range(N):
            temp = sqrtm(Sigma[t][:, :])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(
                            cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                        [E_w[t], np.eye(n)],
                    ]
                )
                >> 0
            )
            temp = sqrtm(Gamma[t][:, :])
            cons.append(
                cp.bmat(
                    [
                        [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                        [E_v[t], np.eye(p)],
                    ]
                )
                >> 0
            )

        cons.append(
            cp.bmat(
                [
                    [
                        E,
                        cp.matmul(
                            cp.matmul(
                                cp.matmul(cp.matmul(H.T, Q_block), G), W_var), D.T
                        )
                        + M_var / 2,
                    ],
                    [
                        (
                            cp.matmul(
                                cp.matmul(
                                    cp.matmul(cp.matmul(H.T, Q_block), G), W_var),
                                D.T,
                            )
                            + M_var / 2
                        ).T,
                        cp.matmul(cp.matmul(D, W_var), D.T) + V_var,
                    ],
                ]
            )
            >> 0
        )
        obj = -cp.trace(cp.matmul(E, inv_cons)) + cp.trace(
            cp.matmul(cp.matmul(cp.matmul(G.T, Q_block), G), W_var)
        )

        prob = cp.Problem(cp.Maximize(obj), cons)
        prob.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=True,
        )
        E_check = (
            (H.T @ Q_block @ G @ W_var.value @ D.T + M_var.value / 2)
            @ np.linalg.inv(D @ W_var.value @ D.T + V_var.value)
            @ (M_var.value / 2 + H.T @ Q_block @ G @ W_var.value @ D.T).T
        )
        M = M_var.value
        M[np.abs(M) <= 1e-11] = 0
        W_var_clean = W_var.value
        V_var_clean = V_var.value
        W_var_clean[W_var_clean <= 1e-11] = 0
        V_var_clean[V_var_clean <= 1e-11] = 0

        E_new = (
            (H.T @ Q_block @ G @ W_var_clean @ D.T + M / 2)
            @ np.linalg.inv(D @ W_var_clean @ D.T + V_var_clean)
            @ (M / 2 + H.T @ Q_block @ G @ W_var_clean @ D.T).T
        )
        obj_clean = -np.trace(np.matmul(E_new, inv_cons)) + np.trace(
            np.matmul(np.matmul(np.matmul(G.T, Q_block), G), W_var_clean)
        )
        W_list = [W_var.value[i*n:(i+1)*n, i*n:(i+1)*n] for i in range(N+1)]
        V_list = [V_var.value[i*p:(i+1)*p, i*p:(i+1)*p] for i in range(N)]

        return W_list[0], W_list[1:], V_list
