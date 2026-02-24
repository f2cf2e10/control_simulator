from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from src.application.ports.outbound.controller import Controller
from src.domain.type import Matrix, MatrixOrSeq
from src.infrastructure.adapters.outbound.utils import MatrixOps


class Lqg(Controller):
    """
    Finite-horizon discrete-time LQG controller.

    Uses:
      u_k = -K_k xhat_k

    Estimator (output feedback):
      if k==0: xhat_0 <- xhat_0 + L_0 (y_0 - C_0 xhat_0)   (optional)
      if k>0 : xhat_k^- = A_{k-1} xhat_{k-1} + B_{k-1} u_{k-1}
              xhat_k   = xhat_k^- + L_k (y_k - C_k xhat_k^-)

    Internally stores A,B,Q,R,Sigma as lists of length N, and C,Gamma as lists of length N+1.
    If a single matrix is provided, it is replicated to the required length.
    """

    def __init__(
        self,
        N: int,
        A: MatrixOrSeq,
        B: MatrixOrSeq,
        C: MatrixOrSeq,
        Q: MatrixOrSeq,
        R: MatrixOrSeq,
        Qn: Matrix,
        Sigma: MatrixOrSeq,
        Gamma: MatrixOrSeq,
        xhat0: Matrix,
        P0: Matrix,
        use_y0_update: bool = True,
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
        self.xhat0 = MatrixOps.to_col(xhat0)
        self.P0 = np.asarray(P0, dtype=float)
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
        self.K_list, self.L_list, self.S_list, self.P_post_list = self._design_lqg()

        # Runtime state
        self._xhat: Optional[Matrix] = None
        self._u_prev: Optional[Matrix] = None

    # ---------------- ControllerPort API ----------------

    def initialize(self) -> None:
        self._xhat = self.xhat0.copy()
        self._k = 0
        self._u_prev = None

    def compute(self, y_k: Matrix) -> Matrix:
        if self._xhat is None:
            raise RuntimeError(
                "Controller not initialized. Call initialize() first.")
        if not (0 <= self._k < self.N):
            raise ValueError(f"k must be in [0, {self.N-1}], got {k}")

        y_k = MatrixOps.to_col(y_k)

        # Predict to time k (except k=0)
        if self._k > 0:
            Akm1 = self.A_list[self._k - 1]
            Bkm1 = self.B_list[self._k - 1]
            if self._u_prev is None:
                self._u_prev = np.zeros((Bkm1.shape[1], 1), dtype=float)
            self._xhat = Akm1 @ self._xhat + Bkm1 @ self._u_prev

        # Measurement update at time k
        if self._k > 0 or self.use_y0_update:
            Ck = self.C_list[self._k]
            Lk = self.L_list[self._k]
            self._xhat = self._xhat + Lk @ (y_k - Ck @ self._xhat)

        # Control law
        uk = -self.K_list[self._k] @ self._xhat
        self._u_prev = uk
        self._k += 1
        return uk

    # ---------------- internals ----------------

    def _design_lqg(self) -> Tuple[List[Matrix], List[Matrix], List[Matrix], List[Matrix]]:
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
        Sigma_list, Gamma_list = self.Sigma_list, self.Gamma_list

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
        P_list[0] = self.P0.copy()

        for k in range(N):
            Ck = C_list[k]
            Gammak = Gamma_list[k]
            Pk = P_list[k]

            Cbar = Ck @ Pk @ Ck.T + Gammak
            Lk = Pk @ Ck.T @ np.linalg.inv(Cbar)
            L_list[k] = Lk

            middle = Pk - Pk @ Ck.T @ np.linalg.solve(Cbar, Ck.dot(Pk))
            if k < N:
                Sigmak = Sigma_list[k]
                Ak = A_list[k]
                P_list[k + 1] = Ak @ middle @ Ak.T + Sigmak

        return K_list, L_list, S_list, P_list
