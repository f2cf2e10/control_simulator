from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import numpy as np

from src.domain.type import Matrix, MatrixOrSeq
from src.application.ports.outbound.plant import Plant
from src.infrastructure.adapters.outbound.noise_samplers import gaussian_noise
from src.infrastructure.adapters.outbound.utils import MatrixOps

NoiseSampler = Callable[[int, Matrix, np.random.Generator], Matrix]


class LinearPlant(Plant):
    """
    Discrete-time linear plant.

      x_{k+1} = A_k x_k + B_k u_k + w_k
      y_k     = C_k x_k + v_k

    Internally stores A,B,Sigma as lists of length N, and C,Gamma as lists of length N+1.
    If a single matrix is provided, it is replicated to the required length.
    """

    def __init__(
        self,
        A: MatrixOrSeq,
        B: MatrixOrSeq,
        C: MatrixOrSeq,
        N: int,
        Sigma: Optional[MatrixOrSeq] = None,
        Gamma: Optional[MatrixOrSeq] = None,
        process_noise_sampler:  NoiseSampler = gaussian_noise,
        measurement_noise_sampler: NoiseSampler = gaussian_noise,
        seed: Optional[int] = None
    ):
        self.N = int(N)
        if self.N <= 0:
            raise ValueError("N must be a positive integer")

        # Always store as lists
        self.A_list: List[Matrix] = MatrixOps.to_list(A, self.N, "A")
        self.B_list: List[Matrix] = MatrixOps.to_list(B, self.N, "B")
        self.C_list: List[Matrix] = MatrixOps.to_list(C, self.N + 1, "C")

        # Dimension checks (k=0)
        A0, B0, C0 = self.A_list[0], self.B_list[0], self.C_list[0]
        n = int(A0.shape[0])
        if A0.shape != (n, n):
            raise ValueError(f"A[0] must be (n,n), got {A0.shape}")
        if B0.shape[0] != n:
            raise ValueError(f"B[0] must have n rows, got {B0.shape}")
        if C0.shape[1] != n:
            raise ValueError(f"C[0] must have n cols, got {C0.shape}")
        p = int(C0.shape[0])

        self.Sigma_list: Optional[List[Matrix]] = None
        if Sigma is not None:
            self.Sigma_list = MatrixOps.to_list(Sigma, self.N, "Sigma")
            S0 = self.Sigma_list[0]
            if S0.shape != (n, n):
                raise ValueError(f"Sigma[0] must be (n,n), got {S0.shape}")

        self.Gamma_list: Optional[List[Matrix]] = None
        if Gamma is not None:
            self.Gamma_list = MatrixOps.to_list(Gamma, self.N + 1, "Gamma")
            G0 = self.Gamma_list[0]
            if G0.shape != (p, p):
                raise ValueError(f"Gamma[0] must be (p,p), got {G0.shape}")
        
        self._rng = np.random.default_rng(seed)
        self._process_noise_sampler = process_noise_sampler
        self._measurement_noise_sampler = measurement_noise_sampler

    def _noise_from_cov(self, cov: Matrix, dim: int) -> Matrix:
        cov = np.asarray(cov, dtype=float)
        if np.allclose(cov, 0.0):
            return np.zeros((dim, 1), dtype=float)
        return np.linalg.cholesky(cov) @ self._rng.standard_normal((dim, 1))

    def dims(self) -> Tuple[int, int, int]:
        n = int(self.A_list[0].shape[0])
        m = int(self.B_list[0].shape[1])
        p = int(self.C_list[0].shape[0])
        return n, m, p

    def set_initial_state(self, x0: Matrix) -> Matrix:
        self._k = 0
        return MatrixOps.to_col(x0)

    def measure(self, x_k: Matrix) -> Matrix:
        x_k = MatrixOps.to_col(x_k)
        Ck = self.C_list[self._k]
        y = Ck @ x_k
        if self.Gamma_list is not None:
            y = y + self._measurement_noise_sampler(self.Gamma_list[self._k], self._rng)
        return y

    def propagate(self, x_k: Matrix, u_k: Matrix) -> Matrix:
        x_k = MatrixOps.to_col(x_k)
        u_k = MatrixOps.to_col(u_k)
        Ak = self.A_list[self._k]
        Bk = self.B_list[self._k]
        x_next = Ak @ x_k + Bk @ u_k
        if self.Sigma_list is not None:
            x_next = x_next + self._process_noise_sampler(self.Sigma_list[self._k], self._rng)
        self._k += 1
        return x_next
