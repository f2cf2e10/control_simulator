from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from src.domain.type import Matrix
import numpy as np


def _cov_factor(cov: Matrix, tol: float = 1e-12) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square, got {cov.shape}")

    # Symmetrize to suppress tiny numerical asymmetries.
    cov = 0.5 * (cov + cov.T)
    dim = cov.shape[0]
    if np.allclose(cov, 0.0):
        return np.zeros((dim, dim), dtype=float)

    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # PSD fallback: cov = Q diag(lam) Q^T, factor = Q diag(sqrt(lam))
        lam, Q = np.linalg.eigh(cov)
        min_lam = float(np.min(lam))
        if min_lam < -tol:
            raise ValueError(
                f"covariance is not positive semidefinite (min eigenvalue={min_lam:.3e})")
        lam = np.clip(lam, 0.0, None)
        return Q @ np.diag(np.sqrt(lam))


class NoiseSampler(ABC):

    @abstractmethod
    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        ...


class ZeroNoise(NoiseSampler):
    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        dim = int(np.asarray(cov).shape[0])
        return np.zeros((dim, 1), dtype=float)


class GaussianNoise(NoiseSampler):

    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        cov = np.asarray(cov, dtype=float)
        dim = int(cov.shape[0])
        L = _cov_factor(cov)
        if not np.any(L):
            return np.zeros((dim, 1), dtype=float)
        return L @ rng.standard_normal((dim, 1))


class UniformNoise(NoiseSampler):

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        cov = np.asarray(cov, dtype=float)
        dim = int(cov.shape[0])
        L = _cov_factor(cov)
        if not np.any(L):
            return np.zeros((dim, 1), dtype=float)
        return L @ rng.uniform(self.a, self.b, size=(dim, 1))
