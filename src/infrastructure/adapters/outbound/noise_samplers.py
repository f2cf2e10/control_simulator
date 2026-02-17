from __future__ import annotations
from typing import Callable
from src.domain.type import Matrix
import numpy as np

NoiseSampler = Callable[[int, Matrix, np.random.Generator], Matrix]


def zero_noise(cov: Matrix, rng: np.random.Generator) -> Matrix:
    dim = int(np.asarray(cov).shape[0])
    return np.zeros((dim, 1), dtype=float)


def gaussian_noise(cov: Matrix, rng: np.random.Generator) -> Matrix:
    cov = np.asarray(cov, dtype=float)
    dim = int(cov.shape[0])
    if np.allclose(cov, 0.0):
        return np.zeros((dim, 1), dtype=float)
    return np.linalg.cholesky(cov) @ rng.standard_normal((dim, 1))


def uniform_noise(cov: Matrix, rng: np.random.Generator) -> Matrix:
    """
    Uniform noise with the SAME covariance as 'cov'.
    Uses z ~ U(-sqrt(3), +sqrt(3)) so Var(z)=1, then chol(cov) @ z.
    """
    cov = np.asarray(cov, dtype=float)
    dim = int(cov.shape[0])
    if np.allclose(cov, 0.0):
        return np.zeros((dim, 1), dtype=float)
    z = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(dim, 1))
    return np.linalg.cholesky(cov) @ z
