from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from src.domain.type import Matrix
import numpy as np


class NoiseSampler(ABC):

    @abstractmethod
    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        ...


def ZeroNoise(NoiseSampler):
    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        dim = int(np.asarray(cov).shape[0])
        return np.zeros((dim, 1), dtype=float)


def GaussianNoise(NoiseSampler):

    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        cov = np.asarray(cov, dtype=float)
        dim = int(cov.shape[0])
        if np.allclose(cov, 0.0):
            return np.zeros((dim, 1), dtype=float)
        return np.linalg.cholesky(cov) @ rng.standard_normal((dim, 1))


def UniformNoise(NoiseSampler):

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, cov: Matrix, rng: np.random.Generator) -> Matrix:
        cov = np.asarray(cov, dtype=float)
        dim = int(cov.shape[0])
        if np.allclose(cov, 0.0):
            return np.zeros((dim, 1), dtype=float)
        return np.linalg.cholesky(cov) @ rng.uniform(self.a, self.b, size=(dim, 1))
