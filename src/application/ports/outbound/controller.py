from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.type import Matrix


class Controller(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Prepare controller for a new run (clear estimator/internal state)."""
        ...

    @abstractmethod
    def compute(self, y_k: Matrix) -> Matrix:
        """Return control input u_k given measurement y_k."""
        ...
