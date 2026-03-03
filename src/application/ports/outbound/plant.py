from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from src.domain.type import Matrix

class Plant(ABC):
    @abstractmethod
    def dims(self) -> Tuple[int, int, int]:
        """Return (n, m, p) for state, controller, output."""
        ...

    @abstractmethod
    def set_initial_state(self, x0: Matrix) -> Matrix:
        """Initialize plant for a new run; return x_0 (normalized shape)."""
        ...

    @abstractmethod
    def measure(self, x_k: Matrix) -> Matrix:
        """Compute y_k from x_k (may include measurement noise)."""
        ...

    @abstractmethod
    def propagate(self, x_k: Matrix, u_k: Matrix) -> Matrix:
        """Compute x_{k+1} from x_k and u_k (may include process noise)."""
        ...
