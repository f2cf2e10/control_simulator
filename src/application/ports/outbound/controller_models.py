from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.domain.type import Matrix


@dataclass
class AncillaryControlLaw:
    transform: Callable[[Matrix, Matrix], Matrix]

    def __call__(self, v0: Matrix, y_k: Matrix) -> Matrix:
        return self.transform(v0, y_k)
