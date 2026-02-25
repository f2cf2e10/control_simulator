from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.domain.type import Matrix


class Cost(ABC):
    @abstractmethod
    def __call__(self, x: List[Matrix], u: List[Matrix]) -> None:
        """Calculate cost"""
        ...

