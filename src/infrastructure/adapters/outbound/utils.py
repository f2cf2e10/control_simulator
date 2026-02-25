import numpy as np

from typing import List, Sequence
from src.domain.type import Matrix, MatrixOrSeq


class MatrixOps:
    @staticmethod
    def is_matrix(obj: object) -> bool:
        return isinstance(obj, np.ndarray) and obj.ndim == 2

    @staticmethod
    def to_col(x: Matrix) -> Matrix:
        x = np.asarray(x, dtype=float)
        return x.reshape(-1, 1) if x.ndim == 1 else x

    @staticmethod
    def to_list(mats: MatrixOrSeq, length: int, name: str) -> List[Matrix]:
        if MatrixOps.is_matrix(mats):
            M = np.asarray(mats, dtype=float)  # type: ignore[arg-type]
            return [M.copy() for _ in range(length)]
        mats_seq = list(mats)  # type: ignore[arg-type]
        if len(mats_seq) != length:
            raise ValueError(
                f"{name} must have length {length}, got {len(mats_seq)}")
        return [np.asarray(M, dtype=float) for M in mats_seq]


