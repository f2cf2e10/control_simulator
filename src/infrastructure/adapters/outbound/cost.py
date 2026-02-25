from typing import List

from src.application.ports.outbound.cost import Cost
from src.domain.type import Matrix, MatrixOrSeq
from src.infrastructure.adapters.outbound.utils import MatrixOps


class Quadratic(Cost):
    def __init__(self, N: int, Q: MatrixOrSeq, R: MatrixOrSeq, Qn: Matrix):
        self.N = N
        self.Q_list: List[Matrix] = MatrixOps.to_list(Q, N-1, "Q")
        self.R_list: List[Matrix] = MatrixOps.to_list(R, N-1, "R")
        self.Qn = Qn

    def __call__(self, x: List[Matrix], u: List[Matrix]):
        cost = [x[i].T @ self.Q_list[i] @ x[i] + 
                u[i].T @ self.R_list[i] @ u[i]
                for i in range(self.N-1)]
        cost.append(x[self.N].T @ self.Qn @ x[self.N])
        return sum(cost)[0][0] 
