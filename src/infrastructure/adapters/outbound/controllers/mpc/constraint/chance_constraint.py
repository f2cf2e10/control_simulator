from __future__ import annotations

from typing import Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from src.application.ports.outbound.constraint import Constraint
from src.domain.type import Matrix


class ChanceConstraint(Constraint):
    def __init__(
        self,
        n: int,
        m: int,
        N: int,
        N_tilde: int,
        epsilon: float,
        Cprev: Matrix,
        Ccurr: Matrix,
        b: Matrix,
        quantile_provider: Callable[[int, float], Matrix],
        mean_state_indices: Optional[Sequence[int]] = None,
        step_start: int = 1,
        step_stop: Optional[int] = None,
    ):
        self.n = n
        self.m = m
        self.N = N
        self.N_tilde = N_tilde
        self.epsilon = epsilon
        self.Cprev = Cprev
        self.Ccurr = Ccurr
        self.b = b
        self.quantile_provider = quantile_provider
        self.mean_state_indices = None if mean_state_indices is None else tuple(mean_state_indices)
        self.step_start = step_start
        self.step_stop = step_stop

    def build(self, z, x0_param):
        constraints = []
        N_eff = self.N - self.N_tilde
        nx = self.N * self.n
        nz = nx + self.N * self.m

        d = self.Cprev.shape[0]
        i_start = max(1, int(self.step_start))
        i_stop = int(self.step_stop) if self.step_stop is not None else N_eff
        i_stop = min(i_stop, N_eff)
        if i_stop <= i_start:
            return constraints

        step_ids = list(range(i_start, i_stop))
        steps = len(step_ids)
        M = sp.lil_matrix((steps * d, nz))
        rhs = np.zeros(steps * d)

        for row_i, i in enumerate(step_ids):
            r = row_i * d
            M[r:r + d, (i - 1) * self.n:i * self.n] = self.Cprev
            M[r:r + d, i * self.n:(i + 1) * self.n] = self.Ccurr
            q = np.asarray(self.quantile_provider(i, self.epsilon), dtype=float).reshape(-1)
            if q.size == 1:
                q = np.full(d, q.item())
            if q.size != d:
                raise ValueError(f"quantile length must be {d}, got {q.size}")
            rhs[r:r + d] = np.asarray(self.b, dtype=float).reshape(-1) - q

        M = M.tocsc()
        mean = x0_param[list(self.mean_state_indices)] if self.mean_state_indices is not None else None
        if mean is None:
            constraints.append(M @ z <= rhs)
            return constraints

        mean_len = int(np.prod(mean.shape))
        mean_vec = cp.reshape(mean, (mean_len,), order="F")
        if mean_len != d:
            if d % mean_len != 0:
                raise ValueError(f"mean length must divide d={d}, got {mean_len}")
            mean_vec = cp.hstack([mean_vec] * (d // mean_len))

        mean_stacked = cp.hstack([mean_vec] * steps)
        constraints.append(M @ z <= rhs + mean_stacked)

        return constraints
