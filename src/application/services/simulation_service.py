from __future__ import annotations

from typing import List
import numpy as np

from src.application.ports.outbound.cost import Cost
from src.application.ports.inbound.simulation_usecase import SimulationUseCase
from src.application.ports.outbound.controller import Controller
from src.application.ports.outbound.plant import Plant
from src.domain.type import Matrix, SimulationResult


class SimulationService(SimulationUseCase):
    def __init__(self, plant: Plant, controller: Controller, cost: Cost, N: int, x0: Matrix):
        self.plant = plant
        self.controller = controller
        self.cost = cost
        self.N = int(N)
        self.x0 = np.asarray(x0, dtype=float)

    def execute(self) -> SimulationResult:
        x: List[Matrix] = []
        y: List[Matrix] = []
        u: List[Matrix] = []

        self.controller.initialize()

        x0 = self.plant.set_initial_state(self.x0)
        x.append(x0)

        y0 = self.plant.measure(x0)
        y.append(y0)

        for k in range(self.N):
            uk = self.controller.compute(y[k])
            u.append(uk)

            x_next = self.plant.propagate(x[k], uk)
            x.append(x_next)

            y_next = self.plant.measure(x_next)
            y.append(y_next)

        cost = self.cost(x, u)
        return SimulationResult(x=x, y=y, u=u, cost=cost)
