import numpy as np
import matplotlib.pyplot as plt

from src.infrastructure.adapters.outbound.controllers.nominal_mpc import NominalMpc
from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg import Lqg
from src.infrastructure.adapters.outbound.noise_samplers import GaussianNoise, ZeroNoise


def main():
    figs_dir = "simulations/figs/mpc/double_integrator"

    dt = 0.1
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt],
                  [dt]])
    C = np.eye(A.shape[0], A.shape[1])      # measure position only

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    Qn = np.diag([50.0, 5.0])

    x0 = np.array([[10.0],
                   [0.0]])

    N = 50
    seed = 1

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C,
        N=N,
        Sigma=None, Gamma=None,
        process_noise_sampler=ZeroNoise(),
        measurement_noise_sampler=ZeroNoise(),
        seed=seed,
    )

    # --- controller (MPC) ---
    mpc = NominalMpc(N=N, A=A, B=B, Q=Q, R=R)

    # --- application service (owns the loop) ---
    sim = SimulationService(
        plant=plant,
        controller=mpc,
        N=N,
        x0=x0,
    )

    result = sim.execute()
    x, y, u = result.x, result.y, result.u

    # --- plots ---
    plt.figure()
    plt.plot([yi[0, 0] for yi in y])
    plt.title("y (measurement)")
    plt.savefig(f"{figs_dir}/y.png")

    plt.figure()
    plt.plot([xi[0, 0] for xi in x])
    plt.title("x[0] (position)")
    plt.savefig(f"{figs_dir}/x0.png")

    plt.figure()
    plt.plot([xi[1, 0] for xi in x])
    plt.title("x[1] (velocity)")
    plt.savefig(f"{figs_dir}/x1.png")

    plt.figure()
    plt.plot([ui[0, 0] for ui in u])
    plt.title("u (control)")
    plt.savefig(f"{figs_dir}/u.png")

    print("Saved plots!")


if __name__ == "__main__":
    main()
