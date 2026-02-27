from infrastructure.adapters.outbound.cost import Quadratic
import numpy as np
import matplotlib.pyplot as plt

from src.infrastructure.adapters.outbound.controllers.nominal_mpc import NominalMpc
from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg import Lqg
from src.infrastructure.adapters.outbound.noise_samplers import zero_noise
from src.infrastructure.adapters.outbound.cost import Quadratic
from src.infrastructure.adapters.inbound.params import plant1


def main():
    seed = 1
    figs_dir = "simulations/figs/mpc/double_integrator"

    n = 4
    m = 2
    h = 0.5
    A = np.array([[1., 0, h, 0],
                  [0, 1., 0, h],
                  [0, 0, 1., 0],
                  [0, 0, 0, 1.]])
    B = np.array([[h*h/2, 0],
                  [0, h*h/2],
                  [h, 0],
                  [0, h]])
    G = np.array([[h, 0],
                  [0, h],
                  [0, 0],
                  [0, 0]])
    C = np.eye(n)
    Q = 10 * np.diag([0.1, 4, 1, 1])
    R = np.eye(m)
    gamma = 0.8
    wmax = 0.01
    wmin = -0.01
    Ccbf1 = np.array([[5./9], [1.], [0], [0]])
    bcbf1 = np.arrauy([[0.5/9]])
    Ccbf2 = np.array([[1.], [-1.], [0], [0]])
    bcbf2 = np.arrauy([[1.6]])
    epsilon = 0.05
    N = 300
    vmax = np.array([[5.], [2.]])
    Sigma = plant1["Sigma"]
    Gamma = plant1["Gamma"]
    x0_cov = plant1["x0_cov"]
    x0_mean = plant1["x0_mean"]

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C,
        N=N,
        Sigma=None, Gamma=None,
        process_noise_sampler=zero_noise,
        measurement_noise_sampler=zero_noise,
        seed=seed,
    )

    # --- controller (MPC) ---
    mpc = NominalMpc(N=N, A=A, B=B, Q=Q, R=R)

    # --- Quadratic Cost ---
    qc = Quadratic(N, Q, R, Qn)

    # --- application service (owns the path) ---
    sim = SimulationService(
        plant=plant,
        controller=mpc,
        cost=qc,
        N=N,
        x0=x0_mean,
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
