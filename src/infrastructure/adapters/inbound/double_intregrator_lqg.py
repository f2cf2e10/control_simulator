import numpy as np
import matplotlib.pyplot as plt

from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg import Lqg
from src.infrastructure.adapters.outbound.noise_samplers import gaussian_noise


def main():
    figs_dir = "simulations/figs/lqg/double_integrator" 

    dt = 0.1
    zeta = 0.1
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt],
                  [dt]])
    C = np.array([[1.0, 0.0]])      # measure position only

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    Qn = np.diag([50.0, 5.0])

    Sigma = np.diag([1e-1, 1e-1])   # process noise covariance
    Gamma = np.array([[1e-1]])      # measurement noise covariance

    x0_cov = np.diag([1.0, 1.0])    # initial estimation covariance P0
    x0_mean = np.array([[10.0],
                        [0.0]])

    N = 50
    seed = 1

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C,
        N=N,
        Sigma=Sigma, Gamma=Gamma,
        process_noise_sampler=gaussian_noise,
        measurement_noise_sampler=gaussian_noise,
        seed=seed,
    )

    # --- controller (LQG) ---
    lqg = Lqg(
        N=N,
        A=A, B=B, C=C,
        Q=Q, R=R, Qn=Qn,
        Sigma=Sigma, Gamma=Gamma,
        xhat0=x0_mean,
        P0=x0_cov,
        use_y0_update=True,
    )

    # --- application service (owns the loop) ---
    sim = SimulationService(
        plant=plant,
        controller=lqg,
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
