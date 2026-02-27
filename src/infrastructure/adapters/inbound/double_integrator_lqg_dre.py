import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg import Lqg
from src.infrastructure.adapters.outbound.noise_samplers import gaussian_noise
from src.infrastructure.adapters.outbound.cost import Quadratic
from src.infrastructure.adapters.inbound.params import plant1


def main():
    seed = 1
    figs_dir = "simulations/figs/lqg_dre/double_integrator"

    A = plant1["A"]
    B = plant1["B"]
    C = plant1["C"]
    Q = plant1["Q"]
    R = plant1["R"]
    Qn = plant1["Qn"]
    Sigma = plant1["Sigma"]
    Gamma = plant1["Gamma"]
    x0_cov = plant1["x0_cov"]
    x0_mean = plant1["x0_mean"]

    N = 50
    zeta = 1.0
    d, _ = A.shape

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C,
        N=N,
        Sigma=Sigma, Gamma=Gamma,
        process_noise_sampler=gaussian_noise,
        measurement_noise_sampler=gaussian_noise,
        seed=seed,
    )
    dzeta = zeta/(N*d + N)
    # --- controller (LQG-DRE) ---
    lqg = Lqg(
        N=N,
        A=A, B=B, C=C,
        Q=Q, R=R, Qn=Qn,
        Sigma=Sigma + np.eye(Sigma.shape[0]) * dzeta,
        Gamma=Gamma + np.eye(Gamma.shape[0]) * dzeta,
        x0=x0_mean,
        P0=x0_cov,
        use_y0_update=True,
    )

    # --- Quadratic Cost ---
    qc = Quadratic(N, Q, R, Qn)

    # --- application service (owns the path) ---
    sim = SimulationService(
        plant=plant,
        controller=lqg,
        cost=qc,
        N=N,
        x0=x0_mean,
    )

    Npaths = 1000
    cost = []
    x = []
    u = []
    for _ in range(Npaths):
        result = sim.execute()
        x.append(result.x[N])
        u.append(result.u)
        cost.append(result.cost)
    df = pd.DataFrame(cost)
    print(df.describe())
    df.to_csv(f"{figs_dir}/cost.csv")
    df = pd.DataFrame([i[0] for i in x])
    df = pd.concat([df,  pd.DataFrame([i[1] for i in x])], axis=1)
    print(df.describe())
    df.to_csv(f"{figs_dir}/state.csv")

    # --- plots ---
    plt.figure()
    plt.boxplot([[xi[0][0] for xi in x], [xi[1][0] for xi in x]])
    plt.xticks([1, 2], ["x[0]", "x[1]"])  # Label the boxes
    plt.title("x")
    plt.savefig(f"{figs_dir}/x.png")

    plt.figure()
    plt.boxplot(cost)
    plt.title("cost")
    plt.savefig(f"{figs_dir}/cost.png")

    print("Saved plots!")


if __name__ == "__main__":
    main()
