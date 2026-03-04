import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.infrastructure.adapters.outbound.cost import Quadratic
from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg.lqg_dro import LqgDro
from src.infrastructure.adapters.outbound.noise_samplers import GaussianNoise, UniformNoise
from src.infrastructure.adapters.inbound.params import marginally_stable_system


def main():
    seed = 171
    figs_dir = "simulations/figs/lqg_dro/double_integrator"
    os.makedirs(figs_dir, exist_ok=True)

    A = marginally_stable_system["A"]
    B = marginally_stable_system["B"]
    C = marginally_stable_system["C"]
    Q = marginally_stable_system["Q"]
    R = marginally_stable_system["R"]
    Qn = marginally_stable_system["Qn"]
    SigmaPlant = marginally_stable_system["SigmaPlant"]
    GammaPlant = marginally_stable_system["GammaPlant"]
    SigmaWrongGuess = marginally_stable_system["SigmaWrongGuess"]
    GammaWrongGuess = marginally_stable_system["GammaWrongGuess"]
    x0_cov = marginally_stable_system["x0_cov"]
    x0_mean = marginally_stable_system["x0_mean"]

    N = 50
    Npaths = 1000
    betas = [0.0, 0.5, 1.0]
    Nbetas = len(betas)
    zeta = 1.0

    df_cost = pd.DataFrame(np.zeros([Npaths, Nbetas]), columns=betas)
    df_x0 = pd.DataFrame(np.zeros([Npaths, Nbetas]), columns=betas)
    df_x1 = pd.DataFrame(np.zeros([Npaths, Nbetas]), columns=betas)

    for beta in betas:
        SigmaController = 0.5 * (
            (1.0 - beta) * SigmaPlant + beta * SigmaWrongGuess
            + ((1.0 - beta) * SigmaPlant + beta * SigmaWrongGuess).T
        )
        GammaController = 0.5 * (
            (1.0 - beta) * GammaPlant + beta * GammaWrongGuess
            + ((1.0 - beta) * GammaPlant + beta * GammaWrongGuess).T
        )

        plant = LinearPlant(
            A=A, B=B, C=C, N=N,
            Sigma=SigmaPlant, Gamma=GammaPlant,
            process_noise_sampler=GaussianNoise(),
            measurement_noise_sampler=GaussianNoise(),
            seed=seed,  # same seed each beta for fair comparison
        )
        lqg_dro = LqgDro(
            N=N,
            A=A, B=B, C=C,
            Q=Q, R=R, Qn=Qn,
            Sigma=SigmaController, Gamma=GammaController,
            x0=x0_mean,
            P0=x0_cov,
            zeta=zeta,
            use_y0_update=True,
        )
        sim = SimulationService(
            plant=plant,
            controller=lqg_dro,
            cost=Quadratic(N, Q, R, Qn),
            N=N,
            x0=x0_mean,
        )

        for i in range(Npaths):
            result = sim.execute()
            xN = result.x[N].reshape(-1)
            df_cost.at[i, beta] = result.cost
            df_x0.at[i, beta] = xN[0]
            df_x1.at[i, beta] = xN[1]

    # diff_noise case: non-Gaussian plant noise, beta=1 controller mismatch
    beta = 1.0
    SigmaController = 0.5 * (
        (1.0 - beta) * SigmaPlant + beta * SigmaWrongGuess
        + ((1.0 - beta) * SigmaPlant + beta * SigmaWrongGuess).T
    )
    GammaController = 0.5 * (
        (1.0 - beta) * GammaPlant + beta * GammaWrongGuess
        + ((1.0 - beta) * GammaPlant + beta * GammaWrongGuess).T
    )

    # Uniform(-sqrt(3), sqrt(3)) has unit variance, so it preserves the target covariance scaling.
    uniform_noise = UniformNoise(-np.sqrt(3.0), np.sqrt(3.0))
    plant = LinearPlant(
        A=A, B=B, C=C, N=N,
        Sigma=SigmaPlant, Gamma=GammaPlant,
        process_noise_sampler=uniform_noise,
        measurement_noise_sampler=uniform_noise,
        seed=seed,
    )
    lqg_dro = LqgDro(
        N=N,
        A=A, B=B, C=C,
        Q=Q, R=R, Qn=Qn,
        Sigma=SigmaController, Gamma=GammaController,
        x0=x0_mean,
        P0=x0_cov,
        zeta=zeta,
        use_y0_update=True,
    )
    sim = SimulationService(
        plant=plant,
        controller=lqg_dro,
        cost=Quadratic(N, Q, R, Qn),
        N=N,
        x0=x0_mean,
    )
    cost_diff_noise = np.zeros(Npaths)
    xN_norm_diff_noise = np.zeros(Npaths)
    for i in range(Npaths):
        result = sim.execute()
        cost_diff_noise[i] = result.cost
        xN = result.x[N].reshape(-1)
        xN_norm_diff_noise[i] = np.linalg.norm(xN)

    df_cost.to_csv(f"{figs_dir}/cost.csv", index=False)
    df_x0.to_csv(f"{figs_dir}/xN_0.csv", index=False)
    df_x1.to_csv(f"{figs_dir}/xN_1.csv", index=False)

    # --- plots ---
    plt.figure()
    plt.boxplot(df_x0)
    plt.xticks(range(1, Nbetas + 1), betas)
    plt.title("xN[0]")
    plt.savefig(f"{figs_dir}/xN_0.png")

    plt.figure()
    plt.boxplot(df_x1)
    plt.xticks(range(1, Nbetas + 1), betas)
    plt.title("xN[1]")
    plt.savefig(f"{figs_dir}/xN_1.png")

    # Cost robustness cases for LQG-DRO only, normalized by beta=0 Gaussian baseline.
    base_cost = df_cost[0.0].to_numpy()
    base_cost = np.maximum(base_cost, 1e-12)
    ratio_beta05 = df_cost[0.5].to_numpy() / base_cost
    ratio_beta10 = df_cost[1.0].to_numpy() / base_cost
    ratio_diff_noise = cost_diff_noise / base_cost

    df_cost_ratio = pd.DataFrame({
        "beta_0.5": ratio_beta05,
        "beta_1.0": ratio_beta10,
        "diff_noise": ratio_diff_noise,
    })
    df_cost_ratio.to_csv(f"{figs_dir}/cost_ratio_cases.csv", index=False)

    plt.figure()
    plt.boxplot(df_cost_ratio)
    plt.xticks([1, 2, 3], ["beta_0.5", "beta_1.0", "diff_noise"])
    plt.title("LQG-DRO cost ratio vs beta=0 baseline")
    plt.ylabel("J(case) / J(beta=0)")
    plt.savefig(f"{figs_dir}/cost.png")

    # State robustness cases for LQG-DRO only, normalized by beta=0 Gaussian baseline.
    xN_norm_base = np.sqrt(df_x0[0.0].to_numpy() ** 2 + df_x1[0.0].to_numpy() ** 2)
    xN_norm_base = np.maximum(xN_norm_base, 1e-12)
    xN_norm_ratio = pd.DataFrame({
        "beta_0.5": np.sqrt(df_x0[0.5].to_numpy() ** 2 + df_x1[0.5].to_numpy() ** 2) / xN_norm_base,
        "beta_1.0": np.sqrt(df_x0[1.0].to_numpy() ** 2 + df_x1[1.0].to_numpy() ** 2) / xN_norm_base,
        "diff_noise": xN_norm_diff_noise / xN_norm_base,
    })
    xN_norm_ratio.to_csv(f"{figs_dir}/xN_norm_ratio_cases.csv", index=False)

    plt.figure()
    plt.boxplot(xN_norm_ratio)
    plt.xticks([1, 2, 3], ["beta_0.5", "beta_1.0", "diff_noise"])
    plt.title("LQG-DRO terminal-state norm ratio vs beta=0 baseline")
    plt.ylabel("||x_N||(case) / ||x_N||(beta=0)")
    plt.savefig(f"{figs_dir}/xN_norm_ratio.png")


if __name__ == "__main__":
    main()
