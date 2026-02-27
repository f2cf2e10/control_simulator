import matplotlib.pyplot as plt
import pandas as pd

from src.infrastructure.adapters.outbound.cost import Quadratic
from src.application.services.simulation_service import SimulationService
from src.infrastructure.adapters.outbound.plants.linear_plant import LinearPlant
from src.infrastructure.adapters.outbound.controllers.lqg import Lqg
from src.infrastructure.adapters.outbound.noise_samplers import GaussianNoise
from src.infrastructure.adapters.inbound.params import experiment1

def main():
    seed = 1
    figs_dir = "simulations/figs/lqg/double_integrator"

    A = experiment1["A"] 
    B = experiment1["B"] 
    C = experiment1["C"] 
    Q = experiment1["Q"] 
    R = experiment1["R"] 
    Qn = experiment1["Qn"] 
    SigmaPlant = experiment1["SigmaPlant"] 
    GammaPlant = experiment1["GammaPlant"] 
    SigmaController =experiment1["SigmaController"]  
    GammaController = experiment1["GammaController"]
    x0_cov = experiment1["x0_cov"] 
    x0_mean = experiment1["x0_mean"] 

    N = 50

    # --- plant (true system) ---
    plant = LinearPlant(
        A=A, B=B, C=C,
        N=N,
        Sigma=SigmaPlant, Gamma=GammaPlant,
        process_noise_sampler=GaussianNoise(),
        measurement_noise_sampler=GaussianNoise(),
        seed=seed,
    )

    # --- controller (LQG) ---
    lqg = Lqg(
        N=N,
        A=A, B=B, C=C,
        Q=Q, R=R, Qn=Qn,
        Sigma=SigmaController, Gamma=GammaController,
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
