import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# ------------------------ Configuration ------------------------
SEED = 42
np.random.seed(SEED)

T = 144  # 24 hours * 6 (10-min intervals)
BASE_DEMAND = 100.0
CAPACITY_NOMINAL = 100.0
SOLAR_NOMINAL = 40.0
WIND_NOMINAL = 35.0
HYDRO_NOMINAL = 25.0

# PSO hyperparameters
N_PARTICLES = 30
MAX_ITERS = 120
W_INERTIA = 0.72
C1 = 1.4
C2 = 1.4

# Moving average window for demand prediction
MA_WINDOW = 6

# Switching cost threshold
SWITCH_THRESHOLD = 1.0

# Safety factor for uncertainty
K_SIGMA = 0.8

# Supply-demand tolerance requirement
TOLERANCE = 0.05


# ------------------------ Environment Simulation ------------------------
def simulate_environment(T, seed=SEED):
    np.random.seed(seed)
    t = np.arange(T)

    demand_true = (
        BASE_DEMAND
        + 15.0 * np.sin(2 * np.pi * t / T - 0.5)
        + 8.0 * np.sin(4 * np.pi * t / T)
        + np.random.normal(0, 3.0, T)
    )

    demand_true = np.clip(demand_true, 0.5 * BASE_DEMAND, 1.2 * BASE_DEMAND)

    cloud = np.ones(T)
    cloud_events = np.random.choice(T, 3, replace=False)
    for c in cloud_events:
        radius = np.random.randint(6, 30)
        profile = np.linspace(0.5, 1.0, 2 * radius + 1)
        s = max(0, c - radius)
        e = min(T, c + radius + 1)
        cloud[s:e] *= profile[(s - (c - radius)):(e - (c - radius))]

    wind_factor = np.random.normal(1.0, 0.15, T)
    wind_events = np.random.choice(T, 2, replace=False)
    for c in wind_events:
        radius = np.random.randint(4, 20)
        profile = np.linspace(0.7, 1.5, 2 * radius + 1)
        s = max(0, c - radius)
        e = min(T, c + radius + 1)
        wind_factor[s:e] *= profile[(s - (c - radius)):(e - (c - radius))]

    wind_factor = np.clip(wind_factor, 0.2, 1.5)

    hydro_flow = 0.9 + 0.05 * np.sin(2 * np.pi * (t + 10) / T) + np.random.normal(0, 0.02, T)
    hydro_available = np.clip(HYDRO_NOMINAL * hydro_flow, 0, HYDRO_NOMINAL)

    solar_mean = SOLAR_NOMINAL * 0.9 * cloud
    wind_mean = WIND_NOMINAL * wind_factor
    hydro_mean = hydro_available

    solar_std = 0.08 * solar_mean + 0.5
    wind_std = 0.12 * wind_mean + 0.3
    hydro_std = 0.03 * hydro_mean + 0.1

    return {
        "t": t,
        "demand_true": demand_true,
        "solar_mean": solar_mean,
        "wind_mean": wind_mean,
        "hydro_mean": hydro_mean,
        "solar_std": solar_std,
        "wind_std": wind_std,
        "hydro_std": hydro_std,
    }


# ------------------------ Demand Prediction ------------------------
def predict_demand_moving_average(demand_true, window=MA_WINDOW):
    T = len(demand_true)
    pred = np.zeros(T)
    for i in range(T):
        s = max(0, i - window)
        pred[i] = np.mean(demand_true[s:i + 1])
    pred = 0.6 * pred + 0.4 * demand_true
    return pred


# ------------------------ Fitness Function ------------------------
def fitness_of_alloc(flat_alloc, T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std):
    alloc = flat_alloc.reshape((T, 3))

    alloc[:, 0] = np.clip(alloc[:, 0], 0, solar_mean)
    alloc[:, 1] = np.clip(alloc[:, 1], 0, wind_mean)
    alloc[:, 2] = np.clip(alloc[:, 2], 0, hydro_mean)

    total_supply = alloc.sum(axis=1)

    lower = (1.0 - TOLERANCE) * demand_pred
    upper = (1.0 + TOLERANCE) * demand_pred

    imbalance = np.where((total_supply < lower) | (total_supply > upper),
                         np.minimum(np.abs(total_supply - lower), np.abs(total_supply - upper)) ** 2,
                         0.0)
    imbalance_pen = np.sum(imbalance)

    changes = np.abs(np.vstack([np.zeros((1, 3)), alloc[1:, :] - alloc[:-1, :]]))
    switching_pen = np.sum(np.maximum(0.0, changes - SWITCH_THRESHOLD))

    eps = 1e-8
    frac_solar = alloc[:, 0] / (solar_mean + eps)
    frac_wind = alloc[:, 1] / (wind_mean + eps)
    frac_hydro = alloc[:, 2] / (hydro_mean + eps)

    per_t_std = np.sqrt((frac_solar * solar_std) ** 2 +
                        (frac_wind * wind_std) ** 2 +
                        (frac_hydro * hydro_std) ** 2)

    reserve_gap = np.maximum(0.0, demand_pred - (total_supply - K_SIGMA * per_t_std))
    uncertainty_pen = np.sum(reserve_gap ** 2)

    cap_pen = np.sum(np.maximum(0.0, total_supply - CAPACITY_NOMINAL) ** 2) * 50.0

    return imbalance_pen + 5.0 * switching_pen + 2.5 * uncertainty_pen + cap_pen


# ------------------------ PSO Algorithm ------------------------
def run_pso(T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std):
    dim = 3 * T
    base_split = np.array([SOLAR_NOMINAL, WIND_NOMINAL, HYDRO_NOMINAL])
    base_split = base_split / base_split.sum()

    def init_particle():
        alloc = np.zeros((T, 3))
        for i in range(T):
            desired = demand_pred[i] * base_split + np.random.normal(0, 1.5, 3)
            desired[0] = np.clip(desired[0], 0, solar_mean[i])
            desired[1] = np.clip(desired[1], 0, wind_mean[i])
            desired[2] = np.clip(desired[2], 0, hydro_mean[i])
            total = desired.sum()
            if total > CAPACITY_NOMINAL and total > 0:
                desired *= CAPACITY_NOMINAL / total
            alloc[i] = desired
        return alloc.flatten()

    particles = np.array([init_particle() for _ in range(N_PARTICLES)])
    velocities = np.random.normal(0, 1.0, particles.shape)

    pbest = particles.copy()
    pbest_scores = np.array([
        fitness_of_alloc(p, T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std)
        for p in particles
    ])

    gidx = np.argmin(pbest_scores)
    gbest = pbest[gidx].copy()
    gbest_score = pbest_scores[gidx]

    history = []
    w = W_INERTIA

    for it in range(MAX_ITERS):
        for i in range(N_PARTICLES):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (
                w * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )
            particles[i] += 0.5 * velocities[i]
            particles[i] = np.clip(particles[i], -50, 300)

            score = fitness_of_alloc(
                particles[i], T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std
            )

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = particles[i].copy()
                if score < gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score

        history.append(gbest_score)
        w *= 0.999

    return gbest.reshape((T, 3)), history, gbest_score


# ------------------------ Final Adjustment ------------------------
def finalize_alloc(alloc, demand_pred, solar_mean, wind_mean, hydro_mean):
    T = alloc.shape[0]
    out = alloc.copy()

    for i in range(T):
        out[i, 0] = np.clip(out[i, 0], 0, solar_mean[i])
        out[i, 1] = np.clip(out[i, 1], 0, wind_mean[i])
        out[i, 2] = np.clip(out[i, 2], 0, hydro_mean[i])

        total = out[i].sum()
        low = (1.0 - TOLERANCE) * demand_pred[i]
        high = (1.0 + TOLERANCE) * demand_pred[i]

        if total < low:
            need = low - total
            add_hydro = min(need, hydro_mean[i] - out[i, 2])
            out[i, 2] += add_hydro
            total = out[i].sum()

        if total > high:
            out[i] *= high / total

        total = out[i].sum()
        if total > CAPACITY_NOMINAL:
            out[i] *= CAPACITY_NOMINAL / total

    return out


# ------------------------ Main ------------------------
def main():
    start = time.time()

    env = simulate_environment(T)
    t = env["t"]
    demand_true = env["demand_true"]
    solar_mean = env["solar_mean"]
    wind_mean = env["wind_mean"]
    hydro_mean = env["hydro_mean"]
    solar_std = env["solar_std"]
    wind_std = env["wind_std"]
    hydro_std = env["hydro_std"]

    demand_pred = predict_demand_moving_average(demand_true)

    best_alloc_raw, history, best_score = run_pso(
        T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std
    )

    best_alloc = finalize_alloc(best_alloc_raw, demand_pred, solar_mean, wind_mean, hydro_mean)
    total_supply = best_alloc.sum(axis=1)
    imbalance = np.abs(total_supply - demand_pred)
    within_tol = (total_supply >= (1 - TOLERANCE) * demand_pred) & \
                 (total_supply <= (1 + TOLERANCE) * demand_pred)

    stability_pct = 100 * np.sum(within_tol) / T
    avg_imb = float(np.mean(imbalance))
    switching_total = float(np.sum(np.abs(np.vstack([np.zeros((1, 3)), best_alloc[1:] - best_alloc[:-1]]))))

    df = pd.DataFrame({
        "time_idx": t,
        "demand_true": demand_true,
        "demand_pred": demand_pred,
        "solar_alloc": best_alloc[:, 0],
        "wind_alloc": best_alloc[:, 1],
        "hydro_alloc": best_alloc[:, 2],
        "total_supply": total_supply,
        "imbalance_abs": imbalance,
        "within_5pct": within_tol
    })

    df_first10 = df.head(10)

    summary = {
        "final_fitness": best_score,
        "stability_pct_within_5pct": stability_pct,
        "avg_abs_imbalance_MW": avg_imb,
        "total_switching_MW": switching_total,
        "runtime_seconds": time.time() - start
    }

    print("\nSample allocation table (first 10 intervals):")
    print(df_first10.to_string(index=False))

    print("\nFitness summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaving PNG plots...")

    # Allocation plot
    plt.figure(figsize=(12, 4))
    plt.stackplot(t, best_alloc[:, 0], best_alloc[:, 1], best_alloc[:, 2])
    plt.title("Optimal Energy Allocation")
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.tight_layout()
    plt.savefig("allocation_plot.png")
    plt.close()

    # Fitness convergence
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.title("Fitness Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig("fitness_convergence.png")
    plt.close()

    # Supply vs demand
    plt.figure(figsize=(12, 4))
    plt.plot(t[:48], demand_pred[:48], label="Predicted demand")
    plt.plot(t[:48], total_supply[:48], label="Total supply")
    plt.fill_between(t[:48], 0.95 * demand_pred[:48], 1.05 * demand_pred[:48], alpha=0.2)
    plt.title("Supply vs Predicted Demand (First 8 Hours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("supply_vs_demand.png")
    plt.close()

    print("Saved: allocation_plot.png, fitness_convergence.png, supply_vs_demand.png")


if __name__ == "__main__":
    main()
