import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from copy import deepcopy

SEED = 42
np.random.seed(SEED)

T = 144  # 24 hours * 6 (10-min intervals)
dt_minutes = 10
BASE_DEMAND = 100.0  # MW nominal mean demand
CAPACITY_NOMINAL = 100.0  # total grid capacity MW
SOLAR_NOMINAL = 40.0
WIND_NOMINAL = 35.0
HYDRO_NOMINAL = 25.0

# PSO hyperparameters (tune these for quality/runtime)
N_PARTICLES = 30
MAX_ITERS = 120
W_INERTIA = 0.72
C1 = 1.4
C2 = 1.4

# Demand predictor window (moving average)
MA_WINDOW = 6  # 6 * 10min = 1 hour causal window

# Switching cost threshold (small variations below this are tolerated)
SWITCH_THRESHOLD = 1.0  # MW

# Safety sigma for uncertain supply (k * sigma)
K_SIGMA = 0.8

# Tolerance for supply meeting demand
TOLERANCE = 0.05  # 5%

# ------------------------ Simulation of environment ------------------------
def simulate_environment(T, seed=SEED):
    np.random.seed(seed)
    t = np.arange(T)

    # Demand: base + daily harmonics + noise
    demand_true = (BASE_DEMAND
                   + 15.0 * np.sin(2 * np.pi * t / T - 0.5)
                   + 8.0 * np.sin(4 * np.pi * t / T)
                   + np.random.normal(0, 3.0, T))
    # clamp extremes to reasonable band
    demand_true = np.clip(demand_true, 0.5 * BASE_DEMAND, 1.2 * BASE_DEMAND)

    # Cloud cover for solar: generate a baseline 1.0 and reduce in cloud events
    cloud = np.ones(T)
    n_cloud_events = 3
    cloud_centers = np.random.choice(T, size=n_cloud_events, replace=False)
    for c in cloud_centers:
        radius = np.random.randint(6, 30)
        # shape factor from 0.5 (center) up to 1.0 (edge)
        profile = np.linspace(0.5, 1.0, 2 * radius + 1)
        s = max(0, c - radius)
        e = min(T, c + radius + 1)
        seg = profile[(s - (c - radius)):(e - (c - radius))]
        cloud[s:e] *= seg

    # Wind: baseline factor with higher variance during storms
    wind_factor = np.random.normal(1.0, 0.15, T)
    n_wind_events = 2
    wind_centers = np.random.choice(T, size=n_wind_events, replace=False)
    for c in wind_centers:
        radius = np.random.randint(4, 20)
        profile = np.linspace(0.7, 1.5, 2 * radius + 1)
        s = max(0, c - radius)
        e = min(T, c + radius + 1)
        seg = profile[(s - (c - radius)):(e - (c - radius))]
        wind_factor[s:e] *= seg

    wind_factor = np.clip(wind_factor, 0.2, 1.5)

    # Hydro: relatively stable with minor flow variations
    hydro_flow = 0.9 + 0.05 * np.sin(2 * np.pi * (t + 10) / T) + np.random.normal(0, 0.02, T)
    hydro_available = np.clip(HYDRO_NOMINAL * hydro_flow, 0.0, HYDRO_NOMINAL)

    # Mean available generation
    solar_mean_available = SOLAR_NOMINAL * 0.9 * cloud  # assume panel efficiency
    wind_mean_available = WIND_NOMINAL * wind_factor
    hydro_mean_available = hydro_available

    # Uncertainty standard deviations (simple heuristic)
    solar_std = 0.08 * solar_mean_available + 0.5
    wind_std = 0.12 * wind_mean_available + 0.3
    hydro_std = 0.03 * hydro_mean_available + 0.1

    return {
        't': t,
        'demand_true': demand_true,
        'solar_mean': solar_mean_available,
        'wind_mean': wind_mean_available,
        'hydro_mean': hydro_mean_available,
        'solar_std': solar_std,
        'wind_std': wind_std,
        'hydro_std': hydro_std
    }

# ------------------------ Demand prediction (causal moving average) ------------------------
def predict_demand_moving_average(demand_true, window=MA_WINDOW):
    T = len(demand_true)
    pred = np.zeros(T)
    for i in range(T):
        s = max(0, i - window)
        pred[i] = np.mean(demand_true[s:i + 1]) if i > 0 else demand_true[0]
    # Mix with true (simulate some skill) but keep predictor causal-ish
    pred = 0.6 * pred + 0.4 * demand_true
    return pred

# ------------------------ Objective / fitness ------------------------
def fitness_of_alloc(flat_alloc, T,
                     demand_pred,
                     solar_mean, wind_mean, hydro_mean,
                     solar_std, wind_std, hydro_std):
    """
    Compute fitness (lower is better).
    Input flat_alloc shape: (3*T,)
    """
    alloc = flat_alloc.reshape((T, 3)).astype(float)

    # Clip per-source to [0, mean_available]
    alloc[:, 0] = np.clip(alloc[:, 0], 0.0, solar_mean)
    alloc[:, 1] = np.clip(alloc[:, 1], 0.0, wind_mean)
    alloc[:, 2] = np.clip(alloc[:, 2], 0.0, hydro_mean)

    total_supply = alloc.sum(axis=1)
    pred = demand_pred

    # 1) imbalance penalty (squared) if outside tolerance ±TOLERANCE
    lower = (1.0 - TOLERANCE) * pred
    upper = (1.0 + TOLERANCE) * pred
    # positive deviation outside band
    imbalance = np.where((total_supply < lower) | (total_supply > upper),
                         np.minimum(np.abs(total_supply - lower), np.abs(total_supply - upper)) ** 2,
                         0.0)
    imbalance_pen = np.sum(imbalance)

    # 2) switching penalty: penalize changes above threshold
    changes = np.abs(np.vstack([np.zeros((1, 3)), alloc[1:, :] - alloc[:-1, :]]))
    # only count excess beyond SWITCH_THRESHOLD
    excess_changes = np.maximum(0.0, changes - SWITCH_THRESHOLD)
    switching_pen = np.sum(excess_changes)

    # 3) uncertainty penalty: if mean_supply - k*sigma < predicted demand, penalize square shortfall
    # estimate std per timestep by scaling source stds proportionally to allocated fraction
    # avoid division by zero: where mean available is tiny, treat fraction as 0
    eps = 1e-8
    frac_solar = np.where(solar_mean > eps, alloc[:, 0] / (solar_mean + eps), 0.0)
    frac_wind = np.where(wind_mean > eps, alloc[:, 1] / (wind_mean + eps), 0.0)
    frac_hydro = np.where(hydro_mean > eps, alloc[:, 2] / (hydro_mean + eps), 0.0)
    # approximate combined std (assume independence)
    per_t_std = np.sqrt((frac_solar * solar_std) ** 2 + (frac_wind * wind_std) ** 2 + (frac_hydro * hydro_std) ** 2)
    mean_supply = total_supply
    reserve_gap = np.maximum(0.0, pred - (mean_supply - K_SIGMA * per_t_std))
    uncertainty_pen = np.sum(reserve_gap ** 2)

    # 4) capacity penalty: heavy if supply exceeds nominal capacity
    cap_pen = np.sum(np.maximum(0.0, total_supply - CAPACITY_NOMINAL) ** 2) * 50.0

    # Combine with weights: these were chosen heuristically to balance the terms
    fitness_value = imbalance_pen + 5.0 * switching_pen + 2.5 * uncertainty_pen + cap_pen

    # Add small L2 regularization to keep numbers bounded
    fitness_value += 1e-3 * np.sum(alloc ** 2)

    return float(fitness_value)

# ------------------------ PSO implementation ------------------------
def run_pso(T, demand_pred,
            solar_mean, wind_mean, hydro_mean,
            solar_std, wind_std, hydro_std,
            n_particles=N_PARTICLES, max_iters=MAX_ITERS,
            w_init=W_INERTIA, c1=C1, c2=C2):
    dim = 3 * T
    # helper - baseline split 40/35/25 scaled to predicted demand
    base_split = np.array([SOLAR_NOMINAL, WIND_NOMINAL, HYDRO_NOMINAL])
    base_split = base_split / base_split.sum()

    def init_particle():
        alloc = np.zeros((T, 3))
        for i in range(T):
            desired = demand_pred[i] * base_split + np.random.normal(0, 1.5, 3)
            # clip by mean available
            desired[0] = np.clip(desired[0], 0.0, solar_mean[i])
            desired[1] = np.clip(desired[1], 0.0, wind_mean[i])
            desired[2] = np.clip(desired[2], 0.0, hydro_mean[i])
            total = desired.sum()
            max_allowed = min(CAPACITY_NOMINAL, demand_pred[i] * (1.0 + TOLERANCE))
            if total > max_allowed and total > 0:
                desired = desired * (max_allowed / total)
            alloc[i, :] = desired
        return alloc.flatten()

    # initialize
    particles = np.array([init_particle() for _ in range(n_particles)])
    velocities = np.random.normal(0, 1.0, size=particles.shape)
    pbest = particles.copy()
    pbest_scores = np.array([fitness_of_alloc(p, T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std) for p in particles])
    gidx = int(np.argmin(pbest_scores))
    gbest = deepcopy(pbest[gidx])
    gbest_score = pbest_scores[gidx]

    history_best = []
    w = w_init

    for it in range(max_iters):
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            particles[i] = particles[i] + 0.5 * velocities[i]  # damping factor

            # clip positions to some broad bounds for numerical stability
            particles[i] = np.clip(particles[i], -100.0, 300.0)

            score = fitness_of_alloc(particles[i], T, demand_pred, solar_mean, wind_mean, hydro_mean, solar_std, wind_std, hydro_std)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = particles[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest = particles[i].copy()
        history_best.append(gbest_score)
        # slight inertia decay
        w *= 0.999
        # optional simple early stopping (if desired)
        # if it > 20 and np.std(history_best[-10:]) < 1e-4: break

    return gbest.reshape((T, 3)), history_best, gbest_score

# ------------------------ Final adjustment and metrics ------------------------
def finalize_alloc(best_alloc, demand_pred, solar_mean, wind_mean, hydro_mean):
    # Clip final allocation and attempt to bring supply within ±TOLERANCE of demand using hydro if possible
    T = best_alloc.shape[0]
    alloc = best_alloc.copy()
    for i in range(T):
        alloc[i, 0] = np.clip(alloc[i, 0], 0.0, solar_mean[i])
        alloc[i, 1] = np.clip(alloc[i, 1], 0.0, wind_mean[i])
        alloc[i, 2] = np.clip(alloc[i, 2], 0.0, hydro_mean[i])
        total = alloc[i].sum()
        desired_lower = (1.0 - TOLERANCE) * demand_pred[i]
        desired_upper = (1.0 + TOLERANCE) * demand_pred[i]

        # If under lower bound, try to raise hydro (most controllable / stable)
        if total < desired_lower:
            need = desired_lower - total
            add_hydro = min(need, hydro_mean[i] - alloc[i, 2])
            if add_hydro > 0:
                alloc[i, 2] += add_hydro
                total += add_hydro

        # If still under lower bound, proportionally scale others up if possible
        if total < desired_lower:
            # compute available headroom
            headroom = np.array([solar_mean[i] - alloc[i, 0],
                                 wind_mean[i] - alloc[i, 1],
                                 hydro_mean[i] - alloc[i, 2]])
            possible_add = np.sum(np.maximum(0.0, headroom))
            need = desired_lower - total
            if possible_add > 1e-6:
                scale = min(1.0, need / possible_add)
                alloc[i] += np.maximum(0.0, headroom) * scale
                total = alloc[i].sum()

        # If over upper bound, scale down proportionally
        if total > desired_upper and total > 1e-6:
            alloc[i] = alloc[i] * (desired_upper / total)

        # final clip to capacity
        total = alloc[i].sum()
        if total > CAPACITY_NOMINAL and total > 1e-6:
            alloc[i] = alloc[i] * (CAPACITY_NOMINAL / total)

    return alloc

# ------------------------ Main execution ------------------------
def main():
    start_time = time.time()

    env = simulate_environment(T)
    t = env['t']
    demand_true = env['demand_true']
    solar_mean = env['solar_mean']
    wind_mean = env['wind_mean']
    hydro_mean = env['hydro_mean']
    solar_std = env['solar_std']
    wind_std = env['wind_std']
    hydro_std = env['hydro_std']

    demand_pred = predict_demand_moving_average(demand_true, window=MA_WINDOW)

    # Run PSO
    best_alloc_raw, history_best, best_score = run_pso(T, demand_pred,
                                                       solar_mean, wind_mean, hydro_mean,
                                                       solar_std, wind_std, hydro_std,
                                                       n_particles=N_PARTICLES, max_iters=MAX_ITERS,
                                                       w_init=W_INERTIA, c1=C1, c2=C2)

    # Final adjustments & metrics
    best_alloc = finalize_alloc(best_alloc_raw, demand_pred, solar_mean, wind_mean, hydro_mean)
    total_supply = best_alloc.sum(axis=1)
    imbalance_abs = np.abs(total_supply - demand_pred)
    within_tol = (total_supply >= (1.0 - TOLERANCE) * demand_pred) & (total_supply <= (1.0 + TOLERANCE) * demand_pred)
    stability_pct = 100.0 * np.sum(within_tol) / T
    avg_imbalance = float(np.mean(imbalance_abs))
    switching_total = float(np.sum(np.abs(np.vstack([np.zeros((1, 3)), best_alloc[1:, :] - best_alloc[:-1, :]]))))

    # Build DataFrame and sample
    df = pd.DataFrame({
        'time_idx': t,
        'demand_true': np.round(demand_true, 3),
        'demand_pred': np.round(demand_pred, 3),
        'solar_alloc': np.round(best_alloc[:, 0], 3),
        'wind_alloc': np.round(best_alloc[:, 1], 3),
        'hydro_alloc': np.round(best_alloc[:, 2], 3),
        'total_supply': np.round(total_supply, 3),
        'imbalance_abs': np.round(imbalance_abs, 3),
        'within_5pct': within_tol
    })

    df_first10 = df.head(10)

    fitness_summary = {
        'final_fitness': float(best_score),
        'stability_pct_within_5pct': float(stability_pct),
        'avg_abs_imbalance_MW': float(avg_imbalance),
        'total_switching_MW': switching_total,
        'runtime_seconds': float(time.time() - start_time)
    }

    # ------------------------ Visualizations ------------------------
    plt.figure(figsize=(12, 4))
    plt.stackplot(t, best_alloc[:, 0], best_alloc[:, 1], best_alloc[:, 2])
    plt.title('Optimal Energy Allocation Over 24h (10-min intervals)')
    plt.xlabel('Time index (10-min steps)')
    plt.ylabel('MW allocated')
    plt.legend(['Solar', 'Wind', 'Hydro'], loc='upper right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history_best)
    plt.title('Fitness Convergence (lower is better)')
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness so far')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Supply vs Predicted Demand for sample window (first 8 hours)
    sample_end = min(T, 48)
    plt.figure(figsize=(12, 4))
    plt.plot(t[:sample_end], demand_pred[:sample_end], label='Predicted Demand', linewidth=2)
    plt.plot(t[:sample_end], total_supply[:sample_end], label='Total Supply', linewidth=2)
    plt.fill_between(t[:sample_end], 0.95 * demand_pred[:sample_end], 1.05 * demand_pred[:sample_end], alpha=0.2, label='±5% band')
    plt.title('Supply vs Predicted Demand (first 8 hours)')
    plt.xlabel('Time index (10-min steps)')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print outputs
    print("\nSample allocation table (first 10 intervals):")
    print(df_first10.to_string(index=False))

    print("\nFitness summary:")
    for k, v in fitness_summary.items():
        print(f"  {k}: {v}")

    # Return for potential programmatic use
    return {
        'df': df,
        'df_first10': df_first10,
        'best_alloc': best_alloc,
        'history_best': history_best,
        'fitness_summary': fitness_summary
    }

if __name__ == '__main__':
    results = main()
