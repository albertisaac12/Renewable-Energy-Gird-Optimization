# pso_optimizer.py
import numpy as np
from copy import deepcopy

class PSOGridOptimizer:

    def __init__(self,
                 series,   # list of snapshots (len T) from simulation.snapshot_series()
                 n_particles=30,
                 max_iters=80,
                 switch_threshold=1.0,
                 tol=0.05,
                 k_sigma=0.8,
                 capacity_nominal=100.0,
                 seed=1234):
        self.series = series
        self.T = len(series)
        self.n_particles = n_particles
        self.max_iters = max_iters
        self.switch_threshold = switch_threshold
        self.tol = tol
        self.k_sigma = k_sigma
        self.capacity_nominal = capacity_nominal
        self.rng = np.random.RandomState(seed)

        # Build arrays
        self.demand_true = np.array([s["demand"] for s in series])
        # For demand prediction: we'll use moving average, but prediction is computed in optimize()
        self.solar_mean = np.array([s["solar_mean"] for s in series])
        self.wind_mean = np.array([s["wind_mean"] for s in series])
        self.hydro_mean = np.array([s["hydro_mean"] for s in series])

        # Estimate per-source uncertainties (heuristic)
        self.solar_std = 0.08 * self.solar_mean + 0.5
        self.wind_std = 0.12 * self.wind_mean + 0.3
        self.hydro_std = 0.03 * self.hydro_mean + 0.1

        # PSO params
        self.w = 0.72
        self.c1 = 1.4
        self.c2 = 1.4

    def predict_demand(self, window=6):
       
        T = self.T
        pred = np.zeros(T)
        for i in range(T):
            s = max(0, i-window)
            pred[i] = np.mean(self.demand_true[s:i+1]) if i>0 else self.demand_true[0]
        pred = 0.6*pred + 0.4*self.demand_true
        return pred

    def fitness(self, flat_alloc, demand_pred):
        alloc = flat_alloc.reshape((self.T, 3))
        # clip
        alloc[:,0] = np.clip(alloc[:,0], 0.0, self.solar_mean)
        alloc[:,1] = np.clip(alloc[:,1], 0.0, self.wind_mean)
        alloc[:,2] = np.clip(alloc[:,2], 0.0, self.hydro_mean)
        total = alloc.sum(axis=1)

        # penalty if outside tolerance
        lower = (1.0 - self.tol) * demand_pred
        upper = (1.0 + self.tol) * demand_pred
        imbalance = np.where((total < lower) | (total > upper),
                             (np.minimum(np.abs(total-lower), np.abs(total-upper)))**2,
                             0.0)
        imbalance_pen = np.sum(imbalance)

        # switching
        changes = np.abs(np.vstack([np.zeros((1,3)), alloc[1:,:] - alloc[:-1,:]]))
        switching_pen = np.sum(np.maximum(0.0, changes - self.switch_threshold))

        # uncertainty penalty: reserve needed to cover k*sigma
        eps = 1e-8
        frac_solar = np.where(self.solar_mean > eps, alloc[:,0]/(self.solar_mean+eps), 0.0)
        frac_wind = np.where(self.wind_mean > eps, alloc[:,1]/(self.wind_mean+eps), 0.0)
        frac_hydro = np.where(self.hydro_mean > eps, alloc[:,2]/(self.hydro_mean+eps), 0.0)

        per_t_std = np.sqrt((frac_solar*self.solar_std)**2 + (frac_wind*self.wind_std)**2 + (frac_hydro*self.hydro_std)**2)
        mean_supply = total
        reserve_gap = np.maximum(0.0, demand_pred - (mean_supply - self.k_sigma*per_t_std))
        uncertainty_pen = np.sum(reserve_gap**2)

        # capacity penalty
        cap_pen = np.sum(np.maximum(0.0, total - self.capacity_nominal)**2) * 50.0

        fitness_val = imbalance_pen + 5.0*switching_pen + 2.5*uncertainty_pen + cap_pen
        # small reg
        fitness_val += 1e-3 * np.sum(alloc**2)
        return float(fitness_val)

    def init_particle(self, demand_pred):
        # initialize particle near base share (40/35/25) scaled to predicted demand
        base_split = np.array([40.0, 35.0, 25.0])
        base_split = base_split / base_split.sum()
        alloc = np.zeros((self.T, 3))
        for i in range(self.T):
            desired = demand_pred[i] * base_split + self.rng.normal(0, 1.5, 3)
            desired[0] = np.clip(desired[0], 0.0, self.solar_mean[i])
            desired[1] = np.clip(desired[1], 0.0, self.wind_mean[i])
            desired[2] = np.clip(desired[2], 0.0, self.hydro_mean[i])
            total = desired.sum()
            max_allowed = min(self.capacity_nominal, demand_pred[i]*(1.0 + self.tol))
            if total > max_allowed and total>0:
                desired = desired * (max_allowed/total)
            alloc[i,:] = desired
        return alloc.flatten()

    def optimize(self):
        demand_pred = self.predict_demand()
        dim = self.T * 3

        # particles & velocities
        particles = np.array([self.init_particle(demand_pred) for _ in range(self.n_particles)])
        velocities = self.rng.normal(0, 1.0, (self.n_particles, dim))

        pbest = particles.copy()
        pbest_scores = np.array([self.fitness(p, demand_pred) for p in particles])
        gbest_idx = int(np.argmin(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        history = [gbest_score]

        w = self.w
        for it in range(self.max_iters):
            for i in range(self.n_particles):
                r1 = self.rng.rand(dim)
                r2 = self.rng.rand(dim)
                velocities[i] = w*velocities[i] + self.c1*r1*(pbest[i]-particles[i]) + self.c2*r2*(gbest-particles[i])
                particles[i] = particles[i] + 0.5*velocities[i]
                # clip broad
                particles[i] = np.clip(particles[i], -100.0, 300.0)
                score = self.fitness(particles[i], demand_pred)
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()
                    if score < gbest_score:
                        gbest_score = score
                        gbest = particles[i].copy()
            history.append(gbest_score)
            w *= 0.999

        best_alloc = gbest.reshape((self.T, 3))
        # Final clipping & small scaling to within capacity/predicted tolerance
        for i in range(self.T):
            best_alloc[i,0] = np.clip(best_alloc[i,0], 0.0, self.solar_mean[i])
            best_alloc[i,1] = np.clip(best_alloc[i,1], 0.0, self.wind_mean[i])
            best_alloc[i,2] = np.clip(best_alloc[i,2], 0.0, self.hydro_mean[i])
            total = best_alloc[i].sum()
            desired = demand_pred[i]
            lower = (1.0 - self.tol) * desired
            upper = min(self.capacity_nominal, (1.0 + self.tol) * desired)
            if total < lower:
                need = lower - total
                add = min(need, self.hydro_mean[i] - best_alloc[i,2])
                best_alloc[i,2] += max(0.0, add)
                total = best_alloc[i].sum()
            if total > upper and total>0:
                best_alloc[i] = best_alloc[i] * (upper/total)
            total = best_alloc[i].sum()
            if total > self.capacity_nominal and total>0:
                best_alloc[i] = best_alloc[i] * (self.capacity_nominal/total)

        return {
            "alloc": best_alloc,
            "fitness": float(gbest_score),
            "history": history,
            "demand_pred": demand_pred
        }
