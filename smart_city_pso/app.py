# app.py
from flask import Flask, render_template, jsonify, request
from simulation import SmartCitySimulation
from threading import Lock
import numpy as np
import time

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global simulation object
simulation = SmartCitySimulation(seed=100, houses=150, buildings=20, evs=30)

# Latest single-step realtime optimized allocation
realtime_opt = {
    "alloc": None,    # [s, w, h]
    "fitness": None,
    "ts": None
}

realtime_lock = Lock()

# Constants
CAPACITY_NOMINAL = 100.0
TOL = 0.05
K_SIGMA = 0.8

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """
    Return a single-step live snapshot. Advances simulation by one step.
    """
    snap = simulation.step()
    total_supply = float(snap["solar_mean"] + snap["wind_mean"] + snap["hydro_mean"])
    snap["total_supply"] = total_supply
    return jsonify(snap)


# -------------------------
# Ultra-fast realtime optimizer (aggressive mode C)
# -------------------------
def ultra_fast_realtime_opt(snapshot, prev_alloc=None,
                            greedy_order=("hydro", "wind", "solar"),
                            micro_pso_particles=6,
                            micro_pso_iters=6):
    """
    Two-stage optimizer:
    1) Greedy exact-fill heuristic that assigns power from most-flexible sources first.
    2) Small micro-PSO refinement to reduce switching/uncertainty cost if needed.

    Returns: dict { alloc: [s, w, h], fitness: float }
    """
    demand = float(snapshot["demand"])
    solar_mean = float(snapshot["solar_mean"])
    wind_mean = float(snapshot["wind_mean"])
    hydro_mean = float(snapshot["hydro_mean"])

    # quick per-source std estimates
    solar_std = 0.08 * solar_mean + 0.5
    wind_std = 0.12 * wind_mean + 0.3
    hydro_std = 0.03 * hydro_mean + 0.1

    # Stage 1: Greedy exact-fill heuristic (aggressive)
    alloc = {"solar": 0.0, "wind": 0.0, "hydro": 0.0}
    remaining = demand

    # Order purposely hydro first (most dispatchable), then wind, then solar
    for src in greedy_order:
        if src == "solar":
            avail = solar_mean
        elif src == "wind":
            avail = wind_mean
        else:
            avail = hydro_mean

        take = min(avail, max(0.0, remaining))
        alloc[src] = take
        remaining -= take
        if remaining <= 1e-6:
            break

    total = alloc["solar"] + alloc["wind"] + alloc["hydro"]

    # If we could not meet demand (remaining > 0), clamp to capacity and leave deficit
    if remaining > 0:
        # if total > capacity again scale down to capacity (unlikely with greedy but safe)
        if total > CAPACITY_NOMINAL and total > 0:
            scale = CAPACITY_NOMINAL / total
            for k in alloc:
                alloc[k] *= scale
            total = sum(alloc.values())
        # compute simple fitness for leftover deficit
        deficit = max(0.0, demand - total)
        base_fitness = deficit**2  # squared deficit penalized strongly
    else:
        base_fitness = 0.0

    # Stage 2: Micro-PSO refinement to reduce switching/uncertainty with minimal cost
    # If deficit is zero and prev_alloc exists, optionally refine to reduce switching.
    def fitness_vec_vec(x):
        # x = [s,w,h]
        s = float(np.clip(x[0], 0.0, solar_mean))
        w = float(np.clip(x[1], 0.0, wind_mean))
        h = float(np.clip(x[2], 0.0, hydro_mean))
        tot = s + w + h

        # imbalance penalty (outside tolerance)
        lower = (1.0 - TOL) * demand
        upper = (1.0 + TOL) * demand
        if tot < lower:
            imbalance = (lower - tot)**2
        elif tot > upper:
            imbalance = (tot - upper)**2
        else:
            imbalance = 0.0

        # switching penalty relative to prev_alloc
        switching_pen = 0.0
        if prev_alloc is not None and len(prev_alloc) == 3:
            changes = np.abs(np.array([s, w, h]) - np.array(prev_alloc))
            switching_pen = np.sum(np.maximum(0.0, changes - 1.0))

        # uncertainty/reserve penalty
        eps = 1e-8
        frac_solar = s / (solar_mean + eps) if solar_mean > eps else 0.0
        frac_wind = w / (wind_mean + eps) if wind_mean > eps else 0.0
        frac_hydro = h / (hydro_mean + eps) if hydro_mean > eps else 0.0
        per_t_std = np.sqrt((frac_solar * solar_std)**2 + (frac_wind * wind_std)**2 + (frac_hydro * hydro_std)**2)
        reserve_gap = max(0.0, demand - (tot - K_SIGMA * per_t_std))
        uncertainty_pen = reserve_gap**2

        cap_pen = max(0.0, tot - CAPACITY_NOMINAL)**2 * 50.0

        val = imbalance + 4.0 * switching_pen + 2.0 * uncertainty_pen + cap_pen
        val += 1e-6 * (s**2 + w**2 + h**2)
        return float(val)

    # If there is a deficit and it's impossible to reach demand, micro-PSO will not help much.
    # We still run a tiny PSO loop around the greedy allocation for a few iterations to refine.
    center = np.array([alloc["solar"], alloc["wind"], alloc["hydro"]], dtype=float)
    dim = 3
    rng = np.random.RandomState(int(time.time()) % 100000)

    # Micro PSO parameters (very small)
    n_particles = max(4, micro_pso_particles)
    max_iters = max(3, micro_pso_iters)
    w = 0.6; c1 = 1.2; c2 = 1.2

    # initialize particles around center with tiny noise
    particles = np.zeros((n_particles, dim))
    velocities = rng.normal(0, 0.2, (n_particles, dim))
    pbest = np.zeros_like(particles)
    pbest_scores = np.full(n_particles, np.inf)

    for i in range(n_particles):
        noise = rng.normal(0, 0.6, dim)
        cand = center + noise
        # clip to means
        cand[0] = np.clip(cand[0], 0.0, solar_mean)
        cand[1] = np.clip(cand[1], 0.0, wind_mean)
        cand[2] = np.clip(cand[2], 0.0, hydro_mean)
        # scale if total exceeds capacity of (1 + TOL) * demand or nominal capacity
        totc = np.sum(cand)
        max_allowed = min(CAPACITY_NOMINAL, (1.0 + TOL) * demand)
        if totc > max_allowed and totc > 0:
            cand = cand * (max_allowed / totc)
        particles[i] = cand
        pbest[i] = cand.copy()
        pbest_scores[i] = fitness_vec_vec(cand)

    gbest_idx = int(np.argmin(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    # run micro-PSO
    ww = w
    for it in range(max_iters):
        for i in range(n_particles):
            r1 = rng.rand(dim); r2 = rng.rand(dim)
            velocities[i] = ww * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            particles[i] = particles[i] + 0.5 * velocities[i]
            # clip and project
            cand = particles[i].copy()
            cand[0] = np.clip(cand[0], 0.0, solar_mean)
            cand[1] = np.clip(cand[1], 0.0, wind_mean)
            cand[2] = np.clip(cand[2], 0.0, hydro_mean)
            totc = np.sum(cand)
            max_allowed = min(CAPACITY_NOMINAL, (1.0 + TOL) * demand)
            if totc > max_allowed and totc > 0:
                cand = cand * (max_allowed / totc)
            particles[i] = cand
            score = fitness_vec_vec(cand)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = cand.copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest = cand.copy()
        ww *= 0.97

    # final allocation is gbest, clipped
    s = float(np.clip(gbest[0], 0.0, solar_mean))
    wv = float(np.clip(gbest[1], 0.0, wind_mean))
    h = float(np.clip(gbest[2], 0.0, hydro_mean))
    total = s + wv + h
    if total > CAPACITY_NOMINAL and total > 0:
        scale = CAPACITY_NOMINAL / total
        s *= scale; wv *= scale; h *= scale

    final_alloc = [s, wv, h]
    final_fitness = float(gbest_score + base_fitness)

    return {"alloc": final_alloc, "fitness": final_fitness}


@app.route("/api/optimize_now", methods=["POST"])
def api_optimize_now():
    """
    Real-time single-step optimization endpoint using the ultra-fast optimizer.
    This uses the current live state but does not advance the live simulation.
    For consistency we fetch a snapshot by advancing and then rolling back one tick.
    """
    # If a snapshot is passed in body, use it (mainly for testing). Otherwise, use current sim state without advancing live stream.
    body = request.get_json(silent=True)
    if body and "snapshot" in body:
        snap = body["snapshot"]
    else:
        snap = simulation.step()
        # roll back the simulation tick so live stream stays continuous
        simulation.t = max(0, simulation.t - 1)

    with realtime_lock:
        prev = realtime_opt["alloc"].copy() if realtime_opt["alloc"] is not None else None

    result = ultra_fast_realtime_opt(snap, prev_alloc=prev,
                                     greedy_order=("hydro", "wind", "solar"),
                                     micro_pso_particles=6,
                                     micro_pso_iters=6)

    with realtime_lock:
        realtime_opt["alloc"] = result["alloc"]
        realtime_opt["fitness"] = result["fitness"]
        realtime_opt["ts"] = time.time()

    return jsonify({
        "status": "ok",
        "alloc": result["alloc"],
        "fitness": result["fitness"]
    })


@app.route("/api/realtime_alloc")
def api_realtime_alloc():
    with realtime_lock:
        return jsonify(realtime_opt)


@app.route("/api/restart", methods=["POST"])
def api_restart():
    global simulation, realtime_opt
    simulation.t = 0
    with realtime_lock:
        realtime_opt = {"alloc": None, "fitness": None, "ts": None}
    return {"status": "ok", "message": "Simulation restarted"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
