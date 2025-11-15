
# Renewable Energy Allocation (PSO-based)

This repository contains code and tools for simulating a small renewable-heavy grid and optimizing per-timestep
dispatch of solar, wind, and hydro resources using Particle Swarm Optimization (PSO). It includes two main
workflows:

- An analysis/benchmark script (`renewable_energy.py`) that simulates a 24-hour scenario, runs a PSO optimizer,
    saves summary plots and prints performance metrics.
- A small demo web application under the `smart_city_pso/` package providing a lightweight real-time simulation
    and an ultra-fast single-step optimizer exposed via a Flask API.

**Primary goals:** experiment with per-timestep allocation strategies, study trade-offs between tracking
predicted demand, switching costs, uncertainty reserves, and capacity constraints.

## Repository layout

- `renewable_energy.py` — standalone analysis script: environment simulation, moving-average demand predictor,
    full-horizon PSO optimizer, postprocessing, and PNG plots (`allocation_plot.png`, `fitness_convergence.png`,
    `supply_vs_demand.png`).
- `smart_city_pso/` — a small package with:
    - `simulation.py` — `SmartCitySimulation` class generating 144-step snapshot series used by the optimizer.
    - `pso_optimizer.py` — `PSOGridOptimizer` class that performs full-horizon optimization over a snapshot series.
    - `app.py` — Flask application exposing endpoints for live snapshots and an ultra-fast realtime optimizer.
- `requirements.txt` — project dependencies (numpy, pandas, matplotlib, scipy, statsmodels, flask).
- `ReadMe.md` — this file.

## Requirements

- Python 3.8+ (developed/tested with Python 3.10).
- Install dependencies (recommended using a virtual environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer to use the included virtual environment under `reg/`, activate it in PowerShell:

```powershell
.\reg\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

1. Analysis script (offline/full-horizon PSO)

```powershell
python renewable_energy.py
```

This runs a single 24-hour simulation, runs the PSO optimizer (configurable via constants at the top of the
script), saves three PNG plots to the repository root, and prints a sample allocation table and fitness summary.

1. Demo web app (ultra-fast single-step optimization)

```powershell
python -m smart_city_pso.app
```

Then open `http://127.0.0.1:5000/` in your browser. The Flask app provides these JSON endpoints useful for
automation or UI integration:

- `GET /api/state` — advances the simulation by one step and returns a snapshot with `demand`, `solar_mean`,
    `wind_mean`, `hydro_mean`, and other fields.
- `POST /api/optimize_now` — runs the ultra-fast realtime optimizer for a single snapshot (accepts an optional
    JSON body with a `snapshot` to optimize without advancing the live sim). Returns `{ alloc: [s,w,h], fitness }`.
- `GET /api/realtime_alloc` — returns the latest optimization result cached on the server.
- `POST /api/restart` — resets the simulation time to 0.

Example: request a single-step optimization using PowerShell:

```powershell
$resp = Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/api/optimize_now
$resp.alloc
$resp.fitness
```

## Configuration & Important Parameters

Most tuning parameters are defined near the top of each module. Key parameters you will likely edit:

- `T`, `dt_minutes` — horizon length and timestep resolution (in `renewable_energy.py` / `PSOGridOptimizer`).
- `BASE_DEMAND`, `CAPACITY_NOMINAL`, `SOLAR_NOMINAL`, `WIND_NOMINAL`, `HYDRO_NOMINAL` — resource sizes.
- PSO hyperparameters: `N_PARTICLES` / `n_particles`, `MAX_ITERS` / `max_iters`, inertia and cognitive/social
    coefficients (`W_INERTIA`, `C1`, `C2`, or `w`, `c1`, `c2` in optimizer classes).
- `MA_WINDOW` — moving-average window for demand prediction.
- `SWITCH_THRESHOLD`, `K_SIGMA`, `TOLERANCE` — switching penalty threshold, safety sigma multiplier for
    reserve calculation, and acceptable supply/demand tolerance band.

Tuning tip: for faster trials reduce particle count and iterations (e.g., `N_PARTICLES=20`, `MAX_ITERS=40`).

## Outputs

- `allocation_plot.png` — stacked area of solar/wind/hydro allocations over time (saved by `renewable_energy.py`).
- `fitness_convergence.png` — best fitness value per PSO iteration (saved by `renewable_energy.py`).
- `supply_vs_demand.png` — predicted demand vs total supply for the first 8 hours (saved by `renewable_energy.py`).
- Console output: sample allocation table (first 10 rows) and a fitness summary including `stability_pct_within_5pct`.

## Programmatic Use

- `PSOGridOptimizer.optimize()` returns a dict with `alloc`, `fitness`, `history`, and `demand_pred` and is the
    recommended programmatic entry point for full-horizon optimization.
- The Flask `ultra_fast_realtime_opt()` in `app.py` implements a greedy + micro-PSO two-stage optimizer intended
    for low-latency single-step allocation; it returns `{ alloc: [s,w,h], fitness }`.

## Development notes & Suggested Improvements

- The uncertainty model is heuristic (per-source std scaling). Consider replacing with statistical models or
    bootstrapped forecasts for more principled reserve estimates.
- Add a CLI or config file (e.g., `--particles`, `--iters`, `--seed`) instead of editing source constants.
- Add unit tests covering: `fitness` calculation, `finalize_alloc` behavior, and `ultra_fast_realtime_opt` edge cases.
- Consider packaging the `smart_city_pso` module and exposing the optimizer through a lightweight CLI.

## Testing & Reproducibility

- The code uses seeded RNGs for reproducibility; set seeds at the top of modules when running experiments.
- To reproduce specific scenarios, call `SmartCitySimulation.snapshot_series()` and pass the series into
    `PSOGridOptimizer(series, ...)` so you can run deterministic comparisons.

