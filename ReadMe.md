
# Renewable Energy Allocation (PSO-based)

This repository contains a simulation and optimizer that allocates renewable generation (solar, wind, hydro)
to meet predicted electricity demand over a 24-hour horizon (10-minute steps). A Particle Swarm Optimization
(PSO) algorithm is used to produce per-timestep allocations while penalizing imbalance, excessive switching,
uncertainty-driven reserve shortfalls, and capacity violations.

## Key Features

- Simulates time-varying demand, cloud events affecting solar, stochastic wind events, and hydro flow.
- Causal moving-average demand predictor for short-term forecasts.
- PSO-based optimizer with tunable hyperparameters.
- Heuristic post-processing to use hydro flexibility to reduce shortfalls.
- Saves summary plots as PNG files and prints a small allocation table and fitness summary.

## Files

- `renewable_energy.py` — Main script: environment simulation, predictor, fitness function, PSO, plotting/saving.
- `ReadMe.md` — This file.

## Requirements

- Python 3.8+ (developed/tested with Python 3.10)
- Required packages: `numpy`, `pandas`, `matplotlib`

If you want to use the provided virtual environment (`reg` folder) in PowerShell:

```powershell
.\reg\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or create/activate your own venv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib
```

## Quick Run

From the repository root run:

```powershell
python renewable_energy.py
```

What the script does:

- Simulates a 24-hour scenario with 10-minute timesteps (default `T = 144`).
- Predicts demand with a causal moving average.
- Runs PSO to compute resource allocations per timestep.
- Saves three PNG plots to the repository root: `allocation_plot.png`, `fitness_convergence.png`, and `supply_vs_demand.png`.
- Prints a sample allocation table (first 10 intervals) and a fitness summary to the console.

## Configuration / Tuning

Top-level configuration variables (in `renewable_energy.py`) you can edit:

- `T` — number of timesteps (default 144 for 24h @ 10-min intervals).
- `BASE_DEMAND`, `CAPACITY_NOMINAL`, `SOLAR_NOMINAL`, `WIND_NOMINAL`, `HYDRO_NOMINAL` — nominal sizes.
- `N_PARTICLES`, `MAX_ITERS`, `W_INERTIA`, `C1`, `C2` — PSO hyperparameters.
- `MA_WINDOW` — moving-average window for demand prediction.
- `SWITCH_THRESHOLD`, `K_SIGMA`, `TOLERANCE` — switching penalty threshold, safety sigma multiplier, and tolerance band.

For faster experimentation reduce `MAX_ITERS` (e.g. 40) and `N_PARTICLES` (e.g. 20).

## Programmatic Use

`renewable_energy.py` is primarily a script: running it prints summaries and saves plots. The `main()` function
prints outputs and currently does not return the results dictionary. If you prefer programmatic access, I can
refactor `main()` to return the `results` dict (allocations, DataFrame, metrics) for use in other code.

## Outputs and Interpretation

- `allocation_plot.png`: stacked area of resource allocations over time.
- `fitness_convergence.png`: best fitness value per PSO iteration (lower is better).
- `supply_vs_demand.png`: predicted demand vs total supply (first 8 hours by default).
- Console summary: includes `stability_pct_within_5pct` (percentage of timesteps within ±5% of predicted demand),
  average absolute imbalance, total switching, final fitness and runtime.

## Limitations

- Research/prototype code only — not production-grade control software.
- Uncertainty treatment and penalty weights are heuristic and may require retuning.
- No license file is included — add one if you plan to publish.

## Suggested Experiments

- Vary PSO hyperparameters to study convergence and runtime trade-offs.
- Replace the moving-average predictor with a learned model for comparison.
- Add a dispatchable thermal resource and include cost in the objective.

## Next Steps (I can help)

- Add a CLI to set hyperparameters without editing the file.
- Refactor `main()` to return a results dict for programmatic use.
- Add unit tests for the fitness function and finalization logic.
- Add or update `requirements.txt` or add `pyproject.toml`.

---

Last updated: 2025-11-15

Generated from the current `renewable_energy.py` implementation.

