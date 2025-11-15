# Renewable Energy Allocation (PSO-based)

This repository contains a small simulation and optimizer for allocating renewable generation (solar, wind, hydro)
to meet predicted electricity demand over a 24-hour horizon (10-minute steps). The optimizer uses a simple
Particle Swarm Optimization (PSO) implementation to find allocations that balance supply, stability, switching
costs, uncertainty, and capacity constraints.

## Key Features

- Simulates realistic, time-varying: demand, cloud events (solar), wind events, and hydro flow.
- Causal demand prediction using a moving average predictor.
- PSO optimizer with configurable hyperparameters to optimize allocation per timestep.
- Heuristic final adjustment to favor stable hydro dispatch for shortfalls.
- Produces plots (stacked allocation, fitness convergence, supply vs predicted demand) and a summary table.

## Files

- `renewable_energy.py` — Main script. Contains environment simulation, predictor, fitness function, PSO, and plotting.
- `ReadMe.md` — This file.

## Requirements

- Python 3.8+ (script developed with Python 3.10 in mind)
- Required packages: `numpy`, `pandas`, `matplotlib`

If you have the provided virtual environment (`reg` folder), you can activate it in PowerShell:

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

Run the main script from the repository root:

```powershell
python renewable_energy.py
```

When executed the script will:

- Simulate the environment for 24 hours with 10-minute timesteps (default `T = 144`).
- Predict demand using a causal moving average.
- Run PSO to find an allocation for each timestep.
- Show three interactive plots (allocation stack, fitness convergence, supply vs predicted demand).
- Print a small allocation table (first 10 rows) and a fitness summary to the console.

## Configuration / Tuning

Configuration variables and PSO hyperparameters are at the top of `renewable_energy.py` and can be adjusted:

- `T`, `dt_minutes` — horizon length and timestep resolution.
- `BASE_DEMAND`, `CAPACITY_NOMINAL`, `SOLAR_NOMINAL`, `WIND_NOMINAL`, `HYDRO_NOMINAL` — nominal sizes.
- `N_PARTICLES`, `MAX_ITERS`, `W_INERTIA`, `C1`, `C2` — PSO hyperparameters.
- `MA_WINDOW` — demand predictor window (in timesteps).
- `SWITCH_THRESHOLD`, `K_SIGMA`, `TOLERANCE` — penalties and safety factors.

For faster runs during experimentation, lower `MAX_ITERS` (e.g. 40) and `N_PARTICLES` (e.g. 20).

## Programmatic Use

The `main()` function returns a dictionary with the following keys which you can use if importing the script:

- `df` — full DataFrame of allocations and demand values.
- `df_first10` — sample of first 10 rows.
- `best_alloc` — final allocation array shaped `(T, 3)` (solar, wind, hydro).
- `history_best` — list of best fitness values per PSO iteration.
- `fitness_summary` — dictionary with final metrics and runtime.

Example import usage:

```python
from renewable_energy import main
results = main()
print(results['fitness_summary'])
```

## Outputs and Interpretation

- The plotted stacked area shows how much each resource is allocated over time.
- The fitness convergence plot helps debug PSO progress; lower is better.
- The `stability_pct_within_5pct` metric in the fitness summary reports the percentage
  of timesteps where supply is within ±5% of predicted demand.

## Limitations and Notes

- This is a research/prototyping script, not production-ready grid control code.
- The uncertainty treatment is approximate (heuristic scaling of per-source stds).
- PSO hyperparameters and penalty weights are heuristic and may require retuning for different scenarios.
- No explicit licensing is included in the repo — add a license file if needed.

## Suggested Experiments

- Vary `MAX_ITERS` and `N_PARTICLES` to study convergence/runtime tradeoffs.
- Replace the demand predictor with an ML model and compare metrics.
- Add a conventional generator (dispatchable thermal) and cost-based objective.

## Contact / Next Steps

If you'd like, I can:

- Add a CLI to configure hyperparameters without editing the script.
- Create unit tests for the fitness function and finalization logic.
- Add a minimal `requirements.txt` or `pyproject.toml` (if you prefer).

---

Generated from `renewable_energy.py`.

