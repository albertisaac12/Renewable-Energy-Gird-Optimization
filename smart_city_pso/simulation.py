# simulation.py
import numpy as np

class SmartCitySimulation:
    """
    Smart City simulation for 144 timesteps (10 min each).
    Provides renewable availability + demand patterns.
    Modified for:
        - Higher renewable capacity (no underpowered states)
        - Reduced randomness (cleaner demo)
        - Stable demand for first 30 timesteps
        - Dynamic demand afterwards
    """

    def __init__(self, seed=42, houses=150, buildings=20, evs=30):
        np.random.seed(seed)
        self.seed = seed
        self.houses = houses
        self.buildings = buildings
        self.evs = evs
        self.t = 0
        self.T = 144    # 24 hours (10 min per step)

        # Increased renewable capacity (100% renewable + stable demo)
        self.solar_nom = 60.0
        self.wind_nom  = 55.0
        self.hydro_nom = 35.0

        rng = np.random.RandomState(seed)

        # -----------------------------
        # Smooth cloud cover (small randomness)
        # -----------------------------
        base_cloud = 0.85 + 0.10*np.sin(2*np.pi*np.arange(self.T)/self.T)
        noise = rng.normal(0, 0.05, self.T)
        self.cloud = np.clip(base_cloud + noise, 0.65, 1.0)

        # -----------------------------
        # Smooth wind factor
        # -----------------------------
        base_wind = 1.1 + 0.25*np.sin(2*np.pi*(np.arange(self.T)/self.T + 0.2))
        noise = rng.normal(0, 0.08, self.T)
        self.wind_factor = np.clip(base_wind + noise, 0.5, 1.5)

        # -----------------------------
        # Hydro flow stable & smooth
        # -----------------------------
        base_flow = 0.9 + 0.05*np.sin(2*np.pi*(np.arange(self.T)/self.T + 0.1))
        noise = rng.normal(0, 0.01, self.T)
        self.hydro_flow = np.clip(base_flow + noise, 0.75, 1.0)

        self.rng = rng

    def step(self):
        i = self.t % self.T

        # -----------------------------
        # Demand pattern
        # -----------------------------

        # For first 30 timesteps keep demand stable near 95 MW
        # So that PSO can clearly show optimization convergence
        if i < 30:
            demand = 95.0 + self.rng.normal(0, 0.5)

        else:
            # After stability demo, vary demand smoothly (sin wave)
            base = 93.0 + 4.0*np.sin(2*np.pi*(i/self.T - 0.1))
            noise = self.rng.normal(0, 0.8)
            demand = base + noise

        # Clamp final demand
        demand = float(np.clip(demand, 90.0, 100.0))

        # -----------------------------
        # Renewable availability
        # -----------------------------
        solar_mean = self.solar_nom * 0.9 * self.cloud[i]
        wind_mean  = self.wind_nom  * self.wind_factor[i]
        hydro_mean = self.hydro_nom * self.hydro_flow[i]

        snapshot = {
            "t_idx": int(i),
            "demand": float(demand),
            "solar_mean": float(solar_mean),
            "wind_mean": float(wind_mean),
            "hydro_mean": float(hydro_mean),
            "cloud": float(self.cloud[i]),
            "wind_factor": float(self.wind_factor[i]),
            "hydro_flow": float(self.hydro_flow[i]),
        }

        self.t += 1
        return snapshot

    def snapshot_series(self):
        """Return a non-advancing 144-step series (used only for offline tools)."""
        t_save = self.t
        series = []
        for _ in range(self.T):
            series.append(self.step())
        self.t = t_save
        return series
