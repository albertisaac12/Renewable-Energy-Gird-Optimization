# simulation.py
import numpy as np

class SmartCitySimulation:
    """
    Simulates renewable availability and scaled demand between 50 and 100 MW.
    """

    def __init__(self, seed=42, houses=150, buildings=20, evs=30):
        np.random.seed(seed)
        self.seed = seed
        self.houses = houses
        self.buildings = buildings
        self.evs = evs
        self.t = 0
        self.T = 144

        # Nominal generation capacities (MW)
        self.solar_nom = 40.0
        self.wind_nom = 35.0
        self.hydro_nom = 25.0

        rng = np.random.RandomState(seed)

        # Cloud cover profile
        self.cloud = np.ones(self.T)
        cloud_centers = rng.choice(self.T, size=3, replace=False)
        for c in cloud_centers:
            radius = rng.randint(6, 30)
            profile = np.linspace(0.5, 1.0, 2*radius+1)
            s = max(0, c-radius)
            e = min(self.T, c+radius+1)
            seg = profile[(s-(c-radius)):(e-(c-radius))]
            self.cloud[s:e] *= seg

        # Wind factor
        self.wind_factor = rng.normal(1.0, 0.15, self.T)
        storm_centers = rng.choice(self.T, size=2, replace=False)
        for c in storm_centers:
            radius = rng.randint(4, 20)
            profile = np.linspace(0.7, 1.5, 2*radius+1)
            s = max(0, c-radius)
            e = min(self.T, c+radius+1)
            seg = profile[(s-(c-radius)):(e-(c-radius))]
            self.wind_factor[s:e] *= seg
        self.wind_factor = np.clip(self.wind_factor, 0.2, 1.5)

        # Hydro
        self.hydro_flow = (
            0.9 + 
            0.05*np.sin(2*np.pi*np.arange(self.T)/self.T + 0.2) +
            rng.normal(0, 0.02, self.T)
        )
        self.hydro_flow = np.clip(self.hydro_flow, 0.6, 1.0)

        self.rng = rng

    def step(self):
        i = self.t % self.T

        # House demand
        hour = i / self.T
        house_factor = 0.8 + 0.6 * np.sin(2*np.pi*(hour - 0.1))
        house_val = 0.45 * self.houses * house_factor

        # Building demand
        building_factor = 1.0 + 0.7 * np.sin(2*np.pi*(hour + 0.15))
        building_val = 0.6 * self.buildings * building_factor

        # EV demand
        ev_active = self.rng.uniform(0.15, 0.6)
        ev_val = ev_active * self.evs * 1.2

        # ---------------------------
        # Final demand 50 - 100 MW
        # ---------------------------
        raw = house_val + building_val + ev_val
        norm = raw / (0.45*self.houses + 0.6*self.buildings + 0.3*self.evs)
        day_curve = 0.5 + 0.5*np.sin(2*np.pi*(i/self.T) - np.pi/2)

        demand = (
            55 + 
            30*day_curve +
            10*norm +
            self.rng.normal(0, 2.0)
        )
        demand = float(max(50, min(demand, 100)))

        # Renewables
        solar_mean = self.solar_nom * 0.9 * self.cloud[i]
        wind_mean = self.wind_nom * self.wind_factor[i]
        hydro_mean = min(self.hydro_nom * self.hydro_flow[i], self.hydro_nom)

        snap = {
            "t_idx": i,
            "demand": demand,
            "solar_mean": float(solar_mean),
            "wind_mean": float(wind_mean),
            "hydro_mean": float(hydro_mean),
        }

        self.t += 1
        return snap

    def snapshot_series(self):
        t_save = self.t
        out = []
        for _ in range(self.T):
            out.append(self.step())
        self.t = t_save
        return out
