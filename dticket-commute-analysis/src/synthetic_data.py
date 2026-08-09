"""
synthetic_data.py — Generates a privacy-safe synthetic employee population.

No real employee data is used, read or approximated anywhere in this project.
Every record is drawn from published-population priors plus a documented
commuter-shed model, so the dataset is safe to share, version and re-run.

The generator has two layers:

1. WHERE people live — residential zones weighted by
       zone_population  x  exp(-distance_to_site / lambda)
   which reproduces the classic distance-decay shape of a real commuter shed
   (people are far likelier to live near their workplace) without ever
   touching an HR system.

2. WHO they are — person attributes drawn conditionally on where they live and
   what they do (car ownership rises with rurality, shift work concentrates in
   production, hybrid working concentrates in office roles).

Crucially, mode choice is NOT generated here. Attributes are exogenous only;
the commute and the adoption decision are *derived* downstream. Generating the
outcome we then "predict" would be circular.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .hvv_network import haversine_km

# ---------------------------------------------------------------------------
# Residential zones
# (name, lat, lon, radius_km, population_1000s, region, urbanity)
# Populations are rounded public figures; they set relative sampling weights
# only, so small inaccuracies do not move the results.
# ---------------------------------------------------------------------------
ZONES: list[tuple] = [
    # --- Norderstedt & Hamburg north (the natural catchment) --------------
    ("Norderstedt",                 53.7060,  9.9950, 3.5,  82, "Kr. Segeberg",  "suburban"),
    ("Hamburg-Langenhorn",          53.6720, 10.0180, 2.0,  45, "Hamburg",       "urban"),
    ("Hamburg-Fuhlsbüttel",         53.6350, 10.0230, 1.5,  13, "Hamburg",       "urban"),
    ("Hamburg-Ohlsdorf",            53.6250, 10.0300, 1.5,  16, "Hamburg",       "urban"),
    ("Hamburg-Niendorf",            53.6200,  9.9500, 2.2,  42, "Hamburg",       "urban"),
    ("Hamburg-Schnelsen",           53.6350,  9.9200, 2.0,  28, "Hamburg",       "suburban"),
    ("Hamburg-Poppenbüttel",        53.6570, 10.0850, 2.2,  24, "Hamburg",       "suburban"),
    ("Hamburg-Sasel/Wellingsbüttel",53.6600, 10.1100, 2.5,  40, "Hamburg",       "suburban"),
    ("Hamburg-Volksdorf/Duvenstedt",53.6500, 10.1600, 2.8,  35, "Hamburg",       "suburban"),
    ("Hamburg-Bramfeld/Farmsen",    53.6100, 10.0900, 2.2,  70, "Hamburg",       "urban"),

    # --- Hamburg inner city ----------------------------------------------
    ("Hamburg-Winterhude",          53.5950, 10.0100, 1.8,  55, "Hamburg",       "urban_core"),
    ("Hamburg-Barmbek",             53.5850, 10.0400, 1.8,  75, "Hamburg",       "urban_core"),
    ("Hamburg-Eppendorf",           53.5900,  9.9850, 1.5,  35, "Hamburg",       "urban_core"),
    ("Hamburg-Eimsbüttel",          53.5780,  9.9550, 1.8,  60, "Hamburg",       "urban_core"),
    ("Hamburg-Rotherbaum",          53.5720,  9.9850, 1.3,  25, "Hamburg",       "urban_core"),
    ("Hamburg-Uhlenhorst",          53.5700, 10.0250, 1.3,  25, "Hamburg",       "urban_core"),
    ("Hamburg-St. Pauli/Neustadt",  53.5520,  9.9650, 1.5,  40, "Hamburg",       "urban_core"),
    ("Hamburg-Altona/Ottensen",     53.5520,  9.9300, 2.0,  65, "Hamburg",       "urban_core"),
    ("Hamburg-Wandsbek",            53.5750, 10.0700, 2.2,  75, "Hamburg",       "urban"),
    ("Hamburg-Rahlstedt",           53.6050, 10.1550, 2.8,  90, "Hamburg",       "suburban"),
    ("Hamburg-Billstedt/Horn",      53.5450, 10.0900, 2.5,  90, "Hamburg",       "urban"),

    # --- Hamburg west & south --------------------------------------------
    ("Hamburg-Stellingen/Eidelstedt",53.6000, 9.9100, 2.0,  50, "Hamburg",       "urban"),
    ("Hamburg-Bahrenfeld/Lurup",    53.5750,  9.9000, 2.5,  70, "Hamburg",       "urban"),
    ("Hamburg-Blankenese/Osdorf",   53.5750,  9.8000, 3.5,  60, "Hamburg",       "suburban"),
    ("Hamburg-Wilhelmsburg",        53.5050, 10.0100, 2.5,  55, "Hamburg",       "urban"),
    ("Hamburg-Harburg",             53.4600,  9.9800, 3.0,  90, "Hamburg",       "urban"),
    ("Hamburg-Neugraben",           53.4720,  9.8600, 3.0,  45, "Hamburg",       "suburban"),
    ("Hamburg-Bergedorf",           53.4880, 10.2100, 3.0,  60, "Hamburg",       "suburban"),

    # --- Schleswig-Holstein north ----------------------------------------
    ("Henstedt-Ulzburg",            53.7900,  9.9800, 2.5,  28, "Kr. Segeberg",  "suburban"),
    ("Kaltenkirchen",               53.8380,  9.9600, 2.0,  23, "Kr. Segeberg",  "suburban"),
    ("Quickborn",                   53.7290,  9.9040, 2.0,  22, "Kr. Pinneberg", "suburban"),
    ("Ellerau",                     53.7530,  9.9210, 1.2,   6, "Kr. Segeberg",  "rural"),
    ("Tangstedt",                   53.7250, 10.0850, 2.5,   6, "Kr. Stormarn",  "rural"),
    ("Bad Bramstedt",               53.9200,  9.8830, 1.5,  14, "Kr. Segeberg",  "rural"),
    ("Bad Segeberg",                53.9350, 10.3110, 2.0,  17, "Kr. Segeberg",  "rural"),
    ("Neumünster",                  54.0740,  9.9820, 2.5,  80, "Kr. Neumünster","suburban"),

    # --- Schleswig-Holstein west -----------------------------------------
    ("Pinneberg",                   53.6580,  9.8006, 2.0,  44, "Kr. Pinneberg", "suburban"),
    ("Halstenbek/Rellingen",        53.6300,  9.8500, 2.0,  32, "Kr. Pinneberg", "suburban"),
    ("Schenefeld",                  53.6000,  9.8300, 1.5,  19, "Kr. Pinneberg", "suburban"),
    ("Wedel",                       53.5825,  9.7047, 2.0,  25, "Kr. Pinneberg", "suburban"),
    ("Tornesch",                    53.7020,  9.7180, 1.5,  14, "Kr. Pinneberg", "rural"),
    ("Uetersen",                    53.6870,  9.6640, 1.5,  18, "Kr. Pinneberg", "rural"),
    ("Elmshorn",                    53.7530,  9.6540, 2.2,  51, "Kr. Pinneberg", "suburban"),
    ("Barmstedt",                   53.7930,  9.7690, 1.5,  10, "Kr. Pinneberg", "rural"),

    # --- Stormarn / Lauenburg (east) -------------------------------------
    ("Ahrensburg",                  53.6720, 10.2410, 2.0,  34, "Kr. Stormarn",  "suburban"),
    ("Großhansdorf",                53.6620, 10.2955, 1.5,   9, "Kr. Stormarn",  "rural"),
    ("Bargteheide",                 53.7280, 10.2650, 1.5,  16, "Kr. Stormarn",  "rural"),
    ("Bad Oldesloe",                53.8090, 10.3760, 2.0,  25, "Kr. Stormarn",  "rural"),
    ("Reinbek/Glinde",              53.5250, 10.2300, 2.5,  45, "Kr. Stormarn",  "suburban"),

    # --- Niedersachsen (south of the Elbe) -------------------------------
    ("Seevetal/Neu Wulmstorf",      53.4400,  9.9000, 3.0,  55, "Kr. Harburg",   "rural"),
    ("Buxtehude",                   53.4708,  9.7003, 2.0,  40, "Kr. Stade",     "suburban"),
    ("Buchholz i.d.N.",             53.3270,  9.8720, 2.0,  40, "Kr. Harburg",   "rural"),
    ("Winsen (Luhe)",               53.3600, 10.2110, 2.0,  35, "Kr. Harburg",   "rural"),
]

ZONE_COLUMNS = ["zone", "zone_lat", "zone_lon", "radius_km", "pop_k", "region", "urbanity"]

# Car availability by urbanity (share of employees with a car they could use).
CAR_AVAILABILITY = {"urban_core": 0.58, "urban": 0.72, "suburban": 0.88, "rural": 0.94}

# Prior share who already hold a Deutschlandticket (nationally ~17 % of adults,
# strongly skewed towards dense, well-served areas).
DTICKET_PRIOR = {"urban_core": 0.32, "urban": 0.24, "suburban": 0.14, "rural": 0.07}

JOB_FAMILIES = ["Production & Technical", "Office & Commercial", "Lab & R&D", "Field & Sales"]
JOB_SHARES = [0.44, 0.38, 0.12, 0.06]


def zone_table() -> pd.DataFrame:
    """Residential zones with their distance to the workplace and sample weight."""
    z = pd.DataFrame(ZONES, columns=ZONE_COLUMNS)
    z["dist_to_site_km"] = [
        haversine_km(r.zone_lat, r.zone_lon, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
        for r in z.itertuples()
    ]
    z["weight"] = z["pop_k"] * np.exp(-z["dist_to_site_km"] / C.DECAY_LAMBDA_KM)
    z["weight"] /= z["weight"].sum()
    return z


def _sample_in_disc(rng, lat, lon, radius_km, n):
    """Uniform-by-area sampling inside a circular zone."""
    r = radius_km * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    dlat = (r * np.cos(theta)) / 111.32
    dlon = (r * np.sin(theta)) / (111.32 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def generate_employees(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """
    Build the synthetic employee table.

    Returns one row per employee with a home location and the exogenous
    attributes needed for the commute and adoption models.
    """
    n = n or C.N_EMPLOYEES
    seed = C.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    z = zone_table()
    idx = rng.choice(len(z), size=n, p=z["weight"].to_numpy())
    picked = z.iloc[idx].reset_index(drop=True)

    lat, lon = _sample_in_disc(
        rng, picked["zone_lat"].to_numpy(), picked["zone_lon"].to_numpy(),
        picked["radius_km"].to_numpy(), n,
    )

    df = pd.DataFrame({
        "employee_id": [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "zone": picked["zone"],
        "region": picked["region"],
        "urbanity": picked["urbanity"],
        "home_lat": np.round(lat, 5),
        "home_lon": np.round(lon, 5),
    })

    # --- Role, working pattern -------------------------------------------
    df["job_family"] = rng.choice(JOB_FAMILIES, size=n, p=JOB_SHARES)

    is_prod = df["job_family"].eq("Production & Technical")
    is_lab = df["job_family"].eq("Lab & R&D")
    is_office = df["job_family"].eq("Office & Commercial")
    is_field = df["job_family"].eq("Field & Sales")

    # Shift work exists almost exclusively in production.
    df["shift_worker"] = np.where(is_prod, rng.random(n) < 0.55, False)

    shift_choice = rng.choice(["Early (06:00)", "Late (14:00)", "Night (22:00)"],
                              size=n, p=[0.50, 0.35, 0.15])
    df["shift_pattern"] = np.where(df["shift_worker"], shift_choice, "Day (08:00)")
    # Early, late and night shifts all start or end outside the peak service window.
    df["offpeak_travel"] = df["shift_worker"]

    # Days physically on site per week.
    days = np.where(is_office, rng.choice([2, 3, 4, 5], size=n, p=[0.28, 0.34, 0.24, 0.14]),
           np.where(is_lab, rng.choice([3, 4, 5], size=n, p=[0.20, 0.35, 0.45]),
           np.where(is_field, rng.choice([1, 2, 3], size=n, p=[0.45, 0.35, 0.20]), 5)))
    df["days_on_site_per_week"] = days
    df["commute_days_per_month"] = np.round(days / 5 * C.COMMUTE_DAYS_PER_MONTH_FULLTIME, 1)

    # --- Demographics -----------------------------------------------------
    df["age"] = np.clip(rng.normal(42, 11, n), 20, 64).round().astype(int)
    df["has_children_under12"] = rng.random(n) < np.where(
        (df["age"] >= 30) & (df["age"] <= 48), 0.42, 0.13)

    car_p = df["urbanity"].map(CAR_AVAILABILITY).to_numpy()
    car_p = np.clip(car_p + np.where(df["has_children_under12"], 0.06, 0.0), 0, 0.98)
    df["car_available"] = rng.random(n) < car_p
    # Field roles need a vehicle during the working day.
    df.loc[is_field, "car_available"] = True

    # Latent pro-environment attitude, standardised. Younger and more urban
    # employees skew slightly higher, as in German mobility surveys.
    urb_shift = df["urbanity"].map(
        {"urban_core": 0.35, "urban": 0.15, "suburban": -0.10, "rural": -0.30}).to_numpy()
    df["green_attitude"] = np.round(
        rng.normal(0, 1, n) + urb_shift + (42 - df["age"].to_numpy()) / 60, 3)

    df["already_has_dticket"] = rng.random(n) < df["urbanity"].map(DTICKET_PRIOR).to_numpy()

    df["field_role"] = is_field

    return df


def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
