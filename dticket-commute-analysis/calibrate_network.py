"""
calibrate_network.py — fit the network's speed parameters to published HVV times.

The routing graph derives journey times from geometry using two parameters per
mode: a per-stop penalty (braking, dwell, re-acceleration) and a cruise speed.
Rather than guessing them, we fit them to ten published station-to-station
journey times by coordinate descent.

Run:  python calibrate_network.py
Then paste the printed block into src/config.py.
"""

from __future__ import annotations

import itertools

from src import config as C
from src import hvv_network as net

# Published HVV journey times, typical weekday daytime (station to station).
REFERENCE = [
    # Reference journey times, fastest published connection, weekday daytime.
    #
    # The U1 entries are read directly off the HVV/Hochbahn line timetable for
    # U1 (a single train's column): Norderstedt Mitte dep 4:06, Garstedt 4:10,
    # Ochsenzoll 4:12, Langenhorn Markt 4:18, Fuhlsbuettel 4:22, Ohlsdorf 4:25.
    # These are the journeys that matter most here, because every commute to
    # this site arrives on the U1 north corridor.
    ("norderstedt_mitte", "garstedt",        4),
    ("norderstedt_mitte", "ochsenzoll",      6),
    ("norderstedt_mitte", "ohlsdorf",       19),
    ("ochsenzoll",        "ohlsdorf",       13),
    ("garstedt",          "ohlsdorf",       15),
    ("langenhorn_markt",  "ohlsdorf",        7),
    ("fuhlsbuettel",      "ohlsdorf",        3),
    ("norderstedt_mitte", "hauptbahnhof",   39),
    ("ochsenzoll",        "hauptbahnhof",   33),
    # Other corridors, from published HVV journey times (approximate).
    ("altona",            "hauptbahnhof",    9),
    ("barmbek",           "jungfernstieg",  13),
    ("poppenbuettel",     "hauptbahnhof",   22),
    ("elmshorn",          "hauptbahnhof",   30),
    ("ahrensburg",        "hauptbahnhof",   20),
    ("harburg",           "hauptbahnhof",   11),
    ("bergedorf",         "hauptbahnhof",   14),
]


def mae() -> float:
    net.station_to_office.cache_clear()
    return sum(abs(net.station_journey_minutes(a, b) - p)
               for a, b, p in REFERENCE) / len(REFERENCE)


def calibrate(rounds: int = 6) -> float:
    """Coordinate descent over cruise speeds and stop penalties."""
    # Physically plausible ranges per mode. Constraining the search matters:
    # left unconstrained, coordinate descent happily trades an absurd 90 km/h
    # U-Bahn cruise speed against a large stop penalty to shave a minute off
    # the fit. A model that fits ten points with impossible parameters will not
    # generalise to the 1,200 journeys we actually care about.
    speed_grids = {
        "U": [30, 33, 36, 40, 44, 48, 52],      # metro, close stop spacing
        "S": [40, 44, 48, 52, 56, 60, 65],      # suburban rail
        "A": [45, 50, 55, 60, 65],              # AKN
        "R": [60, 70, 80, 90, 100],             # regional / express
        "B": [30, 35, 40, 45, 50],              # express bus
        "b": [16, 19, 22, 25],                  # local bus
    }
    penalty_grid = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50]
    modes = list(C.MODE_CRUISE_KMH)

    best = mae()
    for _ in range(rounds):
        improved = False
        for mode in modes:
            for grid, table in ((speed_grids.get(mode, [C.MODE_CRUISE_KMH[mode]]),
                                 C.MODE_CRUISE_KMH),
                                (penalty_grid, C.MODE_STOP_PENALTY_MIN)):
                current = table[mode]
                for candidate in grid:
                    table[mode] = candidate
                    score = mae()
                    if score < best - 1e-6:
                        best, current, improved = score, candidate, True
                table[mode] = current
        if not improved:
            break
    return best


if __name__ == "__main__":
    print(f"MAE before calibration: {mae():.2f} min")
    best = calibrate()
    print(f"MAE after  calibration: {best:.2f} min\n")

    net.station_to_office.cache_clear()
    worst = 0.0
    for a, b, published in REFERENCE:
        modelled = net.station_journey_minutes(a, b)
        worst = max(worst, abs(modelled - published))
        print(f"  {net.STATIONS[a][0]:<22s} -> {net.STATIONS[b][0]:<16s} "
              f"published {published:3d}   modelled {modelled:6.1f}   "
              f"error {modelled - published:+5.1f}")
    print(f"\n  worst single error: {worst:.1f} min")

    print("\n--- paste into src/config.py ---")
    print("MODE_CRUISE_KMH = {")
    for k, v in C.MODE_CRUISE_KMH.items():
        print(f'    "{k}": {v},')
    print("}")
    print("MODE_STOP_PENALTY_MIN = {")
    for k, v in C.MODE_STOP_PENALTY_MIN.items():
        print(f'    "{k}": {v},')
    print("}")
