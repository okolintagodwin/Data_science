"""
commute.py — Door-to-door commute times for public transport and for the car.

The public-transport journey is built from four explicit legs, because that is
how a commuter actually experiences it:

    home ──walk/bike/feeder bus──▶ boarding station
         ──wait for the first service──▶
         ──ride, incl. any transfers──▶ gateway station
         ──last mile (bus or walk)──▶ site

Modelling the legs separately matters here: the site sits ~2.3 km from the
nearest U-Bahn, so the last mile is a first-order effect, not a rounding error.
Collapsing it into "station-to-station time" would overstate how attractive
public transport is for this specific workplace.

Every employee is evaluated against *all* reachable boarding stations and keeps
the best one, which is what a real journey planner does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .hvv_network import STATIONS, LINES, haversine_km, station_to_office


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------
def _haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance matrix (n x m) in km between two sets of points."""
    r = 6371.0088
    p1 = np.radians(lat1)[:, None]
    p2 = np.radians(lat2)[None, :]
    dp = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _station_headways() -> dict[str, float]:
    """Best (shortest) headway among the lines serving each station."""
    out: dict[str, float] = {}
    for line, (_mode, seq) in LINES.items():
        h = C.LINE_HEADWAY_MIN.get(line.split("_")[0], 20)
        for sid in seq:
            out[sid] = min(out.get(sid, 1e9), h)
    return out


def _reachable_stations(offpeak: bool, shuttle: bool):
    """Station ids, coordinates, time-to-office and transfer count."""
    times, transfers = station_to_office(offpeak=offpeak, shuttle=shuttle)
    heads = _station_headways()
    ids = [s for s in times if np.isfinite(times[s])]
    lat = np.array([STATIONS[s][1] for s in ids])
    lon = np.array([STATIONS[s][2] for s in ids])
    t = np.array([times[s] for s in ids])
    tr = np.array([transfers[s] for s in ids])
    hd = np.array([heads.get(s, 20) for s in ids], dtype=float)
    return ids, lat, lon, t, tr, hd


# ---------------------------------------------------------------------------
# Public transport
# ---------------------------------------------------------------------------
def public_transport_commute(
    emp: pd.DataFrame,
    offpeak_column: str | None = "offpeak_travel",
    shuttle: bool = False,
    allow_bike_access: bool = False,
) -> pd.DataFrame:
    """
    Best door-to-door public-transport journey for every employee.

    Parameters
    ----------
    offpeak_column : column of booleans marking employees who travel outside
        the peak (shift workers). They are routed on a thinned network.
    shuttle : enable the scenario employer shuttle for the last mile.
    allow_bike_access : enable Bike+Ride for the home-to-station leg.

    Returns a frame indexed like `emp` with the journey broken into legs.
    """
    parts = []
    if offpeak_column is not None and offpeak_column in emp.columns:
        groups = [(False, emp[~emp[offpeak_column]]), (True, emp[emp[offpeak_column]])]
    else:
        groups = [(False, emp)]

    for offpeak, sub in groups:
        if sub.empty:
            continue
        ids, slat, slon, t_off, tr, hd = _reachable_stations(offpeak, shuttle)
        d = _haversine_matrix(sub["home_lat"].to_numpy(), sub["home_lon"].to_numpy(),
                              slat, slon) * C.DETOUR_FACTOR

        big = 1e6
        # Access leg: whichever of walk / bike / feeder bus is fastest and legal.
        walk = np.where(d <= C.MAX_WALK_KM, 60 * d / C.WALK_SPEED_KMH, big)
        feeder = np.where(d <= C.MAX_FEEDER_KM,
                          C.FEEDER_BUS_ACCESS_PENALTY_MIN + 60 * d / C.FEEDER_BUS_SPEED_KMH,
                          big)
        # Park+Ride is only credible under three conditions: the driver owns a
        # car, the station is a sensible distance away, and driving there makes
        # progress towards the site rather than away from it. Without the last
        # constraint the router happily drives people past the workplace to
        # reach a fast line.
        st_to_office_km = np.array([
            haversine_km(la, lo, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
            for la, lo in zip(slat, slon)
        ])
        home_to_office_km = np.array([
            haversine_km(r.home_lat, r.home_lon, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
            for r in sub.itertuples()
        ])
        makes_progress = st_to_office_km[None, :] < home_to_office_km[:, None]
        has_car = sub["car_available"].to_numpy()[:, None]
        # Park+Ride is a fallback, not a first choice: it is only offered to
        # employees who have no station within walking or feeder-bus range.
        # Otherwise the router would drive city dwellers halfway to work and
        # then call the remaining leg "public transport".
        no_local_access = (np.minimum(walk, feeder) >= big).all(axis=1)[:, None]
        pr_ok = ((d >= C.MIN_PARK_RIDE_KM) & (d <= C.MAX_PARK_RIDE_KM)
                 & makes_progress & has_car & no_local_access)
        park_ride = np.where(pr_ok,
                             C.PARK_RIDE_PARKING_MIN + 60 * d / C.PARK_RIDE_SPEED_KMH,
                             big)
        # Last resort so that every employee gets a finite (if poor) answer
        # instead of an "unreachable" that would silently drop them from the
        # denominators.
        fallback = np.where(d <= C.FALLBACK_FEEDER_KM,
                            C.FEEDER_BUS_ACCESS_PENALTY_MIN + 6.0
                            + 60 * d / C.FEEDER_BUS_SPEED_KMH, big)
        no_option = (np.minimum(np.minimum(walk, feeder), park_ride) >= big).all(axis=1)[:, None]
        fallback = np.where(no_option, fallback, big)

        options = [walk, feeder, park_ride, fallback]
        labels = ["Walk", "Feeder bus", "Park+Ride", "Long feeder bus"]
        if allow_bike_access:
            options.append(np.where(d <= C.MAX_BIKE_KM, 60 * d / C.BIKE_SPEED_KMH, big))
            labels.append("Bike+Ride")
        stack = np.stack(options)
        access = stack.min(axis=0)
        access_mode_idx = stack.argmin(axis=0)

        head = hd * (C.OFFPEAK_HEADWAY_FACTOR if offpeak else 1.0)
        first_wait = np.minimum(C.FIRST_WAIT_SHARE * head, C.FIRST_WAIT_CAP_MIN)

        total = access + first_wait[None, :] + t_off[None, :]
        total = np.where(access >= big, np.inf, total)
        best = np.argmin(total, axis=1)
        rows = np.arange(len(sub))

        res = pd.DataFrame({
            "boarding_station": [STATIONS[ids[j]][0] for j in best],
            "boarding_station_id": [ids[j] for j in best],
            "access_km": np.round(d[rows, best], 2),
            "access_mode": [labels[access_mode_idx[i, j]] for i, j in zip(rows, best)],
            "access_min": np.round(access[rows, best], 1),
            "wait_min": np.round(first_wait[best], 1),
            "ride_and_lastmile_min": np.round(t_off[best], 1),
            "transfers": tr[best],
            "pt_total_min": np.round(total[rows, best], 1),
        }, index=sub.index)
        res["car_km_still_driven"] = np.where(
            res["access_mode"].eq("Park+Ride"), res["access_km"], 0.0)

        # --- Direct local bus, for employees living close to the site ------
        bus = _direct_local_bus(sub, offpeak)
        use_bus = bus["bus_total_min"] < res["pt_total_min"]
        if use_bus.any():
            res.loc[use_bus, "boarding_station"] = "Local bus stop"
            res.loc[use_bus, "boarding_station_id"] = "local_bus"
            res.loc[use_bus, "access_km"] = 0.4
            res.loc[use_bus, "access_mode"] = "Walk to bus stop"
            res.loc[use_bus, "access_min"] = C.LOCAL_BUS_ACCESS_MIN
            res.loc[use_bus, "wait_min"] = bus.loc[use_bus, "bus_wait_min"]
            res.loc[use_bus, "ride_and_lastmile_min"] = bus.loc[use_bus, "bus_ride_min"]
            res.loc[use_bus, "transfers"] = bus.loc[use_bus, "bus_transfers"]
            res.loc[use_bus, "pt_total_min"] = bus.loc[use_bus, "bus_total_min"].round(1)
            res.loc[use_bus, "car_km_still_driven"] = 0.0

        parts.append(res)

    return pd.concat(parts).reindex(emp.index)


def _direct_local_bus(sub: pd.DataFrame, offpeak: bool) -> pd.DataFrame:
    """
    Direct local-bus journey to the site, available inside LOCAL_BUS_MAX_KM.

    Employees in Norderstedt, Glashütte and Langenhorn are served by local HVV
    bus routes that run straight to the industrial area. Forcing them onto rail
    plus a last-mile bus would invent an interchange that does not exist.
    """
    km = np.array([
        haversine_km(r.home_lat, r.home_lon, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
        for r in sub.itertuples()
    ]) * C.LOCAL_BUS_ROUTE_FACTOR

    head = C.LOCAL_BUS_HEADWAY_MIN * (C.OFFPEAK_HEADWAY_FACTOR if offpeak else 1.0)
    wait = min(C.FIRST_WAIT_SHARE * head, C.FIRST_WAIT_CAP_MIN)
    transfers = (km / C.LOCAL_BUS_ROUTE_FACTOR > C.LOCAL_BUS_DIRECT_KM).astype(int)
    ride = 60 * km / C.LOCAL_BUS_SPEED_KMH + transfers * C.LOCAL_BUS_TRANSFER_MIN
    total = np.where(
        km / C.LOCAL_BUS_ROUTE_FACTOR <= C.LOCAL_BUS_MAX_KM,
        C.LOCAL_BUS_ACCESS_MIN + wait + ride,
        np.inf,
    )
    return pd.DataFrame({
        "bus_wait_min": round(wait, 1),
        "bus_ride_min": np.round(ride, 1),
        "bus_transfers": transfers,
        "bus_total_min": total,
    }, index=sub.index)


# ---------------------------------------------------------------------------
# Car baseline
# ---------------------------------------------------------------------------
def car_commute(emp: pd.DataFrame) -> pd.DataFrame:
    """
    Door-to-door car time and marginal cost.

    Average speed rises with trip length (short trips are all urban arterial,
    long trips are mostly A7/A23), and trips originating south of the Elbe pay
    a river-crossing penalty, which is the dominant congestion effect in Hamburg.
    """
    km = np.array([
        haversine_km(r.home_lat, r.home_lon, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
        for r in emp.itertuples()
    ]) * C.CAR_DETOUR_FACTOR

    highway_share = np.clip((km - 5.0) / 30.0, 0, 1)
    urban_speed = np.where(emp["urbanity"].isin(["urban_core", "urban"]),
                           C.CAR_SPEED_URBAN_KMH, C.CAR_SPEED_SUBURBAN_KMH)
    speed = urban_speed * (1 - highway_share) + C.CAR_SPEED_HIGHWAY_KMH * highway_share

    elbe_penalty = np.where(emp["home_lat"].to_numpy() < 53.53, 9.0, 0.0)

    minutes = (60 * km / speed + C.CAR_ACCESS_EGRESS_MIN
               + C.CAR_PARKING_SEARCH_MIN + elbe_penalty)

    days = emp["commute_days_per_month"].to_numpy()
    monthly_cost = 2 * km * days * C.CAR_COST_PER_KM + C.CAR_MONTHLY_PARKING_EUR

    return pd.DataFrame({
        "car_km_one_way": np.round(km, 2),
        "car_total_min": np.round(minutes, 1),
        "car_cost_eur_month": np.round(monthly_cost, 2),
    }, index=emp.index)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def band(minutes: float) -> str:
    for lo, hi, label in C.COMMUTE_BANDS:
        if lo <= minutes < hi:
            return label
    return C.COMMUTE_BANDS[-1][2]


def build_commutes(emp: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Employee table joined with both commute options and the derived bands."""
    pt = public_transport_commute(emp, **kwargs)
    car = car_commute(emp)
    df = pd.concat([emp, pt, car], axis=1)

    df["commute_band"] = df["pt_total_min"].map(band)
    df["pt_vs_car_min"] = (df["pt_total_min"] - df["car_total_min"]).round(1)
    df["pt_car_time_ratio"] = (df["pt_total_min"] / df["car_total_min"]).round(2)
    df["active_mobility_candidate"] = df["car_km_one_way"] <= C.ACTIVE_MOBILITY_MAX_KM
    return df
