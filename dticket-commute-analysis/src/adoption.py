"""
adoption.py — Deutschlandticket adoption scoring.

Method: generalised cost
------------------------
Rather than inventing regression coefficients that nobody can audit, every
disutility is converted into the same unit — euros per month — and the two
options are compared head to head:

    GC_pt  = ticket price
           + value of time x perceived public-transport time
           + reliability penalty per interchange
           + personal comfort premium for giving up the car

    GC_car = marginal running cost
           + value of time x car time
           + monthly fixed cost of the car, but only for employees who do
             not already own one

    gap = GC_pt - GC_car                     (positive = public transport loses)
    P(adopt) = 1 / (1 + exp(gap / scale))

Why this is worth the trouble:

* Every input is a quantity a business can argue about — €/month, minutes,
  €/km — not an abstract beta coefficient.
* The 50 % point falls exactly where the two options cost the same, so the
  model is anchored by construction rather than fitted to a target.
* It inverts. For each employee we can ask "how many euros a month would close
  the gap?", which turns the analysis into a budgeting instrument.
* Perceived time weights (walking 1.6x, waiting 1.8x, 5 min per interchange)
  are standard transport-appraisal practice, not free parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Ticket price
# ---------------------------------------------------------------------------
def employee_ticket_price(employer_share: float = 0.0) -> float:
    """
    What the employee actually pays each month.

    German rules grant a further 5 % discount on top of the employer's
    contribution once that contribution reaches 25 %, so a 25 % subsidy costs
    the employee only 70 % of the headline price.
    """
    price = C.DTICKET_PRICE_EUR
    if employer_share >= C.JOBTICKET_EMPLOYER_SHARE:
        # The 5 % discount is granted on the ticket, so it cannot push the
        # employee's share below zero.
        net = price * (1 - C.JOBTICKET_EXTRA_DISCOUNT) - price * employer_share
    else:
        net = price * (1 - employer_share)
    return round(max(net, 0.0), 2)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def perceived_pt_minutes(df: pd.DataFrame) -> pd.Series:
    """Public-transport time as the commuter experiences it, not as the clock does."""
    return (
        df["ride_and_lastmile_min"]
        + C.ACCESS_TIME_MULTIPLIER * df["access_min"]
        + C.WAIT_TIME_MULTIPLIER * df["wait_min"]
        + C.TRANSFER_PENALTY_MIN * df["transfers"]
    )


def comfort_premium(df: pd.DataFrame) -> pd.Series:
    """Person-level reluctance to give up the car, in €/month."""
    m = C.COMFORT_MODIFIERS
    prem = pd.Series(C.CAR_COMFORT_PREMIUM_EUR, index=df.index, dtype=float)
    prem += df["field_role"].astype(float) * m["field_role"]
    prem += df["shift_worker"].astype(float) * m["shift_worker"]
    prem += df["has_children_under12"].astype(float) * m["has_children_under12"]
    prem += df["green_attitude"] * m["green_attitude"]
    prem += df["already_has_dticket"].astype(float) * m["already_has_dticket"]
    return prem


def ticket_value_eur(out: pd.DataFrame) -> pd.Series:
    """
    What a Deutschlandticket is worth to this person, €/month.

    Commuting value = the single fares they would otherwise buy, weighted by how
    likely they are to actually commute by public transport. Leisure value = the
    evenings, weekends and regional trips the ticket also covers, which is why
    people hold it even when they drive to work.
    """
    fare = (C.HVV_FARE_BASE_EUR + C.HVV_FARE_PER_KM * out["car_km_one_way"]).clip(
        upper=C.HVV_FARE_CAP_EUR)
    commute_value = out["pt_commute_probability"] * 2 * out["commute_days_per_month"] * fare

    leisure = pd.Series(C.LEISURE_VALUE_BASE_EUR, index=out.index, dtype=float)
    leisure += out["green_attitude"] * C.LEISURE_VALUE_GREEN_EUR
    leisure += out["urbanity"].eq("urban").astype(float) * C.LEISURE_VALUE_URBAN_EUR
    leisure += (out["age"] < 30).astype(float) * C.LEISURE_VALUE_YOUNG_EUR
    leisure += (~out["car_available"]).astype(float) * C.LEISURE_VALUE_NO_CAR_EUR
    return (commute_value + leisure.clip(lower=0)).round(2)


def score(df: pd.DataFrame, employer_share: float = 0.0,
          comfort_offset: float = 0.0) -> pd.DataFrame:
    """
    Add generalised costs, the adoption probability and the break-even subsidy.

    Parameters
    ----------
    employer_share : fraction of the ticket price paid by J&J (0.0 - 1.0).
    comfort_offset : €/month added to everyone's car comfort premium. This is
        the single calibration handle - see `calibrate()`.

    Returns a copy of `df` with the scoring columns appended.
    """
    out = df.copy()
    days = out["commute_days_per_month"]
    vot_per_min = C.VALUE_OF_TIME_EUR_H / 60.0
    trips = 2 * days                              # there and back

    ticket = employee_ticket_price(employer_share)
    out["ticket_price_eur"] = ticket

    out["pt_perceived_min"] = perceived_pt_minutes(out).round(1)
    out["pt_time_cost_eur"] = (out["pt_perceived_min"] * trips * vot_per_min).round(2)
    out["car_time_cost_eur"] = (out["car_total_min"] * trips * vot_per_min).round(2)

    out["comfort_premium_eur"] = (comfort_premium(out) + comfort_offset).round(2)
    reliability = out["transfers"] * C.RELIABILITY_PENALTY_PER_TRANSFER_EUR

    out["gc_pt_eur"] = (ticket + out["pt_time_cost_eur"] + reliability
                        + out["comfort_premium_eur"]).round(2)

    # The fixed cost of car ownership is sunk for owners, unavoidable for others.
    fixed = np.where(out["car_available"], 0.0, C.CAR_FIXED_COST_EUR_MONTH)
    out["gc_car_eur"] = (out["car_cost_eur_month"] + out["car_time_cost_eur"] + fixed).round(2)

    out["gc_gap_eur"] = (out["gc_pt_eur"] - out["gc_car_eur"]).round(2)
    z = np.clip(out["gc_gap_eur"] / C.LOGIT_SCALE_EUR, -30, 30)   # avoid overflow

    # Decision 1 - would this person COMMUTE by public transport?
    out["pt_commute_probability"] = (1 / (1 + np.exp(z))).round(4)

    # Decision 2 - would this person HOLD a Deutschlandticket? Commuters
    # obviously would; others may still buy it for everything else they travel
    # to, which is exactly why a subsidised ticket reaches beyond the people who
    # give up their car.
    out["ticket_value_eur"] = ticket_value_eur(out)
    out["takeup_gap_eur"] = (ticket - out["ticket_value_eur"]).round(2)
    zt = np.clip(out["takeup_gap_eur"] / C.TAKEUP_LOGIT_SCALE_EUR, -30, 30)
    takeup = 1 / (1 + np.exp(zt))
    # Someone who already holds the ticket does not give it up.
    takeup = np.where(out["already_has_dticket"], np.maximum(takeup, 0.95), takeup)
    out["adoption_probability"] = np.round(
        np.maximum(takeup, out["pt_commute_probability"]), 4)

    # Monthly euros per employee that would bring the two options level.
    # Negative means public transport already wins without any subsidy.
    out["breakeven_subsidy_eur"] = out["gc_gap_eur"].clip(lower=0).round(2)

    out["potential_segment"] = pd.cut(
        out["adoption_probability"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Unlikely", "Persuadable", "Likely", "Very likely"],
    )
    out["high_potential"] = out["adoption_probability"] >= C.HIGH_POTENTIAL_THRESHOLD
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(df: pd.DataFrame, target_share: float | None = None,
              employer_share: float = 0.0) -> float:
    """
    Solve for the car-comfort offset that reproduces an external anchor.

    The behavioural model has exactly one free constant. Rather than tuning it
    by eye, we fix it so that the *baseline* public-transport commute share
    matches the national figure (14 %, Mikrozensus). Everything reported
    afterwards is a change relative to that calibrated starting point.

    Returns the offset in €/month (bisection, tolerance 0.1 pp).
    """
    target = C.TARGET_BASELINE_PT_COMMUTE_SHARE if target_share is None else target_share
    lo, hi = -200.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        share = score(df, employer_share, comfort_offset=mid)["pt_commute_probability"].mean()
        if abs(share - target) < 0.001:
            return round(mid, 2)
        # Higher offset -> public transport less attractive -> lower share.
        if share > target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


# ---------------------------------------------------------------------------
# What actually drives adoption
# ---------------------------------------------------------------------------
def driver_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank the factors that move adoption, as standardised effects.

    A linear model is fitted to the adoption probability using standardised
    predictors, so the coefficients are directly comparable: each one is the
    change in adoption (percentage points) for a one-standard-deviation move in
    that factor. This answers "which lever matters most?" rather than merely
    "which variables are correlated?".
    """
    from sklearn.linear_model import LinearRegression

    # Note on variable choice: `pt_total_min` and `pt_car_time_ratio` are
    # strongly collinear, and including both produces a sign-flipped, nonsense
    # coefficient on raw travel time. The ratio is kept because it is the
    # quantity commuters actually weigh - a 40-minute trip that beats a
    # 20-minute drive is a very different proposition from one that loses to it.
    # Distance to the site is included as a control: without it, the access and
    # time variables partly proxy for "lives far away", which raises the value
    # of the ticket for leisure travel and muddies their coefficients.
    feats = pd.DataFrame({
        "PT time ÷ car time": df["pt_car_time_ratio"],
        "Distance to site (km)": df["car_km_one_way"],
        "Walk/ride to station (km)": df["access_km"],
        "Number of interchanges": df["transfers"],
        "Car available": df["car_available"].astype(float),
        "Days on site per week": df["days_on_site_per_week"],
        "Shift worker": df["shift_worker"].astype(float),
        "Field role (needs car)": df["field_role"].astype(float),
        "Green attitude": df["green_attitude"],
        "Young children at home": df["has_children_under12"].astype(float),
        "Age": df["age"],
    })
    x = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    y = df["adoption_probability"] * 100
    model = LinearRegression().fit(x, y)
    res = pd.DataFrame({
        "factor": feats.columns,
        "effect_pp_per_sd": np.round(model.coef_, 2),
    })
    res["direction"] = np.where(res["effect_pp_per_sd"] >= 0, "increases", "decreases")
    res["abs_effect"] = res["effect_pp_per_sd"].abs()
    res = res.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
    res.attrs["r2"] = round(model.score(x, y), 3)
    return res.reset_index(drop=True)


def sensitivity(emp: pd.DataFrame, build_commutes_fn, comfort_offset: float,
                employer_share: float = 0.0) -> pd.DataFrame:
    """
    Stress-test the headline result against the assumptions it rests on.

    Each row re-runs the whole model with one parameter moved to a plausible
    alternative value. A result that survives this table can be presented to a
    decision-maker; one that does not should be presented as a range.
    """
    import copy

    base_com = build_commutes_fn(emp)
    base = score(base_com, employer_share, comfort_offset)["adoption_probability"].mean() * 100

    variants = [
        ("Value of time  8 €/h (from 10)", "VALUE_OF_TIME_EUR_H", 8.0),
        ("Value of time 13 €/h (from 10)", "VALUE_OF_TIME_EUR_H", 13.0),
        ("Car cost 0.20 €/km (from 0.30)", "CAR_COST_PER_KM", 0.20),
        ("Car cost 0.45 €/km (from 0.30)", "CAR_COST_PER_KM", 0.45),
        ("Choice model sharper (scale 30)", "LOGIT_SCALE_EUR", 30.0),
        ("Choice model flatter (scale 65)", "LOGIT_SCALE_EUR", 65.0),
        ("Leisure value +50 %", "LEISURE_VALUE_BASE_EUR", C.LEISURE_VALUE_BASE_EUR * 1.5),
        ("Leisure value -50 %", "LEISURE_VALUE_BASE_EUR", C.LEISURE_VALUE_BASE_EUR * 0.5),
    ]
    rows = []
    for label, attr, value in variants:
        original = getattr(C, attr)
        setattr(C, attr, value)
        try:
            com = build_commutes_fn(emp)
            got = score(com, employer_share, comfort_offset)["adoption_probability"].mean() * 100
        finally:
            setattr(C, attr, original)
        rows.append({"assumption_changed": label,
                     "adoption_pct": round(got, 1),
                     "change_pp": round(got - base, 1)})
    out = pd.DataFrame(rows).sort_values("change_pp")
    out.attrs["baseline"] = round(base, 1)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> dict:
    """Headline numbers for the executive summary."""
    n = len(df)
    exp_adopters = df["adoption_probability"].sum()          # hold the ticket
    exp_pt_commuters = df["pt_commute_probability"].sum()     # commute by PT
    switchers = df.loc[df["car_available"], "pt_commute_probability"].sum()

    # Only a car owner who actually switches takes a car off the road. Park+Ride
    # commuters still drive to the station, so only the rail portion of their
    # trip counts as avoided car kilometres.
    net_km = (df["car_km_one_way"] - df.get("car_km_still_driven", 0)).clip(lower=0)
    km_saved = (df["pt_commute_probability"] * df["car_available"].astype(float)
                * net_km * 2 * df["commute_days_per_month"]).sum()
    co2_t = (km_saved * (C.CO2_CAR_G_PKM - C.CO2_PT_G_PKM) / 1e6
             * C.WORKING_MONTHS_PER_YEAR)

    return {
        "employees": n,
        "median_pt_min": float(df["pt_total_min"].median()),
        "median_car_min": float(df["car_total_min"].median()),
        "pct_within_30": float((df["pt_total_min"] <= 30).mean() * 100),
        "pct_within_45": float((df["pt_total_min"] <= 45).mean() * 100),
        "pct_within_60": float((df["pt_total_min"] <= 60).mean() * 100),
        "pct_over_60": float((df["pt_total_min"] > 60).mean() * 100),
        "expected_adopters": float(exp_adopters),
        "adoption_rate_pct": float(exp_adopters / n * 100),
        "pt_commute_rate_pct": float(exp_pt_commuters / n * 100),
        "high_potential_employees": int(df["high_potential"].sum())
        if "high_potential" in df else 0,
        "expected_car_switchers": float(switchers),
        "annual_car_km_avoided": float(km_saved * C.WORKING_MONTHS_PER_YEAR),
        "annual_co2_saved_tonnes": float(co2_t),
        "median_breakeven_subsidy": float(df["breakeven_subsidy_eur"].median()),
    }


def band_table(df: pd.DataFrame) -> pd.DataFrame:
    """Employees and adoption by door-to-door commute band."""
    order = [b[2] for b in C.COMMUTE_BANDS]
    g = df.groupby("commute_band", observed=True).agg(
        employees=("employee_id", "count"),
        expected_adopters=("adoption_probability", "sum"),
        mean_adoption_prob=("adoption_probability", "mean"),
        median_pt_min=("pt_total_min", "median"),
        median_car_min=("car_total_min", "median"),
    )
    g = g.reindex([b for b in order if b in g.index])
    g["share_of_employees_pct"] = (g["employees"] / len(df) * 100).round(1)
    g["expected_adopters"] = g["expected_adopters"].round(1)
    g["mean_adoption_prob"] = (g["mean_adoption_prob"] * 100).round(1)
    return g.rename(columns={"mean_adoption_prob": "adoption_rate_pct"})


def zone_table(df: pd.DataFrame, min_employees: int = 8) -> pd.DataFrame:
    """Connectivity and adoption by residential area."""
    g = df.groupby("zone", observed=True).agg(
        employees=("employee_id", "count"),
        median_pt_min=("pt_total_min", "median"),
        median_car_min=("car_total_min", "median"),
        median_access_km=("access_km", "median"),
        median_transfers=("transfers", "median"),
        adoption_rate_pct=("adoption_probability", "mean"),
        median_gap_eur=("gc_gap_eur", "median"),
    )
    g["adoption_rate_pct"] = (g["adoption_rate_pct"] * 100).round(1)
    g["pt_car_ratio"] = (g["median_pt_min"] / g["median_car_min"]).round(2)
    return g[g["employees"] >= min_employees].sort_values("adoption_rate_pct", ascending=False)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def run_scenario(emp: pd.DataFrame, build_commutes_fn, label: str,
                 employer_share: float = 0.0, shuttle: bool = False,
                 allow_bike_access: bool = False,
                 comfort_offset: float = 0.0) -> dict:
    """Rebuild commutes under a set of levers and re-score."""
    com = build_commutes_fn(emp, shuttle=shuttle, allow_bike_access=allow_bike_access)
    sc = score(com, employer_share=employer_share, comfort_offset=comfort_offset)
    s = summarise(sc)
    s.update({
        "scenario": label,
        "employer_share_pct": employer_share * 100,
        "shuttle": shuttle,
        "bike_and_ride": allow_bike_access,
        "employee_price_eur": employee_ticket_price(employer_share),
    })
    return s


def employer_cost_of_subsidy(n_employees_taking: float, employer_share: float) -> float:
    """Annual employer cost of the Jobticket subsidy, in euros."""
    return (C.DTICKET_PRICE_EUR * employer_share * n_employees_taking * 12)
