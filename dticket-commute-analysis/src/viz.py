"""
viz.py — Charts and the interactive map.

Everything here is presentation only: no numbers are computed that are not
already in the scored dataframe, so the figures can never disagree with the
tables.
"""

from __future__ import annotations

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from . import config as C
from .hvv_network import stations_frame

matplotlib.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

BAND_COLORS = {
    "≤ 30 min": "#1a9850",
    "31–45 min": "#91cf60",
    "46–60 min": "#fdae61",
    "> 60 min": "#d73027",
}
ACCENT = "#C8102E"      # a restrained red for the workplace / highlights
INK = "#22333B"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_commute_bands(df: pd.DataFrame, ax=None):
    """Share of employees in each door-to-door commute band."""
    order = [b[2] for b in C.COMMUTE_BANDS]
    counts = df["commute_band"].value_counts().reindex(order).fillna(0)
    share = counts / counts.sum() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, share.to_numpy(),
                  color=[BAND_COLORS[b] for b in order], edgecolor="white")
    for b, pct, n in zip(bars, share.to_numpy(), counts.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{pct:.1f}%\n({int(n)})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Share of employees (%)")
    ax.set_title("Door-to-door public-transport commute time", loc="left", fontweight="bold")
    ax.set_ylim(0, max(share.max() * 1.3, 10))
    ax.grid(axis="x", visible=False)
    return ax


def plot_zone_ranking(zone_df: pd.DataFrame, n: int = 10, ax=None):
    """Best and worst connected residential areas, by adoption potential."""
    top = zone_df.head(n)
    bottom = zone_df.tail(n)
    combined = pd.concat([bottom, top])
    colors = ["#d73027"] * len(bottom) + ["#1a9850"] * len(top)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ax.barh(combined.index, combined["adoption_rate_pct"], color=colors, edgecolor="white")
    for y, (v, t) in enumerate(zip(combined["adoption_rate_pct"], combined["median_pt_min"])):
        ax.text(v + 0.6, y, f"{v:.0f}%  ·  {t:.0f} min", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Estimated Deutschlandticket adoption (%)")
    ax.set_title("Where public transport works — and where it does not",
                 loc="left", fontweight="bold")
    ax.set_xlim(0, combined["adoption_rate_pct"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    return ax


def plot_scenarios(scen: pd.DataFrame, ax=None):
    """
    Ticket take-up vs. actual modal shift, side by side for each option.

    The gap between the two bars is the story: money buys ticket holders,
    time buys commuters.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    labels = scen["scenario"].tolist()
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, scen["adoption_rate_pct"], h, color="#0f766e",
            label="would hold the ticket", edgecolor="white")
    ax.barh(y - h / 2, scen["pt_commute_rate_pct"], h, color="#f59e0b",
            label="would commute by public transport", edgecolor="white")
    for yi, (a, b) in enumerate(zip(scen["adoption_rate_pct"], scen["pt_commute_rate_pct"])):
        ax.text(a + 1, yi + h / 2, f"{a:.0f}%", va="center", fontsize=8)
        ax.text(b + 1, yi - h / 2, f"{b:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of employees")
    ax.set_xlim(0, max(scen["adoption_rate_pct"]) * 1.18)
    ax.set_title("Holding the ticket is not the same as leaving the car",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    return ax


def plot_time_scatter(df: pd.DataFrame, ax=None):
    """Public transport vs car time, coloured by adoption probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(df["car_total_min"], df["pt_total_min"],
                    c=df["adoption_probability"], cmap="RdYlGn", s=14,
                    alpha=0.8, edgecolors="none", vmin=0, vmax=1)
    lim = [0, min(df["pt_total_min"].quantile(0.99), 150)]
    ax.plot(lim, lim, "--", color=INK, lw=1, label="Public transport = car")
    ax.plot(lim, [2 * v for v in lim], ":", color=INK, lw=1, label="Twice the car time")
    ax.set_xlim(0, df["car_total_min"].quantile(0.99))
    ax.set_ylim(0, lim[1])
    ax.set_xlabel("Car commute (min)")
    ax.set_ylabel("Public-transport commute (min)")
    ax.set_title("The competitiveness gap drives adoption", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.colorbar(sc, ax=ax, label="Adoption probability")
    return ax


def plot_drivers(df: pd.DataFrame, ax=None):
    """Adoption rate across the segments that matter."""
    segs: list[tuple[str, float]] = []
    segs.append(("No car available", df.loc[~df["car_available"], "adoption_probability"].mean()))
    segs.append(("Car available", df.loc[df["car_available"], "adoption_probability"].mean()))
    segs.append(("Already has D-Ticket", df.loc[df["already_has_dticket"], "adoption_probability"].mean()))
    segs.append(("Day shift", df.loc[~df["shift_worker"], "adoption_probability"].mean()))
    segs.append(("Shift worker", df.loc[df["shift_worker"], "adoption_probability"].mean()))
    segs.append(("No young children", df.loc[~df["has_children_under12"], "adoption_probability"].mean()))
    segs.append(("Children under 12", df.loc[df["has_children_under12"], "adoption_probability"].mean()))
    segs.append(("0 transfers", df.loc[df["transfers"] == 0, "adoption_probability"].mean()))
    segs.append(("2+ transfers", df.loc[df["transfers"] >= 2, "adoption_probability"].mean()))
    segs.append(("Walk to station", df.loc[df["access_mode"] == "Walk", "adoption_probability"].mean()))

    labels = [s[0] for s in segs]
    values = [s[1] * 100 for s in segs]
    overall = df["adoption_probability"].mean() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#1a9850" if v >= overall else "#d73027" for v in values]
    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.axvline(overall, color=INK, ls="--", lw=1.2)
    ax.text(overall, -0.9, f" site average {overall:.0f}%", fontsize=8.5, color=INK)
    for y, v in enumerate(values):
        ax.text(v + 0.7, y, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Adoption (%)")
    ax.set_title("Key factors influencing adoption", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return ax


def plot_subsidy_curve(curve: pd.DataFrame, ax=None):
    """Adoption and employer cost as the subsidy rises."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(curve["employer_share_pct"], curve["adoption_rate_pct"],
            marker="o", color="#1a9850", lw=2, label="Adoption")
    ax.set_xlabel("Employer subsidy (%)")
    ax.set_ylabel("Adoption (%)", color="#1a9850")
    ax2 = ax.twinx()
    ax2.plot(curve["employer_share_pct"], curve["employer_cost_eur_year"] / 1000,
             marker="s", color=ACCENT, lw=2, ls="--", label="Employer cost")
    ax2.set_ylabel("Employer cost (€ thousand / year)", color=ACCENT)
    ax2.grid(False)
    ax.set_title("Diminishing returns on subsidy alone", loc="left", fontweight="bold")
    return ax


# ---------------------------------------------------------------------------
# Interactive map
# ---------------------------------------------------------------------------
def build_map(df: pd.DataFrame, include_heatmap: bool = True) -> folium.Map:
    """
    Interactive map: employees by commute band, HVV stations, high-potential
    users, and the workplace.
    """
    m = folium.Map(
        location=[(C.WORKPLACE["lat"] + df["home_lat"].median()) / 2,
                  (C.WORKPLACE["lon"] + df["home_lon"].median()) / 2],
        zoom_start=11, tiles="CartoDB positron", control_scale=True,
    )

    # --- workplace ---------------------------------------------------------
    folium.Marker(
        [C.WORKPLACE["lat"], C.WORKPLACE["lon"]],
        tooltip=f"<b>{C.WORKPLACE['name']}</b><br>{C.WORKPLACE['address']}",
        icon=folium.Icon(color="red", icon="briefcase", prefix="fa"),
    ).add_to(m)
    for radius, label in [(5000, "5 km"), (10000, "10 km"), (20000, "20 km")]:
        folium.Circle(
            [C.WORKPLACE["lat"], C.WORKPLACE["lon"]], radius=radius,
            color="#94a3b8", weight=1, fill=False, dash_array="4,6", tooltip=label,
        ).add_to(m)

    # --- employees by commute band ----------------------------------------
    for band in [b[2] for b in C.COMMUTE_BANDS]:
        sub = df[df["commute_band"] == band]
        if sub.empty:
            continue
        fg = folium.FeatureGroup(name=f"Employees · {band} ({len(sub)})", show=True)
        for r in sub.itertuples():
            folium.CircleMarker(
                [r.home_lat, r.home_lon], radius=3.4,
                color=BAND_COLORS[band], fill=True, fill_opacity=0.75, weight=0,
                tooltip=(f"{r.zone}<br>PT {r.pt_total_min:.0f} min "
                         f"· car {r.car_total_min:.0f} min<br>"
                         f"via {r.boarding_station} ({r.access_mode})<br>"
                         f"Adoption {r.adoption_probability:.0%}"),
            ).add_to(fg)
        fg.add_to(m)

    # --- high-potential users ---------------------------------------------
    hp = df[df["adoption_probability"] >= C.HIGH_POTENTIAL_THRESHOLD]
    if not hp.empty:
        fg = folium.FeatureGroup(name=f"High-potential adopters ({len(hp)})", show=False)
        cluster = MarkerCluster().add_to(fg)
        for r in hp.itertuples():
            folium.CircleMarker(
                [r.home_lat, r.home_lon], radius=5, color="#0f766e",
                fill=True, fill_opacity=0.9, weight=1,
                tooltip=f"{r.zone} · {r.adoption_probability:.0%} · {r.pt_total_min:.0f} min",
            ).add_to(cluster)
        fg.add_to(m)

    # --- HVV stations ------------------------------------------------------
    st = stations_frame()
    fg = folium.FeatureGroup(name=f"HVV stations ({len(st)})", show=True)
    for r in st.itertuples():
        folium.CircleMarker(
            [r.lat, r.lon], radius=2.6, color="#1f2937", fill=True,
            fill_opacity=0.85, weight=0,
            tooltip=f"<b>{r.station}</b><br>{r.lines}<br>{r.dist_to_office_km:.1f} km from site",
        ).add_to(fg)
    fg.add_to(m)

    # --- adoption heat map -------------------------------------------------
    if include_heatmap:
        fg = folium.FeatureGroup(name="Adoption-potential heat map", show=False)
        HeatMap(
            df[["home_lat", "home_lon", "adoption_probability"]].to_numpy().tolist(),
            radius=17, blur=22, min_opacity=0.25,
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html()))
    return m


def _legend_html() -> str:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:11px;'
        f'height:11px;background:{c};border-radius:50%;margin-right:7px"></span>{b}</div>'
        for b, c in BAND_COLORS.items()
    )
    return f"""
    <div style="position:fixed;bottom:26px;left:14px;z-index:9999;background:white;
                padding:11px 13px;border:1px solid #cbd5e1;border-radius:7px;
                font-family:system-ui,sans-serif;font-size:12px;
                box-shadow:0 1px 5px rgba(0,0,0,.18)">
      <div style="font-weight:700;margin-bottom:5px">Door-to-door commute</div>
      {rows}
      <div style="margin-top:6px;color:#64748b;font-size:11px">
        Synthetic employees — no real data
      </div>
    </div>"""
