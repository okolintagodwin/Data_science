# Deutschlandticket commute potential — J&J Medical GmbH, Norderstedt

How attractive would public transport be for employees commuting to
**Robert-Koch-Straße 1, 22851 Norderstedt**, and how many would take up the
Deutschlandticket if J&J promoted or subsidised it?

Everything here runs **offline in about 30 seconds** with a fixed random seed. No API
keys, no network calls, no real employee data.

---

## Run it

```bash
pip install -r requirements.txt
jupyter lab notebooks/deutschlandticket_commute_analysis.ipynb   # Run All
```

The notebook is the deliverable and reads top to bottom. The modules in `src/` do the
work so the notebook stays about the analysis rather than the plumbing.

---

## Headline findings

| | |
|---|---|
| Within 30 min door-to-door by public transport | **14 %** |
| Within 45 min | 24 % |
| Within 60 min | 45 % |
| Over 60 min | **55 %** |
| Median public transport commute | 63 min, vs **27 min** by car (2.3×) |
| Would hold a Deutschlandticket — no support | **42 %** |
| Would hold one — with a 25 % JobTicket subsidy | **55 %** |
| Would actually *commute* by public transport | 14 % → 17 % |
| CO₂ avoided under the JobTicket scenario | ~36 t/year |

**Strongest connectivity:** the U1 corridor (Langenhorn, Fuhlsbüttel, Ohlsdorf,
Norderstedt) and inner Hamburg, where car ownership is low and driving is slow.

**Weakest:** the north-eastern suburbs (Volksdorf, Rahlstedt, Ahrensburg) and the western
belt (Halstenbek, Pinneberg). These have good service *into Hamburg* and almost none
around it — the network is radial, the employer is on the ring.

### The finding that matters most

The site has **no rail station within walking distance**. The nearest U-/S-Bahn stops are
~3 km away, so every public-transport commute ends with a feeder bus that costs 12–16
minutes including the wait.

That is why **subsidy and modal shift come apart**. Making the ticket free roughly doubles
the number of *ticket holders* — but barely moves the number of *public transport
commuters*, because the median employee loses ~35 minutes each way. At €10/hour that is
about €230/month of time, and a €63 ticket cannot buy it back.

**Price is not the binding constraint. Time is.** A shuttle from U1 Ochsenzoll delivers
more modal shift than making the ticket free, at a fraction of the cost per switched
commuter.

---

## What is in here

```
notebooks/deutschlandticket_commute_analysis.ipynb   the deliverable (executed, with outputs)
src/config.py           every assumption in the study, named and sourced
src/synthetic_data.py   45-zone synthetic workforce with distance decay
src/hvv_network.py      189-station HVV graph, line-aware Dijkstra routing
calibrate_network.py    fits network speeds to published journey times
src/commute.py          door-to-door journey builder + car baseline
src/adoption.py         choice model, calibration, drivers, sensitivity
src/viz.py              charts and the Folium map
build_notebook.py       regenerates the notebook from source
data/                   generated synthetic employees and commutes (CSV)
outputs/                summary report, CSVs, figures, interactive map
```

Start with `outputs/summary_report.txt` for the one-screen answer and
`outputs/commute_map.html` for the interactive map.

---

## Method, in one page

**1 · Synthetic population (1,200 employees).** Sampled across 45 residential zones with
weight `population × exp(−distance / 12 km)`, which reproduces the typical German
commuter-shed shape. Attributes are deliberately *correlated*: car ownership depends on
how urban the zone is, shift patterns follow from job family, only office roles work
hybrid. Independent attributes would quietly break the adoption estimate.

**2 · Network model.** The HVV rapid-transit network as a graph — 189 stations, line topology,
per-mode commercial speeds, per-line headways — with Dijkstra over `(station, line)`
states so interchanges are penalised properly. Journey times are *derived* from geometry
rather than hand-entered, so there are very few free parameters.

**3 · Door-to-door routing.** `home → walk/feeder bus/bike/P+R → wait → ride → last-mile
bus → site`. Every reachable boarding station is evaluated and the fastest *complete*
journey wins — frequently not the nearest station. The car alternative is built the same
way, because an adoption model is only as good as the option it competes against.

**4 · Validation.** Station-to-station times are checked against sixteen reference
journeys — nine read directly off the published Hochbahn U1 timetable, the corridor every
commute to this site arrives on. **Mean absolute error 1.7 min** before fitting.

**5 · Choice model.** Every disutility is converted into €/month, so the comparison is
auditable by someone who has never seen a logit:

```
GC_pt  = ticket + VoT × perceived PT time + interchange penalty + comfort premium
GC_car = fuel & wear + VoT × car time + (cost of a car, if they own none)
P      = logistic( −(GC_pt − GC_car) / scale )
```

**Two decisions are modelled separately** — *would I commute by public transport* and
*would I hold a Deutschlandticket* (which also covers evenings, weekends and holidays).
Conflating them badly under-states take-up of a free ticket and over-states cars removed.

**6 · Calibration.** The model has one free constant. It is solved so the baseline
reproduces the national public-transport commute share of 14 % (Mikrozensus). The
required adjustment is **€2.73/month** — the parameters were chosen from appraisal
practice first and reproduce the national figure almost exactly without being bent to
fit. All results are then read as *changes* from a defensible starting point.

**7 · Stress testing.** Eight assumptions (value of time, car cost per km, model
sharpness, leisure value) are each moved to a plausible alternative and the whole model
re-run. The headline moves by a few points; the scenario ranking never inverts.

---

## Honest limitations

| Limitation | Why it is acceptable | Production version |
|---|---|---|
| Synthetic population | Required by the brief | k-anonymised home postcodes from HR |
| Modelled network, not a timetable | MAE 1.7 min against published times, fully reproducible | HVV GTFS + OpenTripPlanner at real shift times |
| One free behavioural parameter | Calibrated to a national anchor, then stress-tested | Stated-preference survey of ~200 staff |
| Straight-line distance × detour factor | Small error relative to 15-min bands | OSM street network via OSMnx |
| Car-owner switching rate (~5 %) | The most uncertain number here | Validate against the staff parking census |

**The percentages are relative instruments.** Trust the ranking of areas and the ranking
of scenarios; treat the absolute numbers as a range, not a forecast.

---

## Key sources

- Deutschlandticket price €63.00/month from 1 Jan 2026 (Verkehrsministerkonferenz,
  18.09.2025); DeutschlandJobTicket €59.85 at a ≥25 % employer subsidy, employee pays
  €44.10, tax-free under §3 Nr. 15 EStG.
- HVV lines 178 / 192 / 278 / 378 / 493 serve *Glashütte Robert-Koch-Straße*.
- Public transport share of commuting trips in Germany ≈ 14 % (Mikrozensus,
  *Weg zur Arbeit*).
- CO₂ factors: 145 g/pkm car, 60 g/pkm local public transport (UBA-style).



## Disclaimer
- Due to time constraint, LLM was used to develop a significant part of this code (The HVV network model, some parts of the data generation, and some of the key assumptions in the config.py file)

- Some of these assumptions were used because they sounded reasonable. e.g DECAY_LAMBDA_KM = 12.0, WALK_SPEED_KMH = 4.8, BIKE_SPEED_KMH = 15.0, FEEDER_BUS_SPEED_KMH = 20.0, DETOUR_FACTOR = 1.30, MAX_WALK_KM = 2.0, MAX_BIKE_KM = 6.0, MAX_FEEDER_KM = 5.0, FALLBACK_FEEDER_KM = 14.0 etc.




