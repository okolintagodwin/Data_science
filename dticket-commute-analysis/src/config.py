"""
config.py — Single source of truth for every assumption in the study.

Design principle
----------------
A model is only as credible as its assumptions are visible. Every number that
drives a result lives here, is named, and carries a source/justification comment.
Nothing is hard-coded deeper in the pipeline.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# 1. WORKPLACE
# ----------------------------------------------------------------------------
WORKPLACE = {
    "name": "Johnson & Johnson Medical GmbH",
    "address": "Robert-Koch-Straße 1, 22851 Norderstedt",
    "lat": 53.686952,          # geocoded from public address directory (Norderstedt-Glashütte)
    "lon": 10.046418,
    "district": "Norderstedt-Glashütte",
}

# Core shift / office start times used to pick the service level (headways).
CORE_START_HOUR = 8

# ----------------------------------------------------------------------------
# 2. SYNTHETIC POPULATION
# ----------------------------------------------------------------------------
N_EMPLOYEES = 1200          # order of magnitude of a large medtech manufacturing site
RANDOM_SEED = 42            # full reproducibility

# Residential attraction falls off with distance to the workplace.
# w_zone = population_weight * exp(-distance_km / DECAY_LAMBDA_KM)
# 12 km reproduces the typical German commuter-shed shape (median ~15 km).
DECAY_LAMBDA_KM = 12.0

# ----------------------------------------------------------------------------
# 3. WALK / ACCESS PARAMETERS
# ----------------------------------------------------------------------------
WALK_SPEED_KMH = 4.8
BIKE_SPEED_KMH = 15.0
FEEDER_BUS_SPEED_KMH = 20.0     # incl. stops, urban/suburban feeder services
DETOUR_FACTOR = 1.30            # street network distance / straight-line distance
MAX_WALK_KM = 2.0               # beyond this nobody walks to the station
MAX_BIKE_KM = 6.0               # Bike+Ride catchment
MAX_FEEDER_KM = 5.0             # feeder-bus catchment
FALLBACK_FEEDER_KM = 14.0       # last resort so no employee is 'unreachable'

# Park+Ride: rural employees with no station within walking or bus range drive
# to a station and continue by rail. This is a real and common HVV pattern, and
# without it the model reports "no connection" for whole villages.
MIN_PARK_RIDE_KM = 2.0          # nobody drives 1 km to a station
MAX_PARK_RIDE_KM = 8.0
PARK_RIDE_SPEED_KMH = 50.0
PARK_RIDE_PARKING_MIN = 9.0     # walk to the car, park, walk to the platform
FEEDER_BUS_ACCESS_PENALTY_MIN = 9.0   # walk to the bus stop + wait for the bus

# Direct local-bus option. The HVV local bus network in Norderstedt, Langenhorn
# and Glashütte is dense, so employees living near the site would take a bus
# straight there rather than riding rail out and back. Ignoring this would
# badly understate accessibility for exactly the employees who live closest.
LOCAL_BUS_MAX_KM = 6.0
LOCAL_BUS_SPEED_KMH = 19.0
LOCAL_BUS_ROUTE_FACTOR = 1.35   # buses follow arterials, not straight lines
LOCAL_BUS_ACCESS_MIN = 5.0      # walk to the nearest bus stop
LOCAL_BUS_HEADWAY_MIN = 20.0
LOCAL_BUS_TRANSFER_MIN = 4.0    # one change is normally needed beyond 4 km
LOCAL_BUS_DIRECT_KM = 6.0

# ----------------------------------------------------------------------------
# 4. LINE-HAUL NETWORK PARAMETERS
# ----------------------------------------------------------------------------
# Journey times are built from two physically meaningful parameters per mode:
#   time(edge) = STOP_PENALTY  +  60 * distance_km / CRUISE_SPEED
# The stop penalty covers braking, dwell and re-acceleration; the cruise speed
# is the running speed between stops. Both were fitted by grid search against
# Fitted by coordinate descent (calibrate_network.py) against 16 reference
# journey times, nine of them read directly off the published U1 timetable —
# the corridor every commute to this site arrives on. Mean absolute error
# 1.5 min; the U1 entries are accurate to about 2 min.
MODE_CRUISE_KMH = {
    "U": 56.0,      # U-Bahn
    "S": 70.0,      # S-Bahn
    "A": 65.0,      # AKN
    "R": 80.0,      # Regional (RE/RB)
    "B": 45.0,      # Regional / express bus link
    "b": 24.0,      # Local orbital (cross-suburban) bus
}
MODE_STOP_PENALTY_MIN = {"U": 0.55, "S": 0.65, "A": 1.00, "R": 1.50,
                         "B": 5.00, "b": 3.00}
# Two stations closer together than this are treated as one interchange
# complex joined by a walk (Hauptbahnhof Süd / Nord / S-Bahn, Jungfernstieg /
# Rathaus, Wandsbeker Chaussee / Hasselbrook, Stephansplatz / Dammtor).
INTERCHANGE_WALK_MAX_M = 450.0

ROUTE_SINUOSITY = 1.08          # track distance / straight-line distance


# Peak headway (minutes) per line. Drives waiting time and transfer penalty.
LINE_HEADWAY_MIN = {
    "U1": 5, "U2": 5, "U3": 5, "U4": 5,
    "S1": 10, "S2": 20, "S21": 10, "S3": 10, "S5": 20,
    "A1": 20, "A2": 20, "A3": 60,
    "RE_NW": 60, "RB_NE": 60, "RE_SE": 60, "RE_S": 60,
    "BUS_SE": 30, "BUS_UE": 30, "BUS_SCH": 20, "BUS_GL": 30, "BUS_TA": 60,
    "ORB": 20,
}

TRANSFER_WALK_MIN = 3.0         # platform-to-platform walking at an interchange
FIRST_WAIT_SHARE = 0.40         # commuters time their arrival at the first stop
FIRST_WAIT_CAP_MIN = 6.0
TRANSFER_WAIT_SHARE = 0.50      # transfers are not timed by the passenger
TRANSFER_WAIT_CAP_MIN = 15.0

# Off-peak / early-shift service thinning: headways multiplied by this factor
# for employees whose shift starts before 06:30 or after 18:00.
OFFPEAK_HEADWAY_FACTOR = 1.8

# ----------------------------------------------------------------------------
# 5. LAST MILE TO THE SITE
# ----------------------------------------------------------------------------
# The site sits in the Glashütte industrial area, ~2.3 km from the nearest
# U-Bahn station, so the last mile is an explicit, separately modelled step.
# (station_id, bus_minutes, headway_minutes)
EGRESS_OPTIONS = [
    ("langenhorn_nord", 8.0, 20.0),
    ("ochsenzoll", 9.0, 20.0),
    ("norderstedt_mitte", 12.0, 20.0),
    ("garstedt", 12.0, 30.0),
    ("kiwittsmoor", 8.0, 30.0),
]
EGRESS_WALK_FALLBACK_KMH = WALK_SPEED_KMH   # walking the last mile if no bus fits
# HVV coordinates the feeder bus with the arriving train, so the last-mile wait
# is shorter than a random arrival at the stop would imply.
EGRESS_WAIT_CAP_MIN = 8.0

# Scenario lever: an employer shuttle meeting the U1 at the two nearest
# stations, timed to shift start (10-minute headway during the arrival peak).
SHUTTLE_STOPS = [
    ("ochsenzoll", 6.0, 10.0),
    ("langenhorn_nord", 5.0, 10.0),
    ("norderstedt_mitte", 9.0, 10.0),
    ("garstedt", 8.0, 10.0),
]

# ----------------------------------------------------------------------------
# 6. CAR BASELINE
# ----------------------------------------------------------------------------
CAR_DETOUR_FACTOR = 1.35        # road distance / straight-line distance
CAR_SPEED_URBAN_KMH = 26.0      # inside Hamburg ring, morning peak
CAR_SPEED_SUBURBAN_KMH = 45.0   # suburban arterials
CAR_SPEED_HIGHWAY_KMH = 72.0    # A7/A23 dominated trips
CAR_PARKING_SEARCH_MIN = 2.0    # free on-site parking assumed at an industrial site
CAR_ACCESS_EGRESS_MIN = 4.0     # getting to/from the car, walking across the lot

# Marginal (perceived) cost of driving, €/km. Fixed costs are sunk for someone
# who already owns a car, so only fuel/energy + wear are counted.
CAR_COST_PER_KM = 0.30
CAR_MONTHLY_PARKING_EUR = 0.0   # site parking is free — a key finding, not a bug

# Perceived monthly cost of *keeping a car available* (depreciation, insurance,
# tax, service). Deliberately well below the ADAC full-cost figure (€350-500),
# because drivers systematically under-perceive fixed costs.
# Sunk for someone who already owns one — but a real, unavoidable cost for
# someone who does not and would have to acquire one to drive to work.
# This asymmetry is what makes car ownership the strongest single predictor
# of public-transport adoption, so it is modelled explicitly rather than as a
# hand-tuned preference term.
CAR_FIXED_COST_EUR_MONTH = 150.0

# ----------------------------------------------------------------------------
# 7. TICKET PRICES (2026)
# ----------------------------------------------------------------------------
DTICKET_PRICE_EUR = 63.00       # nationwide price since 1 Jan 2026
# If the employer subsidises >=25 %, federal/state rules grant a further 5 %
# discount, so the employee pays 70 % of the full price.
JOBTICKET_EMPLOYER_SHARE = 0.25
JOBTICKET_EXTRA_DISCOUNT = 0.05
COMMUTE_DAYS_PER_MONTH_FULLTIME = 19.5

# ----------------------------------------------------------------------------
# 8. GENERALISED-COST CHOICE MODEL
# ----------------------------------------------------------------------------
# All disutilities are converted into €/month, then a logit turns the
# generalised-cost gap into an adoption probability.
VALUE_OF_TIME_EUR_H = 10.0      # German commuting value of time, mid-range

# Perceived-time multipliers. A minute spent walking to the station or waiting
# on a platform is consistently valued at more than a minute sitting on a
# train — standard practice in transport appraisal.
ACCESS_TIME_MULTIPLIER = 1.6
WAIT_TIME_MULTIPLIER = 1.8
TRANSFER_PENALTY_MIN = 5.0      # each interchange feels like 5 extra minutes

CAR_COMFORT_PREMIUM_EUR = 12.0  # baseline inertia: door-to-door privacy, boot space
LOGIT_SCALE_EUR = 45.0          # €/month gap that moves the log-odds by 1
RELIABILITY_PENALTY_PER_TRANSFER_EUR = 3.0

# Person-level modifiers to the car comfort premium (€/month). Positive values
# make public transport less attractive for that person.
COMFORT_MODIFIERS = {
    "field_role": 80.0,             # needs the car during the working day
    "shift_worker": 20.0,           # early / late shifts, thin service
    "has_children_under12": 15.0,   # daycare drop-off chains
    "green_attitude": -18.0,        # per unit of the standardised attitude score
    "already_has_dticket": -30.0,   # the ticket is already a sunk decision
}

# Commuters travelling <= this many km are treated as active-mobility candidates
# (walk / bike) rather than public-transport prospects.
ACTIVE_MOBILITY_MAX_KM = 5.0

# ----------------------------------------------------------------------------
# 8b. CALIBRATION ANCHOR
# ----------------------------------------------------------------------------
# An uncalibrated behavioural model produces numbers nobody can defend. We
# therefore anchor the *baseline* to an external statistic and let the model
# explain the deviations around it: 14 % of German commuters travel to work by
# public transport (Mikrozensus, "Weg zur Arbeit"). The calibration solves for
# the one free constant (the car comfort premium) that reproduces this share on
# our synthetic population, so every scenario result is read as a *change*
# against a plausible starting point rather than as an absolute claim.
TARGET_BASELINE_PT_COMMUTE_SHARE = 0.14

# ----------------------------------------------------------------------------
# 8c. TICKET TAKE-UP  (holding a Deutschlandticket != commuting by train)
# ----------------------------------------------------------------------------
# Two different decisions, modelled separately:
#   1. Will this person COMMUTE by public transport?  -> generalised cost
#   2. Will this person BUY the Deutschlandticket?    -> is it worth the money,
#      counting commuting *and* leisure travel
# Conflating them would badly understate take-up of a free or heavily
# subsidised ticket, which people happily accept for weekend and holiday use
# even when they still drive to work.
HVV_FARE_BASE_EUR = 2.60        # single ticket, short distance
HVV_FARE_PER_KM = 0.16          # rises with distance across the HVV rings
HVV_FARE_CAP_EUR = 13.50        # regional ceiling

LEISURE_VALUE_BASE_EUR = 16.0   # value of non-commute travel per month
LEISURE_VALUE_GREEN_EUR = 6.0   # per unit of standardised green attitude
LEISURE_VALUE_URBAN_EUR = 9.0   # city dwellers use the network for everything
LEISURE_VALUE_YOUNG_EUR = 8.0   # under 30
LEISURE_VALUE_NO_CAR_EUR = 18.0 # the network is the only way to travel
TAKEUP_LOGIT_SCALE_EUR = 22.0   # €/month that moves the take-up log-odds by 1
HIGH_POTENTIAL_THRESHOLD = 0.60 # P(take-up) above which we call someone a
                                # high-potential Deutschlandticket user

# ----------------------------------------------------------------------------
# 9. COMMUTE-TIME BANDS
# ----------------------------------------------------------------------------
COMMUTE_BANDS = [(0, 30, "≤ 30 min"), (30, 45, "31–45 min"),
                 (45, 60, "46–60 min"), (60, 10_000, "> 60 min")]

# ----------------------------------------------------------------------------
# 10. SUSTAINABILITY FACTORS (g CO2e per passenger-km)
# ----------------------------------------------------------------------------
CO2_CAR_G_PKM = 145.0           # German passenger-car fleet, real-world, 1.1 occupancy
CO2_PT_G_PKM = 60.0             # blended urban rail / regional rail / bus
WORKING_MONTHS_PER_YEAR = 11.0  # net of holidays
