"""
hvv_network.py — A compact, offline model of the HVV rapid-transit network.

Why offline?
------------
A live routing API (HVV/geofox, OpenTripPlanner on the HVV GTFS feed) would give
exact timetable answers, but it needs credentials, is rate-limited, and makes the
notebook non-reproducible. This module instead models the network explicitly:

    station coordinates  +  line topology  +  mode speeds  +  headways

Journey times are *derived* from that geometry rather than hand-entered, which
means the model has very few free parameters — and those parameters are
calibrated against published HVV journey times in the notebook (section 3).

Swapping in the real thing later is a drop-in replacement: keep
`travel_time_to_office()`'s signature and feed it a GTFS-based router.

Coordinates are accurate to roughly ±300 m, which is well inside the resolution
of the walking-access model.
"""

from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx
import pandas as pd

from . import config as C

# ---------------------------------------------------------------------------
# Station coordinates:  id -> (display name, lat, lon)
# ---------------------------------------------------------------------------
STATIONS: dict[str, tuple[str, float, float]] = {
    # --- U1 north (the corridor that serves the site) ---------------------
    "norderstedt_mitte": ("Norderstedt Mitte", 53.7063, 9.9905),
    "richtweg": ("Richtweg", 53.7000, 9.9971),
    "garstedt": ("Garstedt", 53.6950, 9.9999),
    "ochsenzoll": ("Ochsenzoll", 53.6906, 10.0062),
    "kiwittsmoor": ("Kiwittsmoor", 53.6852, 10.0089),
    "langenhorn_nord": ("Langenhorn Nord", 53.6796, 10.0121),
    "langenhorn_markt": ("Langenhorn Markt", 53.6674, 10.0180),
    "fuhlsbuettel_nord": ("Fuhlsbüttel Nord", 53.6470, 10.0246),
    "fuhlsbuettel": ("Fuhlsbüttel", 53.6349, 10.0248),
    "klein_borstel": ("Klein Borstel", 53.6272, 10.0290),
    "ohlsdorf": ("Ohlsdorf", 53.6222, 10.0327),
    "sengelmannstrasse": ("Sengelmannstraße", 53.6117, 10.0248),
    "alsterdorf": ("Alsterdorf", 53.6039, 10.0160),
    "lattenkamp": ("Lattenkamp", 53.5966, 9.9995),
    "hudtwalckerstrasse": ("Hudtwalckerstraße", 53.5919, 9.9942),
    "kellinghusenstrasse": ("Kellinghusenstraße", 53.5872, 9.9905),
    "klosterstern": ("Klosterstern", 53.5789, 9.9971),
    "hallerstrasse": ("Hallerstraße", 53.5688, 9.9928),
    "stephansplatz": ("Stephansplatz", 53.5578, 9.9885),
    "jungfernstieg": ("Jungfernstieg", 53.5535, 9.9938),
    "messberg": ("Meßberg", 53.5479, 10.0006),
    "steinstrasse": ("Steinstraße", 53.5494, 10.0028),
    "hauptbahnhof_sued": ("Hauptbahnhof Süd", 53.5520, 10.0074),
    "lohmuehlenstrasse": ("Lohmühlenstraße", 53.5578, 10.0163),
    "luebecker_strasse": ("Lübecker Straße", 53.5613, 10.0270),
    "wartenau": ("Wartenau", 53.5636, 10.0342),
    "ritterstrasse": ("Ritterstraße", 53.5678, 10.0421),
    "wandsbeker_chaussee": ("Wandsbeker Chaussee", 53.5688, 10.0523),
    "wandsbek_markt": ("Wandsbek Markt", 53.5716, 10.0684),
    "strassburger_strasse": ("Straßburger Straße", 53.5786, 10.0729),
    "alter_teichweg": ("Alter Teichweg", 53.5849, 10.0685),
    "wandsbek_gartenstadt": ("Wandsbek-Gartenstadt", 53.5898, 10.0693),
    "trabrennbahn": ("Trabrennbahn", 53.5960, 10.0850),
    "farmsen": ("Farmsen", 53.6009, 10.1030),
    "berne": ("Berne", 53.6083, 10.1119),
    "meiendorfer_weg": ("Meiendorfer Weg", 53.6155, 10.1170),
    "volksdorf": ("Volksdorf", 53.6428, 10.1583),
    "buchenkamp": ("Buchenkamp", 53.6521, 10.1852),
    "ahrensburg_west": ("Ahrensburg West", 53.6597, 10.2213),
    "ahrensburg_ost": ("Ahrensburg Ost", 53.6614, 10.2426),
    "schmalenbeck": ("Schmalenbeck", 53.6626, 10.2645),
    "kiekut": ("Kiekut", 53.6640, 10.2795),
    "grosshansdorf": ("Großhansdorf", 53.6620, 10.2955),
    "buckhorn": ("Buckhorn", 53.6564, 10.1461),
    "hoisbuettel": ("Hoisbüttel", 53.6672, 10.1391),
    "ohlstedt": ("Ohlstedt", 53.6763, 10.1290),

    # --- U2 ---------------------------------------------------------------
    "niendorf_nord": ("Niendorf Nord", 53.6362, 9.9483),
    "schippelsweg": ("Schippelsweg", 53.6300, 9.9432),
    "joachim_maehl_strasse": ("Joachim-Mähl-Straße", 53.6248, 9.9455),
    "niendorf_markt": ("Niendorf Markt", 53.6182, 9.9509),
    "hagendeel": ("Hagendeel", 53.6055, 9.9509),
    "lutterothstrasse": ("Lutterothstraße", 53.5876, 9.9497),
    "osterstrasse": ("Osterstraße", 53.5813, 9.9556),
    "emilienstrasse": ("Emilienstraße", 53.5762, 9.9608),
    "christuskirche": ("Christuskirche", 53.5716, 9.9663),
    "schlump": ("Schlump", 53.5679, 9.9713),
    "messehallen": ("Messehallen", 53.5589, 9.9758),
    "gaensemarkt": ("Gänsemarkt", 53.5555, 9.9866),
    "hauptbahnhof_nord": ("Hauptbahnhof Nord", 53.5539, 10.0068),
    "berliner_tor": ("Berliner Tor", 53.5533, 10.0250),
    "burgstrasse": ("Burgstraße", 53.5522, 10.0323),
    "hammer_kirche": ("Hammer Kirche", 53.5540, 10.0430),
    "rauhes_haus": ("Rauhes Haus", 53.5556, 10.0525),
    "horner_rennbahn": ("Horner Rennbahn", 53.5560, 10.0685),
    "legienstrasse": ("Legienstraße", 53.5545, 10.0834),
    "billstedt": ("Billstedt", 53.5406, 10.1000),
    "merkenstrasse": ("Merkenstraße", 53.5375, 10.1113),
    "steinfurther_allee": ("Steinfurther Allee", 53.5343, 10.1214),
    "muemmelmannsberg": ("Mümmelmannsberg", 53.5330, 10.1329),

    # --- U3 ---------------------------------------------------------------
    "saarlandstrasse": ("Saarlandstraße", 53.5872, 10.0243),
    "borgweg": ("Borgweg", 53.5876, 10.0122),
    "sierichstrasse": ("Sierichstraße", 53.5883, 10.0000),
    "eppendorfer_baum": ("Eppendorfer Baum", 53.5806, 9.9832),
    "hoheluftbruecke": ("Hoheluftbrücke", 53.5760, 9.9743),
    "sternschanze": ("Sternschanze", 53.5624, 9.9662),
    "feldstrasse": ("Feldstraße", 53.5570, 9.9700),
    "st_pauli": ("St. Pauli", 53.5508, 9.9700),
    "landungsbruecken": ("Landungsbrücken", 53.5459, 9.9694),
    "baumwall": ("Baumwall", 53.5450, 9.9827),
    "roedingsmarkt": ("Rödingsmarkt", 53.5470, 9.9884),
    "rathaus": ("Rathaus", 53.5502, 9.9930),
    "moenckebergstrasse": ("Mönckebergstraße", 53.5514, 10.0006),
    "uhlandstrasse": ("Uhlandstraße", 53.5670, 10.0234),
    "mundsburg": ("Mundsburg", 53.5722, 10.0184),
    "hamburger_strasse": ("Hamburger Straße", 53.5773, 10.0269),
    "dehnhaide": ("Dehnhaide", 53.5817, 10.0356),
    "barmbek": ("Barmbek", 53.5866, 10.0428),
    "habichtstrasse": ("Habichtstraße", 53.5896, 10.0501),

    # --- U4 ---------------------------------------------------------------
    "elbbruecken": ("Elbbrücken", 53.5350, 10.0289),
    "hafencity_universitaet": ("HafenCity Universität", 53.5385, 10.0038),
    "ueberseequartier": ("Überseequartier", 53.5424, 9.9985),

    # --- S1 ---------------------------------------------------------------
    "wedel": ("Wedel", 53.5825, 9.7047),
    "rissen": ("Rissen", 53.5830, 9.7540),
    "suelldorf": ("Sülldorf", 53.5772, 9.7726),
    "iserbrook": ("Iserbrook", 53.5744, 9.7896),
    "blankenese": ("Blankenese", 53.5610, 9.8110),
    "hochkamp": ("Hochkamp", 53.5620, 9.8300),
    "klein_flottbek": ("Klein Flottbek", 53.5629, 9.8552),
    "othmarschen": ("Othmarschen", 53.5588, 9.8817),
    "bahrenfeld": ("Bahrenfeld", 53.5622, 9.9086),
    "altona": ("Altona", 53.5525, 9.9350),
    "koenigstrasse": ("Königstraße", 53.5487, 9.9490),
    "reeperbahn": ("Reeperbahn", 53.5497, 9.9601),
    "stadthausbruecke": ("Stadthausbrücke", 53.5507, 9.9848),
    "hauptbahnhof": ("Hauptbahnhof", 53.5528, 10.0068),
    "landwehr": ("Landwehr", 53.5626, 10.0430),
    "hasselbrook": ("Hasselbrook", 53.5678, 10.0492),
    "friedrichsberg": ("Friedrichsberg", 53.5786, 10.0475),
    "alte_woehr": ("Alte Wöhr", 53.5960, 10.0393),
    "ruebenkamp": ("Rübenkamp", 53.6062, 10.0357),
    "airport": ("Hamburg Airport", 53.6337, 9.9967),
    "kornweg": ("Kornweg", 53.6338, 10.0492),
    "hoheneichen": ("Hoheneichen", 53.6432, 10.0602),
    "wellingsbuettel": ("Wellingsbüttel", 53.6480, 10.0700),
    "poppenbuettel": ("Poppenbüttel", 53.6570, 10.0847),

    # --- S2 / S21 east ----------------------------------------------------
    "rothenburgsort": ("Rothenburgsort", 53.5378, 10.0511),
    "tiefstack": ("Tiefstack", 53.5385, 10.0656),
    "billwerder_moorfleet": ("Billwerder-Moorfleet", 53.5220, 10.0797),
    "mittlerer_landweg": ("Mittlerer Landweg", 53.5085, 10.1096),
    "allermoehe": ("Allermöhe", 53.4977, 10.1416),
    "nettelnburg": ("Nettelnburg", 53.4915, 10.1808),
    "bergedorf": ("Bergedorf", 53.4890, 10.2085),
    "reinbek": ("Reinbek", 53.5106, 10.2472),
    "wohltorf": ("Wohltorf", 53.5222, 10.2790),
    "aumuehle": ("Aumühle", 53.5290, 10.3140),
    "holstenstrasse": ("Holstenstraße", 53.5583, 9.9614),
    "dammtor": ("Dammtor", 53.5606, 9.9896),

    # --- S3 west / north-west --------------------------------------------
    "diebsteich": ("Diebsteich", 53.5648, 9.9192),
    "langenfelde": ("Langenfelde", 53.5798, 9.9174),
    "stellingen": ("Stellingen", 53.5905, 9.9227),
    "eidelstedt": ("Eidelstedt", 53.6060, 9.9040),
    "elbgaustrasse": ("Elbgaustraße", 53.6042, 9.8843),
    "krupunder": ("Krupunder", 53.6146, 9.8563),
    "halstenbek": ("Halstenbek", 53.6270, 9.8390),
    "thesdorf": ("Thesdorf", 53.6440, 9.8236),
    "pinneberg": ("Pinneberg", 53.6580, 9.8006),

    # --- S3 / S5 south ----------------------------------------------------
    "hammerbrook": ("Hammerbrook", 53.5470, 10.0180),
    "veddel": ("Veddel", 53.5271, 10.0175),
    "wilhelmsburg": ("Wilhelmsburg", 53.5030, 10.0132),
    "harburg": ("Harburg", 53.4562, 9.9903),
    "harburg_rathaus": ("Harburg Rathaus", 53.4602, 9.9787),
    "heimfeld": ("Heimfeld", 53.4593, 9.9613),
    "neuwiedenthal": ("Neuwiedenthal", 53.4694, 9.8967),
    "neugraben": ("Neugraben", 53.4723, 9.8586),
    "fischbek": ("Fischbek", 53.4699, 9.8290),
    "neu_wulmstorf": ("Neu Wulmstorf", 53.4670, 9.7940),
    "buxtehude": ("Buxtehude", 53.4708, 9.7003),

    # --- AKN A1 ----------------------------------------------------------
    "hoergensweg": ("Hörgensweg", 53.6150, 9.9070),
    "schnelsen": ("Schnelsen", 53.6330, 9.9160),
    "burgwedel": ("Burgwedel", 53.6480, 9.9130),
    "boenningstedt": ("Bönningstedt", 53.6720, 9.9200),
    "hasloh": ("Hasloh", 53.6980, 9.9180),
    "quickborn_sued": ("Quickborn Süd", 53.7180, 9.9130),
    "quickborn": ("Quickborn", 53.7290, 9.9040),
    "ellerau": ("Ellerau", 53.7530, 9.9210),
    "tanneneck": ("Tanneneck", 53.7660, 9.9560),
    "ulzburg_sued": ("Ulzburg Süd", 53.7760, 9.9860),
    "henstedt_ulzburg": ("Henstedt-Ulzburg", 53.7900, 9.9810),
    "kaltenkirchen_sued": ("Kaltenkirchen Süd", 53.8250, 9.9640),
    "kaltenkirchen": ("Kaltenkirchen", 53.8380, 9.9600),
    "lentfoehrden": ("Lentföhrden", 53.8940, 9.9130),
    "bad_bramstedt": ("Bad Bramstedt", 53.9200, 9.8830),

    # --- AKN A2 (Norderstedt <-> Ulzburg Süd) -----------------------------
    "quickborner_strasse": ("Quickborner Straße", 53.7130, 9.9880),
    "moorbekhalle": ("Moorbekhalle", 53.7190, 9.9860),
    "friedrichsgabe": ("Friedrichsgabe", 53.7280, 9.9840),
    "haslohfurth": ("Haslohfurth", 53.7430, 9.9820),
    "meeschensee": ("Meeschensee", 53.7620, 9.9840),

    # --- AKN A3 (Ulzburg Süd <-> Elmshorn) --------------------------------
    "alveslohe": ("Alveslohe", 53.7910, 9.9070),
    "bevern": ("Bevern", 53.7920, 9.8300),
    "barmstedt": ("Barmstedt", 53.7930, 9.7690),
    "langeln": ("Langeln", 53.7860, 9.7300),
    "bokholt": ("Bokholt", 53.7770, 9.7060),
    "sparrieshoop": ("Sparrieshoop", 53.7660, 9.6840),
    "elmshorn": ("Elmshorn", 53.7530, 9.6540),

    # --- Regional rail ----------------------------------------------------
    "tornesch": ("Tornesch", 53.7020, 9.7180),
    "prisdorf": ("Prisdorf", 53.6740, 9.7580),
    "itzehoe": ("Itzehoe", 53.9250, 9.5170),
    "neumuenster": ("Neumünster", 54.0740, 9.9820),
    "ahrensburg": ("Ahrensburg (DB)", 53.6720, 10.2410),
    "bargteheide": ("Bargteheide", 53.7280, 10.2650),
    "bad_oldesloe": ("Bad Oldesloe", 53.8090, 10.3760),
    "schwarzenbek": ("Schwarzenbek", 53.5030, 10.4820),
    "winsen": ("Winsen (Luhe)", 53.3600, 10.2110),
    "buchholz": ("Buchholz i.d.N.", 53.3270, 9.8720),

    # --- Bus-only towns (deliberately included: they show the weak spots) --
    "bad_segeberg": ("Bad Segeberg (Bus)", 53.9350, 10.3110),
    "uetersen": ("Uetersen (Bus)", 53.6870, 9.6640),
    "schenefeld": ("Schenefeld (Bus)", 53.6000, 9.8300),
    "glinde": ("Glinde (Bus)", 53.5410, 10.2100),
    "tangstedt": ("Tangstedt (Bus)", 53.7250, 10.0850),
}

# ---------------------------------------------------------------------------
# Line topology:  line -> (mode letter, ordered station sequence)
# ---------------------------------------------------------------------------
LINES: dict[str, tuple[str, list[str]]] = {
    "U1": ("U", [
        "norderstedt_mitte", "richtweg", "garstedt", "ochsenzoll", "kiwittsmoor",
        "langenhorn_nord", "langenhorn_markt", "fuhlsbuettel_nord", "fuhlsbuettel",
        "klein_borstel", "ohlsdorf", "sengelmannstrasse", "alsterdorf", "lattenkamp",
        "hudtwalckerstrasse", "kellinghusenstrasse", "klosterstern", "hallerstrasse",
        "stephansplatz", "jungfernstieg", "messberg", "steinstrasse",
        "hauptbahnhof_sued", "lohmuehlenstrasse", "luebecker_strasse", "wartenau",
        "ritterstrasse", "wandsbeker_chaussee", "wandsbek_markt",
        "strassburger_strasse", "alter_teichweg", "wandsbek_gartenstadt",
        "trabrennbahn", "farmsen", "berne", "meiendorfer_weg", "volksdorf",
        "buchenkamp", "ahrensburg_west", "ahrensburg_ost", "schmalenbeck",
        "kiekut", "grosshansdorf",
    ]),
    "U1_ohlstedt": ("U", ["volksdorf", "buckhorn", "hoisbuettel", "ohlstedt"]),
    "U2": ("U", [
        "niendorf_nord", "schippelsweg", "joachim_maehl_strasse", "niendorf_markt",
        "hagendeel", "lutterothstrasse", "osterstrasse", "emilienstrasse",
        "christuskirche", "schlump", "messehallen", "gaensemarkt", "jungfernstieg",
        "hauptbahnhof_nord", "berliner_tor", "burgstrasse", "hammer_kirche",
        "rauhes_haus", "horner_rennbahn", "legienstrasse", "billstedt",
        "merkenstrasse", "steinfurther_allee", "muemmelmannsberg",
    ]),
    "U3": ("U", [
        "wandsbek_gartenstadt", "habichtstrasse", "barmbek", "saarlandstrasse",
        "borgweg", "sierichstrasse", "kellinghusenstrasse", "eppendorfer_baum",
        "hoheluftbruecke", "schlump", "sternschanze", "feldstrasse", "st_pauli",
        "landungsbruecken", "baumwall", "roedingsmarkt", "rathaus",
        "moenckebergstrasse", "hauptbahnhof_sued", "berliner_tor",
        "luebecker_strasse", "uhlandstrasse", "mundsburg", "hamburger_strasse",
        "dehnhaide", "barmbek",
    ]),
    "U4": ("U", [
        "elbbruecken", "hafencity_universitaet", "ueberseequartier", "jungfernstieg",
        "hauptbahnhof_nord", "berliner_tor", "burgstrasse", "hammer_kirche",
        "rauhes_haus", "horner_rennbahn", "legienstrasse", "billstedt",
    ]),
    "S1": ("S", [
        "wedel", "rissen", "suelldorf", "iserbrook", "blankenese", "hochkamp",
        "klein_flottbek", "othmarschen", "bahrenfeld", "altona", "koenigstrasse",
        "reeperbahn", "landungsbruecken", "stadthausbruecke", "jungfernstieg",
        "hauptbahnhof", "berliner_tor", "landwehr", "hasselbrook", "friedrichsberg",
        "barmbek", "alte_woehr", "ruebenkamp", "ohlsdorf", "kornweg", "hoheneichen",
        "wellingsbuettel", "poppenbuettel",
    ]),
    "S1_airport": ("S", ["ohlsdorf", "airport"]),
    "S2": ("S", [
        "altona", "koenigstrasse", "reeperbahn", "landungsbruecken",
        "stadthausbruecke", "jungfernstieg", "hauptbahnhof", "berliner_tor",
        "rothenburgsort", "tiefstack", "billwerder_moorfleet", "mittlerer_landweg",
        "allermoehe", "nettelnburg", "bergedorf", "reinbek", "wohltorf", "aumuehle",
    ]),
    "S21": ("S", [
        "elbgaustrasse", "eidelstedt", "stellingen", "langenfelde", "holstenstrasse",
        "sternschanze", "dammtor", "hauptbahnhof", "berliner_tor", "rothenburgsort",
        "tiefstack", "billwerder_moorfleet", "mittlerer_landweg", "allermoehe",
        "nettelnburg", "bergedorf",
    ]),
    "S3": ("S", [
        "pinneberg", "thesdorf", "halstenbek", "krupunder", "elbgaustrasse",
        "eidelstedt", "stellingen", "langenfelde", "diebsteich", "altona",
        "koenigstrasse", "reeperbahn", "landungsbruecken", "stadthausbruecke",
        "jungfernstieg", "hauptbahnhof", "hammerbrook", "elbbruecken", "veddel",
        "wilhelmsburg", "harburg", "harburg_rathaus", "heimfeld", "neuwiedenthal",
        "neugraben",
    ]),
    "S5": ("S", [
        "neugraben", "fischbek", "neu_wulmstorf", "buxtehude",
    ]),
    "A1": ("A", [
        "eidelstedt", "hoergensweg", "schnelsen", "burgwedel", "boenningstedt",
        "hasloh", "quickborn_sued", "quickborn", "ellerau", "tanneneck",
        "ulzburg_sued", "henstedt_ulzburg", "kaltenkirchen_sued", "kaltenkirchen",
        "lentfoehrden", "bad_bramstedt",
    ]),
    "A2": ("A", [
        "norderstedt_mitte", "quickborner_strasse", "moorbekhalle", "friedrichsgabe",
        "haslohfurth", "meeschensee", "ulzburg_sued",
    ]),
    "A3": ("A", [
        "ulzburg_sued", "alveslohe", "bevern", "barmstedt", "langeln", "bokholt",
        "sparrieshoop", "elmshorn",
    ]),
    "RE_NW": ("R", [
        "hauptbahnhof", "dammtor", "altona", "pinneberg", "prisdorf", "tornesch",
        "elmshorn", "itzehoe",
    ]),
    "RE_NW2": ("R", ["elmshorn", "neumuenster"]),
    "RB_NE": ("R", [
        "hauptbahnhof", "hasselbrook", "ahrensburg", "bargteheide", "bad_oldesloe",
    ]),
    "RE_SE": ("R", ["hauptbahnhof", "bergedorf", "schwarzenbek"]),
    "RE_S": ("R", ["hauptbahnhof", "harburg", "winsen", "buchholz"]),

    # Regional bus links for towns without a rail head — these are what make
    # the "poorly connected" areas visible in the results.
    "BUS_SE": ("B", ["bad_segeberg", "henstedt_ulzburg"]),
    "BUS_UE": ("B", ["uetersen", "tornesch"]),
    "BUS_SCH": ("B", ["schenefeld", "elbgaustrasse"]),
    "BUS_GL": ("B", ["glinde", "billstedt"]),
    "BUS_TA": ("B", ["tangstedt", "ochsenzoll"]),

    # Orbital (cross-suburban) bus links. Hamburg's rail network is radial:
    # without these, a trip from Volksdorf to Norderstedt is forced through the
    # city centre and back out, which is not what anyone actually does. Each
    # link below stands for a real HVV cross-suburban bus corridor.
    "ORB_poppenbuettel_norderstedt": ("b", ["poppenbuettel", "norderstedt_mitte"]),
    "ORB_volksdorf_poppenbuettel":   ("b", ["volksdorf", "poppenbuettel"]),
    "ORB_ahrensburg_poppenbuettel":  ("b", ["ahrensburg", "poppenbuettel"]),
    "ORB_berne_poppenbuettel":       ("b", ["berne", "poppenbuettel"]),
    "ORB_garstedt_niendorf":         ("b", ["garstedt", "niendorf_markt"]),
    "ORB_garstedt_schnelsen":        ("b", ["garstedt", "schnelsen"]),
    "ORB_eidelstedt_niendorf":       ("b", ["eidelstedt", "niendorf_markt"]),
    "ORB_ochsenzoll_poppenbuettel":  ("b", ["ochsenzoll", "poppenbuettel"]),
    "ORB_norderstedt_quickborn":     ("b", ["norderstedt_mitte", "quickborn"]),
    "ORB_niendorf_hoheluft":         ("b", ["niendorf_markt", "hoheluftbruecke"]),
}

OFFICE_NODE = "__office__"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def stations_frame() -> pd.DataFrame:
    """Station table as a DataFrame, with the lines serving each station."""
    served: dict[str, set[str]] = {sid: set() for sid in STATIONS}
    for line, (_mode, seq) in LINES.items():
        for sid in seq:
            served[sid].add(line.split("_")[0])
    rows = [
        {"station_id": sid, "station": name, "lat": lat, "lon": lon,
         "lines": ", ".join(sorted(served[sid])) or "—"}
        for sid, (name, lat, lon) in STATIONS.items()
    ]
    df = pd.DataFrame(rows)
    df["dist_to_office_km"] = [
        haversine_km(r.lat, r.lon, C.WORKPLACE["lat"], C.WORKPLACE["lon"])
        for r in df.itertuples()
    ]
    return df.sort_values("dist_to_office_km").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _edge_minutes(a: str, b: str, mode: str) -> float:
    """
    In-vehicle time between two consecutive stops, derived from geometry.

        t = stop_penalty + 60 * track_distance / cruise_speed

    The stop penalty covers braking, dwell and re-acceleration; the cruise
    speed is the running speed between stops. This two-parameter form is what
    lets one calibration serve both closely-spaced inner-city stations and
    fast outer sections.
    """
    _, la, lo = STATIONS[a]
    _, lb, lob = STATIONS[b]
    km = haversine_km(la, lo, lb, lob) * C.ROUTE_SINUOSITY
    return C.MODE_STOP_PENALTY_MIN[mode] + 60.0 * km / C.MODE_CRUISE_KMH[mode]


def build_graph(offpeak: bool = False, shuttle: bool = False) -> nx.DiGraph:
    """
    Build the line-aware routing graph.

    Nodes are (station_id, line) pairs so that interchanges can be charged a
    real transfer penalty — a station-only graph would silently let passengers
    change lines for free, which flatters public transport badly.

    Parameters
    ----------
    offpeak : thin the service (early / late shifts) by multiplying headways.
    shuttle : add the scenario employer shuttle from Ochsenzoll to the site.
    """
    g = nx.DiGraph()
    head_factor = C.OFFPEAK_HEADWAY_FACTOR if offpeak else 1.0

    # 1. Ride edges along each line (both directions).
    for line, (mode, seq) in LINES.items():
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                continue
            t = _edge_minutes(a, b, mode)
            g.add_edge((a, line), (b, line), weight=t, kind="ride", line=line)
            g.add_edge((b, line), (a, line), weight=t, kind="ride", line=line)

    # 2. Transfer edges between lines at the same station.
    by_station: dict[str, set[str]] = {}
    for node in list(g.nodes):
        by_station.setdefault(node[0], set()).add(node[1])
    for sid, lines in by_station.items():
        for l1 in lines:
            for l2 in lines:
                if l1 == l2:
                    continue
                head = C.LINE_HEADWAY_MIN.get(l2.split("_")[0], 20) * head_factor
                wait = min(C.TRANSFER_WAIT_SHARE * head, C.TRANSFER_WAIT_CAP_MIN)
                g.add_edge((sid, l1), (sid, l2),
                           weight=C.TRANSFER_WALK_MIN + wait,
                           kind="transfer", line=l2)

    # 2b. Walking interchanges between *co-located but separately named*
    # stations. Hauptbahnhof is the case that matters: the U1/U3 platforms
    # (Hbf Süd), the U2/U4 platforms (Hbf Nord) and the S-Bahn/regional
    # concourse carry different names in the network data but are one station
    # complex, 100-200 m apart inside a single building. Without these edges the
    # router cannot change trains at Hamburg's busiest interchange and quietly
    # detours everyone travelling from the south and east, inflating their
    # commute by 5-10 minutes. The same applies to Jungfernstieg/Rathaus,
    # Wandsbeker Chaussee/Hasselbrook and Stephansplatz/Dammtor.
    ids = list(by_station)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            metres = haversine_km(*STATIONS[a][1:], *STATIONS[b][1:]) * 1000
            if metres > C.INTERCHANGE_WALK_MAX_M:
                continue
            walk = metres / 1000 * C.DETOUR_FACTOR / C.WALK_SPEED_KMH * 60
            for src, dst in ((a, b), (b, a)):
                for l1 in by_station[src]:
                    for l2 in by_station[dst]:
                        head = C.LINE_HEADWAY_MIN.get(l2.split("_")[0], 20) * head_factor
                        wait = min(C.TRANSFER_WAIT_SHARE * head, C.TRANSFER_WAIT_CAP_MIN)
                        g.add_edge((src, l1), (dst, l2),
                                   weight=walk + C.TRANSFER_WALK_MIN + wait,
                                   kind="transfer", line=l2)

    # 3. Last mile: gateway stations -> the workplace.
    options = list(C.EGRESS_OPTIONS)
    if shuttle:
        options = list(C.SHUTTLE_STOPS) + options
    best_egress: dict[str, float] = {}
    for sid, bus_min, head in options:
        wait = min(C.TRANSFER_WAIT_SHARE * head * head_factor, C.EGRESS_WAIT_CAP_MIN)
        walk_alt = 60.0 * haversine_km(
            *STATIONS[sid][1:], C.WORKPLACE["lat"], C.WORKPLACE["lon"]
        ) * C.DETOUR_FACTOR / C.EGRESS_WALK_FALLBACK_KMH
        cost = min(bus_min + C.TRANSFER_WALK_MIN + wait, walk_alt)
        # A station can appear more than once (e.g. the scenario shuttle and the
        # scheduled bus both serve Ochsenzoll) — always keep the faster option.
        best_egress[sid] = min(best_egress.get(sid, float("inf")), cost)
    for sid, cost in best_egress.items():
        for line in by_station.get(sid, set()):
            g.add_edge((sid, line), OFFICE_NODE, weight=cost,
                       kind="egress", line="last_mile")
    return g


@lru_cache(maxsize=8)
def station_to_office(offpeak: bool = False, shuttle: bool = False):
    """
    Minutes from every station to the workplace, plus the number of transfers.

    One reverse Dijkstra from the workplace answers the question for the whole
    network at once, so scoring 1 200 employees is instantaneous.

    Returns
    -------
    (times, transfers) : two dicts keyed by station_id.
    """
    g = build_graph(offpeak=offpeak, shuttle=shuttle)
    rev = g.reverse(copy=True)
    dist, paths = nx.single_source_dijkstra(rev, OFFICE_NODE, weight="weight")

    times: dict[str, float] = {}
    transfers: dict[str, int] = {}
    for node, d in dist.items():
        if node == OFFICE_NODE:
            continue
        sid = node[0]
        if sid not in times or d < times[sid]:
            times[sid] = d
            path = list(reversed(paths[node]))     # office -> ... -> node, reversed
            transfers[sid] = sum(
                1 for a, b in zip(path[:-1], path[1:])
                if a != OFFICE_NODE and b != OFFICE_NODE and a[1] != b[1]
            )
    return times, transfers


def station_journey_minutes(origin: str, dest: str, offpeak: bool = False) -> float:
    """
    Rail-to-rail journey time between two stations (excluding the initial wait).

    Used in the notebook to validate the model against published HVV times.
    """
    g = build_graph(offpeak=offpeak)
    src, snk = "__src__", "__snk__"
    for node in list(g.nodes):
        if isinstance(node, tuple):
            if node[0] == origin:
                g.add_edge(src, node, weight=0.0)
            if node[0] == dest:
                g.add_edge(node, snk, weight=0.0)
    return nx.shortest_path_length(g, src, snk, weight="weight")
