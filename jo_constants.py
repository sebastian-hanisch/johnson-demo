"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("small", "small_cycle", "ev", "random", "city")
NET_LABELS = {
    "small": "🔀 Kleines Netz (eine negative Kante)",
    "small_cycle": "🔁 Kleines Netz mit negativem Zyklus",
    "ev": "🔋 E-Lieferwagen mit Rekuperation (erzeugt)",
    "random": "🕸️ Zufallsnetz mit Potenzialen (erzeugt)",
    "city": "🏙️ Stadtnetz ohne negative Kanten (erzeugt)",
}
SMALL_NETS = ("small", "small_cycle")                      # eigene kleine Graphen mit Namen: Kosten an den Kanten, Tabelle, feste Aufgabe
SIZED_NETS = ("city", "ev")                                # Netze mit Größenregler (Raster)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 8                # n = Seite², höchstens 400 Knoten
HILL_MIN, HILL_MAX, DEFAULT_HILL = 0, 40, 30
ETA_MIN, ETA_MAX, DEFAULT_ETA = 0, 90, 60
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 1.5
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 400, 100
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 1.5, 12.0, 3.0    # mittlerer Ausgangsgrad: dünn bis dicht
POT_MIN, POT_MAX, DEFAULT_POT = 0, 20, 8                   # Potenzialspanne (Zufallsnetz): 0 = keine negativen Kanten
DISTANCE_MIN, DISTANCE_MAX, DEFAULT_DISTANCE = 10, 100, 60     # Prozent: Rang der Entfernung des Ziels vom Start (Routen-Ansicht)
DEFAULT_SEED = 7
DEFAULT_NET = "small"

METHODS = ("johnson", "constant")
DEFAULT_METHOD = "johnson"
METHOD_LABELS = {"johnson": "Johnson: Potenziale aus Bellman-Ford", "constant": "Falle: alle Kosten um eine Konstante anheben"}

SWEEP_SEEDS = tuple(range(100000, 100005))

COLORS = {"jo": "#d62728", "wrong": "#1f77b4", "negative": "#2ca02c", "cycle": "#d62728", "start": "#111111", "goal": "#ff7f0e", "tight": "#ffd54f", "q": "#9467bd"}

_BASE = dict(side=DEFAULT_SIDE, hill=DEFAULT_HILL, eta=DEFAULT_ETA, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, pot=DEFAULT_POT,
             method=DEFAULT_METHOD, distance=DEFAULT_DISTANCE, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🔁 Negativer Zyklus": {**_BASE, "net": "small_cycle"},
    "🔋 E-Lieferwagen": {**_BASE, "net": "ev"},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines Liefernetz (dasselbe wie in der Floyd-Warshall-Demo) mit einer Rückvergütung Ost → Nord (−3 Euro): Bellman-Ford ab q braucht 3 Runden und 45 Kantenprüfungen und findet genau ein Potenzial ungleich null (Nord: −3). Danach ist keine Kante mehr negativ, n-mal Dijkstra braucht 54 Prüfungen: zusammen 99 gegen 216 Vergleiche bei Floyd-Warshall und 180 bei n-mal Bellman-Ford. n-mal Dijkstra ohne Umgewichten liegt an 7 von 30 Paaren falsch.",
    "🔁 Negativer Zyklus": "Dasselbe Netz mit einer Rückvergütung auf Süd → West: Hin- und Rückweg zwischen Süd und West kosten −2 Euro. Bellman-Ford ab q meldet den Zyklus nach 2 Runden (30 Kantenprüfungen); ohne Potenziale kann Johnson nicht weitermachen, für Paare, die den Zyklus berühren, gibt es keine kürzeste Route.",
    "🔋 E-Lieferwagen": "Hügeliges Stadtnetz (8 × 8, 30 m Hügel, 60 % Rückgewinnung): 42 von 336 Kanten sind negativ. Bellman-Ford ab q braucht 1 200 Kantenprüfungen (3 Runden), danach ist keine Kante mehr negativ (32 sind eng). Zusammen mit n-mal Dijkstra (21 504) sind es 22 704 gegen 262 144 bei Floyd-Warshall und 137 424 bei n-mal Bellman-Ford; n-mal Dijkstra ohne Umgewichten liegt an 524 von 4 032 Paaren falsch.",
    "🕸️ Zufallsnetz": "Zufallsnetz (100 Knoten, mittlerer Grad 3, Potenzialspanne 8): 38 von 300 Kanten sind negativ. Johnson braucht 31 600 Kantenprüfungen (1 600 Bellman-Ford ab q, 30 000 n-mal Dijkstra) gegen 1 000 000 Vergleiche bei Floyd-Warshall und 190 500 bei n-mal Bellman-Ford. Die Falle (alle Kosten um 5 anheben) liegt an 26.6 % der Paare daneben, am gezeigten Paar 24 statt 21.",
    "🏙️ Stadtnetz": "Stadtnetz ohne negative Kanten (8 × 8): alle Potenziale sind 0, die umgewichteten Kosten sind die ursprünglichen. Die Sicherheit kostet 800 Kantenprüfungen für Bellman-Ford ab q (2 Runden), 3.7 % der 21 504 des n-mal Dijkstra: zusammen 22 304 gegen 262 144 bei Floyd-Warshall.",
}
