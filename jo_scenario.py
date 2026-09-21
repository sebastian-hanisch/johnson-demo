"""Die Netze der Demo: kleines Netz mit einer negativen Kante (und seine Variante mit negativem Zyklus), erzeugtes Stadtnetz, E-Lieferwagen mit Rekuperation (negative Kanten, nie ein Zyklus)
und Zufallsnetz mit einstellbarer Dichte. Alle Graphen sind eigene Konstruktionen, alle Kosten ganze Zahlen."""

import math
from dataclasses import dataclass

import numpy as np

import jo_constants as C
from jo_graph import Graph, from_arcs


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    unit: str                      # Einheit der Kosten
    title: str
    note: str
    geometric: bool = True         # Lage der Knoten ist eine Karte (Kanten zeichnen), sonst Knoten zufällig verteilt
    heights: tuple = ()            # E-Lieferwagen: Höhe je Kreuzung in Metern


# --- Kleines Netz ---------------------------------------------------------------------------------------------------------------------------------

SMALL_STOPS = [("Depot", 0.0, 2.0), ("Nord", 2.5, 4.0), ("Ost", 2.5, 0.0), ("Süd", 5.5, 4.0), ("West", 5.5, 0.0), ("Hafen", 8.0, 2.0)]
SMALL_LEGS = [("Depot", "Nord", 1), ("Depot", "Ost", 2), ("Nord", "Süd", 5), ("Ost", "Nord", -3), ("Ost", "West", 6), ("Süd", "Hafen", 2), ("West", "Süd", 1), ("West", "Hafen", 7), ("Hafen", "Depot", 3)]
CYCLE_LEGS = [("Depot", "Nord", 1), ("Depot", "Ost", 2), ("Nord", "Süd", 5), ("Ost", "Nord", -3), ("Ost", "West", 6), ("Süd", "Hafen", 2), ("West", "Süd", 1), ("West", "Hafen", 7), ("Süd", "West", -3)]


def small_network(cycle=False):
    legs = CYCLE_LEGS if cycle else SMALL_LEGS
    names = [s[0] for s in SMALL_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in legs], [(s[1], s[2]) for s in SMALL_STOPS], names, directed=True)
    note = ("Ein eigenes kleines Liefernetz (Kosten in Euro je Strecke): die Strecke Ost → Nord hat eine Rückvergütung von 3 Euro (negative Kosten). " +
            ("Hier fehlt die Rückfahrt Hafen → Depot, dafür ist Süd → West mit −3 Euro vergütet: der Hin- und Rückweg zwischen Süd und West (1 − 3 = −2 Euro) ist ein negativer Zyklus." if cycle else
             "Alle Rundläufe kosten etwas: es gibt eine negative Kante, aber keinen negativen Zyklus."))
    return Network("small_cycle" if cycle else "small", g, "Euro", "Kleines Netz mit negativem Zyklus" if cycle else "Kleines Netz", note, True)


# --- Raster-Topologie (Stadtnetz und E-Lieferwagen) -------------------------------------------------------------------------------------------

def _primitive_offsets(reach):
    r = int(math.floor(reach + 1e-9))
    out = []
    for dy in range(0, r + 1):
        for dx in range(-r, r + 1):
            if (dy == 0 and dx <= 0) or dx * dx + dy * dy > reach * reach + 1e-9 or math.gcd(abs(dx), dy) != 1:
                continue
            out.append((dy, dx))
    return out


def _grid(side, reach, blocked_pct, seed):
    """Gestörtes Raster mit Lage der Kreuzungen und Liste der Straßen (ungerichtete Paare); ein Teil der Straßen ist gesperrt, das Netz bleibt zusammenhängend."""
    rng = np.random.default_rng([int(seed), 101])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    pairs = []
    for dy, dx in _primitive_offsets(reach):
        for i in range(side):
            for j in range(side):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < side and 0 <= j2 < side:
                    pairs.append((i * side + j, i2 * side + j2))
    pairs = np.array(pairs)
    order = rng.permutation(len(pairs))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    in_tree = np.zeros(len(pairs), dtype=bool)
    for k in order:
        a, b = find(pairs[k, 0]), find(pairs[k, 1])
        if a != b:
            parent[a] = b
            in_tree[k] = True
    removable = [k for k in order if not in_tree[k]]
    drop = set(removable[: int(round(len(pairs) * blocked_pct / 100.0))])
    keep = np.array([k not in drop for k in range(len(pairs))])
    return xy, pairs[keep], rng


def build_city(side, reach, spread, seed, blocked_pct=20):
    xy, pairs, rng = _grid(side, reach, blocked_pct, seed)
    length = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    cost = np.maximum(1.0, np.rint(length * (1.0 + spread * rng.random(len(pairs)))))
    return from_arcs(side * side, [(pairs[k, 0], pairs[k, 1], cost[k]) for k in range(len(pairs))], xy)


def city_network(side, reach, spread, seed):
    g = build_city(side, reach, spread, seed)
    return Network("city", g, "m", "Stadtnetz", "Erzeugtes Stadtnetz: Kreuzungen auf einem gestörten Raster, alle Kosten (Meter mal Streuung) positiv - hier ist auch n-mal Dijkstra richtig, und es geht um den Aufwand.")


# --- E-Lieferwagen mit Rekuperation ------------------------------------------------------------------------------------------------------------

WH_PER_M = 0.15                    # Verbrauch auf ebener Strecke: 150 Wh je km
WH_PER_M_CLIMB = 5.45              # Lageenergie eines 2-Tonnen-Wagens: 5.45 Wh je Meter Höhe (m·g·h)


def build_hills(side, hill_m, seed):
    """Glatte Hügellandschaft: drei Wellen mit zufälliger Richtung und Phase, Wellenlänge 8 bis 14 Blocklängen; `hill_m` = größter Höhenunterschied (Spitze zu Tal) in Metern."""
    rng = np.random.default_rng([int(seed), 505])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1).astype(float)
    h = np.zeros(n)
    for _ in range(3):
        ang, phase = rng.uniform(0, 2 * math.pi, 2)
        wave = rng.uniform(8.0, 14.0)
        h += np.sin(2 * math.pi * (ij[:, 1] * math.cos(ang) + ij[:, 0] * math.sin(ang)) / wave + phase)
    if h.max() > h.min():
        h = (h - h.min()) / (h.max() - h.min())
    return h * hill_m


def ev_network(side, hill_m, eta_pct, seed, reach=1.5, blocked_pct=20):
    """Energiekosten in Wh je Strecke: Reibung (0.15 Wh je Meter) plus Lageenergie beim Bergauffahren (5.45 Wh je Meter Höhe); bergab bekommt der Wagen `eta_pct` Prozent der Lageenergie zurück -
    das kann größer sein als die Reibung: die Strecke hat dann negative Kosten. Ein Kreis kostet nie etwas ab: Auf- und Abstieg sind gleich, und mit Rückgewinnung unter 100 % geht Energie verloren."""
    xy, pairs, _ = _grid(side, reach, blocked_pct, seed)
    heights = build_hills(side, hill_m, seed)
    arcs = []
    for a, b in pairs:
        length = float(np.hypot(*(xy[a] - xy[b])))
        dh = float(heights[b] - heights[a])
        for u, v, d in ((int(a), int(b), dh), (int(b), int(a), -dh)):
            lift = WH_PER_M_CLIMB * (d if d > 0 else d * eta_pct / 100.0)
            arcs.append((u, v, float(round(WH_PER_M * length + lift))))
    g = from_arcs(side * side, arcs, xy, directed=True)
    note = ("Erzeugtes hügeliges Stadtnetz für einen E-Lieferwagen: Kosten in Wh (Reibung plus Lageenergie); bergab wird Energie zurückgewonnen, manche Strecken haben deshalb negative Kosten. "
            "Es gibt nie einen negativen Zyklus: was man bergab gewinnt, hat man bergauf bezahlt.")
    return Network("ev", g, "Wh", "E-Lieferwagen mit Rekuperation", note, True, tuple(float(x) for x in heights))


# --- Zufallsnetz mit negativen Kanten ------------------------------------------------------------------------------------------------------------

def build_random(n, degree, pot_span, seed):
    """Gerichteter, stark zusammenhängender Zufallsgraph (ein Rundweg durch alle Knoten plus zufällige Kanten bis zum mittleren Ausgangsgrad `degree`), Kosten 1 bis 9 plus Potenzial(u) - Potenzial(v).
    Jeder Kreis hat dieselbe Summe wie ohne Potenziale (die Potenziale heben sich auf): es entstehen negative Kanten, aber nie ein negativer Zyklus. Das ist die Idee hinter Johnsons Umgewichtung."""
    rng = np.random.default_rng([int(seed), 707])
    perm = rng.permutation(n)
    pot = np.random.default_rng([int(seed), 606]).integers(0, pot_span + 1, n) if pot_span > 0 else np.zeros(n, dtype=int)     # eigener Zufallsstrom: die Struktur des Netzes hängt nicht von der Spanne ab
    arcs = {(int(perm[i]), int(perm[(i + 1) % n])): int(rng.integers(1, 10)) for i in range(n)}
    target = int(round(n * degree))
    while len(arcs) < target:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs:
            arcs[(u, v)] = int(rng.integers(1, 10))
    xy = rng.random((n, 2)) * 1000.0
    return from_arcs(n, [(u, v, float(c + pot[u] - pot[v])) for (u, v), c in arcs.items()], xy, directed=True)


def random_network(n, degree, pot_span, seed):
    g = build_random(int(n), float(degree), int(pot_span), int(seed))
    return Network("random", g, "Einheiten", "Zufallsnetz mit negativen Kanten",
                   "Erzeugter gerichteter Zufallsgraph, Kosten 1 bis 9 plus Potenzial(Start) − Potenzial(Ziel): je größer die Potenzialspanne, desto mehr negative Kanten - aber nie ein negativer Zyklus. Keine Karte, nur Punkte.", False)


# --- Zusammenbau -------------------------------------------------------------------------------------------------------------------------------

def make_network(net, side=C.DEFAULT_SIDE, hill=C.DEFAULT_HILL, eta=C.DEFAULT_ETA, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, pot=C.DEFAULT_POT, seed=C.DEFAULT_SEED):
    if net == "small":
        return small_network(False)
    if net == "small_cycle":
        return small_network(True)
    if net == "ev":
        return ev_network(int(side), float(hill), float(eta), int(seed))
    if net == "random":
        return random_network(int(nodes), float(degree), int(pot), int(seed))
    if net == "city":
        return city_network(int(side), float(reach), float(spread), int(seed))
    raise ValueError(net)


def hidden_potentials(n, pot_span, seed):
    """Die Potenziale, mit denen `build_random` die Kosten 1 bis 9 verschoben hat (Kosten = Kosten + pot(Start) - pot(Ziel)); h = -pot macht daraus wieder die ursprünglichen Kosten."""
    return np.random.default_rng([int(seed), 606]).integers(0, pot_span + 1, n) if pot_span > 0 else np.zeros(n, dtype=int)
