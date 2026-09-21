"""Johnson (1977): kürzeste Routen zwischen allen Paaren (oder von einer Startmenge) bei negativen Kanten, ohne n³.

1. Hilfsknoten q mit Kanten der Kosten 0 zu allen Knoten; Bellman-Ford ab q liefert die Potenziale h(v) = d(q, v) <= 0 (oder einen negativen Zyklus).
2. Umgewichten: c'(u, v) = c(u, v) + h(u) - h(v). Die Dreiecksungleichung d(q, v) <= d(q, u) + c(u, v) heißt genau c' >= 0.
3. Dijkstra von jedem Start auf c'.
4. Zurückrechnen: d(s, t) = d'(s, t) - h(s) + h(t).

Jede Route P von s nach t ändert ihre Kosten um dieselbe Konstante h(s) - h(t) (die Potenziale der inneren Knoten heben sich auf): die kürzesten Routen bleiben dieselben.
Die "Falle" (`method="constant"`): alle Kosten um eine Konstante anheben, bis nichts mehr negativ ist - das bestraft Routen mit vielen Kanten und ändert die kürzesten Routen.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus jo_graph.py; Dijkstra und Bellman-Ford kommen aus jo_sp.py; networkx nur in den Tests."""

import dataclasses
from dataclasses import dataclass, field

import numpy as np

import jo_sp
from jo_graph import from_arcs

INF = float("inf")
METHODS = ("johnson", "constant")


def add_virtual_source(g):
    """Der Graph mit einem zusätzlichen Knoten q = g.n, von dem eine Kante der Kosten 0 zu jedem Knoten führt (in q endet keine Kante: q liegt auf keinem Zyklus)."""
    n = g.n
    src = np.repeat(np.arange(n), g.degree())
    arcs = [(int(u), int(v), float(w)) for u, v, w in zip(src, g.indices, g.weight)] + [(n, v, 0.0) for v in range(n)]
    q_xy = g.xy.mean(axis=0) if n else np.zeros(2)
    names = tuple(g.names) + ("q",) if g.names else ()
    return from_arcs(n + 1, arcs, np.vstack([g.xy, q_xy]), names, directed=True, clean=False)


def reweight(g, h):
    """Der Graph mit den Kosten c'(u, v) = c(u, v) + h(u) - h(v) (dieselbe Struktur)."""
    src = np.repeat(np.arange(g.n), g.degree())
    return dataclasses.replace(g, weight=g.weight + h[src] - h[g.indices])


def is_feasible(g, h, tol=1e-9):
    """Ein Potenzial h ist zulässig, wenn alle umgewichteten Kosten nichtnegativ sind."""
    return bool((reweight(g, np.asarray(h, dtype=float)).weight >= -tol).all())


@dataclass
class Johnson:
    method: str
    sources: list
    negative_cycle: bool                               # ein negativer Zyklus im Netz: keine Potenziale, keine Ergebnisse
    cycle: list = field(default_factory=list)          # der Zyklus als geschlossene Knotenfolge
    cycle_cost: float = 0.0
    h: object = None                                   # Potenziale (n); None bei der Falle
    reweighted: object = None                          # Graph mit c' (bei der Falle: c + M)
    shift: float = 0.0                                 # bei der Falle: die Konstante M
    bf: object = None                                  # Bellman-Ford ab q auf dem erweiterten Graphen (Protokoll für die Ansicht)
    dist: object = None                                # len(sources) x n: Kosten im ORIGINALNETZ (bei der Falle: die Kosten der gefundenen Route)
    dist_rw: object = None                             # len(sources) x n: Entfernungen d' im umgewichteten Netz
    parent: object = None                              # len(sources) x n: Vorgänger aus Dijkstra
    order: list = field(default_factory=list)          # je Start die Festlegereihenfolge von Dijkstra
    counters: dict = field(default_factory=dict)       # bf_checks, bf_rounds, dijkstra_checks (gesamt), checks_total
    tight_edges: int = 0                               # Kanten mit c' = 0

    def row(self, s):
        return self.sources.index(int(s))

    def route(self, s, t):
        """Route von s nach t aus den Dijkstra-Vorgängern (Kanten des Originalnetzes); leer, wenn unerreichbar."""
        r = self.row(s)
        if not np.isfinite(self.dist[r, t]):
            return []
        path, cur = [int(t)], int(t)
        while cur != int(s):
            cur = int(self.parent[r, cur])
            path.append(cur)
        return path[::-1]


def johnson(g, sources=None, method="johnson", cycle_check=True):
    """Alle Paare (sources=None) oder nur die Zeilen der Startmenge. `method="johnson"`: Potenziale aus Bellman-Ford; `"constant"`: die Falle (Kosten um |min| anheben, keine Potenziale)."""
    if method not in METHODS:
        raise ValueError(method)
    n = g.n
    srcs = list(range(n)) if sources is None else [int(s) for s in sources]
    counters = {"bf_checks": 0, "bf_rounds": 0, "dijkstra_checks": 0, "checks_total": 0}
    if method == "johnson":
        aug = add_virtual_source(g)
        bf = jo_sp.bellman_ford(aug, n, "early_stop", trace=True, cycle_check=cycle_check)
        counters["bf_checks"], counters["bf_rounds"] = bf.counters["checks"], bf.rounds
        if bf.negative_cycle:
            counters["checks_total"] = counters["bf_checks"]
            cyc = [v for v in bf.cycle]
            return Johnson(method, srcs, True, cyc, float(bf.cycle_cost), None, None, 0.0, bf, None, None, None, [], counters, 0)
        h = np.asarray(bf.dist[:n], dtype=float)
        rw = reweight(g, h)
        assert (rw.weight >= -1e-9).all()
        shift = 0.0
    else:
        bf, h = None, None
        shift = float(max(0.0, -g.weight.min())) if g.m else 0.0
        rw = dataclasses.replace(g, weight=g.weight + shift)
    rows_d, rows_p, order = np.full((len(srcs), n), INF), np.full((len(srcs), n), -1, dtype=np.int64), []
    rows_true = np.full((len(srcs), n), INF)
    dj_checks = 0
    for r, s in enumerate(srcs):
        res = jo_sp.dijkstra(rw, s)
        dj_checks += res.counters["checks"]
        rows_d[r], rows_p[r] = res.dist, res.parent
        order.append(list(res.order))
        if method == "johnson":
            fin = np.isfinite(res.dist)
            rows_true[r, fin] = res.dist[fin] - h[s] + h[np.where(fin)[0]]
        else:                                                                      # Kosten der gefundenen Route im Originalnetz nachrechnen (Elternkette in Festlegereihenfolge)
            rows_true[r, s] = 0.0
            for v in res.order:
                p = int(res.parent[v])
                if p >= 0:
                    rows_true[r, v] = rows_true[r, p] + float(g.weight[g.arc(p, v)])
    counters["dijkstra_checks"] = dj_checks
    counters["checks_total"] = counters["bf_checks"] + dj_checks
    tight = int((rw.weight == 0).sum()) if method == "johnson" else 0
    return Johnson(method, srcs, False, [], 0.0, h, rw, shift, bf, rows_true, rows_d, rows_p, order, counters, tight)


def constant_shift_report(g, exact):
    """Die Falle gegen die richtige Matrix `exact` (n x n): Anteil der Paare, deren gefundene Route nicht die billigste ist, mittlerer Mehrpreis dieser Paare (Kosten der gefundenen Route / richtige Kosten - 1
    für positive richtige Kosten, sonst Differenz) und Paare mit anderer Kantenzahl. Nur ohne negativen Zyklus."""
    wrong = johnson(g, None, "constant")
    off = ~np.eye(g.n, dtype=bool) & np.isfinite(exact)
    bad = off & ~np.isclose(wrong.dist, exact)
    with np.errstate(invalid="ignore"):
        diff = (wrong.dist - exact)[bad]
    return {"pairs": int(off.sum()), "wrong": int(bad.sum()), "share_wrong": float(bad.sum() / max(off.sum(), 1)), "mean_excess": float(diff.mean()) if diff.size else 0.0, "shift": wrong.shift}
