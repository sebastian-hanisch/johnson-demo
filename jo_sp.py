"""Bellman-Ford (Ford 1956, Bellman 1958) in vier Varianten, dazu Dijkstra als Gegenstück (aus der Dijkstra-Demo, hier nur mit Alarm bei negativen Kanten).

Alles ist eigene Umsetzung auf dem CSR-Graphen aus jo_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe).

Varianten (`variant`):
- "textbook":     genau n - 1 Runden über alle Kanten, dann eine Prüfrunde (kein früher Abbruch), Kanten nacheinander, Verbesserungen wirken sofort ("in place");
- "early_stop":   wie textbook, aber Abbruch nach der ersten Runde ohne Verbesserung;
- "synchronous":  wie early_stop, aber jede Runde liest nur die Werte der vorigen Runde: nach Runde k ist dist genau die billigste Route mit höchstens k Kanten;
- "queue":        Warteschlangen-Variante ("SPFA"): nur Knoten, deren Wert sich verbessert hat, prüfen ihre ausgehenden Kanten.

Kantenreihenfolge (`edge_order`), nur bei den Runden-Varianten mit Verbesserung "in place" von Bedeutung:
"as_given" (Kanten nach Startknoten sortiert, wie im Graphen), "random", "near_first" (Kanten ab den Knoten mit kleinster wahrer Entfernung zuerst - eine Reihenfolge, die man vorher nicht kennen kann),
"far_first" (das Gegenteil).

Zyklus-Prüfung (`cycle_check=True`, nur Runden-Varianten): nach jeder Runde mit Verbesserung wird in den Vorgänger-Zeigern nach einem Kreis gesucht (ein Kreis in den Zeigern ist immer ein negativer Zyklus des Netzes).
Ohne die Prüfung wird ein Zyklus erst in Runde n bemerkt, also nach n · m Kantenprüfungen."""

from dataclasses import dataclass, field

import numpy as np

from jo_graph import route_cost
from jo_queues import QUEUES

INF = float("inf")
VARIANTS = ("textbook", "early_stop", "synchronous", "queue")
EDGE_ORDERS = ("as_given", "random", "near_first", "far_first")


@dataclass
class BellmanFord:
    dist: np.ndarray                                   # Kosten vom Start; bei negativem Zyklus der Stand beim Abbruch (unbrauchbar)
    parent: np.ndarray                                 # Vorgänger, -1 = Start oder nicht erreicht
    negative_cycle: bool                               # von der Quelle aus erreichbarer negativer Zyklus gefunden
    cycle: list = field(default_factory=list)          # der Zyklus als geschlossene Knotenfolge [a, b, ..., a]; leer ohne Zyklus
    cycle_cost: float = 0.0                            # Summe seiner Kantenkosten (< 0)
    counters: dict = field(default_factory=dict)       # rounds, productive_rounds, checks (Kantenprüfungen), improvements, pops/requeues (queue)
    improved_per_round: list = field(default_factory=list)   # Zahl der Verbesserungen je Runde (queue: je 100 Entnahmen keine Angabe: leer)
    history: list = field(default_factory=list)        # nur mit trace=True: je Runde (dist, parent, [(u, v, alt, neu)]) - bei "queue" je Entnahme
    source: int = 0

    def route(self, target):
        """Kürzeste Route zum Ziel als Knotenfolge; leer, wenn nicht erreichbar oder wenn ein negativer Zyklus erreichbar ist (dann gibt es keine kürzeste)."""
        if self.negative_cycle or not np.isfinite(self.dist[target]):
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]

    @property
    def rounds(self):
        return self.counters.get("rounds", 0)


def _edges(g):
    src = np.repeat(np.arange(g.n), g.degree())
    return src, g.indices.copy(), g.weight.copy()


def _order_edges(g, src, dst, w, edge_order, source, seed):
    if edge_order == "as_given":
        return src, dst, w
    if edge_order == "random":
        perm = np.random.default_rng([int(seed), 909]).permutation(len(src))
    elif edge_order in ("near_first", "far_first"):
        ref = bellman_ford(g, source, "queue")
        key = np.where(np.isfinite(ref.dist[src]), ref.dist[src], 1e18)
        perm = np.argsort(key if edge_order == "near_first" else -key, kind="stable")
    else:
        raise ValueError(edge_order)
    return src[perm], dst[perm], w[perm]


def _find_cycle(parent, start, n, g):
    """Vom Knoten `start`, der noch verbessert wurde, n Schritte den Vorgängern folgen (landet sicher auf dem Zyklus), dann den Zyklus einmal umlaufen."""
    v = int(start)
    for _ in range(n):
        v = int(parent[v])
    cyc, u = [v], int(parent[v])
    while u != v:
        cyc.append(u)
        u = int(parent[u])
    cyc.append(v)
    cyc = cyc[::-1]                                     # Vorgänger-Zeiger laufen rückwärts: umdrehen, dann ist es die Fahrtrichtung
    return cyc, route_cost(g, cyc)


def _parent_cycle(parent, n):
    """Ein Kreis in den Vorgänger-Zeigern (Knotenfolge entgegen der Fahrtrichtung) oder None; jeder Knoten wird höchstens einmal besucht."""
    state = [0] * n
    for s in range(n):
        if state[s]:
            continue
        path, v = [], s
        while v >= 0 and state[v] == 0:
            state[v] = 1
            path.append(v)
            v = parent[v]
        if v >= 0 and state[v] == 1:
            return path[path.index(v):]
        for x in path:
            state[x] = 2
    return None


def bellman_ford(g, source, variant="early_stop", edge_order="as_given", trace=False, seed=0, cycle_check=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    n, source = g.n, int(source)
    dist, parent = [INF] * n, [-1] * n
    dist[source] = 0.0
    counters = {"rounds": 0, "productive_rounds": 0, "checks": 0, "improvements": 0}
    per_round, history = [], []

    if variant == "queue":
        return _queue_variant(g, source, dist, parent, counters, trace)

    src, dst, w = _order_edges(g, *_edges(g), edge_order, source, seed)
    src_l, dst_l, w_l = src.tolist(), dst.tolist(), w.tolist()
    sync = variant == "synchronous"
    cycle_node = -1
    for rnd in range(1, n + 1):                         # Runde n ist die Prüfrunde: verbessert sie noch, gibt es einen negativen Zyklus
        read = list(dist) if sync else dist
        changed, improved, last = [], 0, -1
        for u, v, c in zip(src_l, dst_l, w_l):
            du = read[u]
            if du != INF and du + c < dist[v]:
                if trace:
                    changed.append((u, v, dist[v], du + c))
                dist[v], parent[v] = du + c, u
                improved += 1
                last = v
        counters["checks"] += len(src_l)
        counters["improvements"] += improved
        counters["rounds"] = rnd
        per_round.append(improved)
        if improved:
            counters["productive_rounds"] += 1
        if trace:
            history.append((list(dist), list(parent), changed))
        if not improved:
            if variant != "textbook":
                break
            continue
        if rnd == n:
            cycle_node = last
        elif cycle_check:
            loop = _parent_cycle(parent, n)
            if loop is not None:
                cyc = loop[::-1]                                 # Zeiger laufen rückwärts: umdrehen (Fahrtrichtung) und schließen
                cyc = cyc + [cyc[0]]
                counters["detected_round"] = rnd
                return BellmanFord(np.array(dist), np.array(parent), True, cyc, route_cost(g, cyc), counters, per_round, history, source)
    if cycle_node >= 0:
        cyc, cost = _find_cycle(parent, cycle_node, n, g)
        return BellmanFord(np.array(dist), np.array(parent), True, cyc, cost, counters, per_round, history, source)
    return BellmanFord(np.array(dist), np.array(parent), False, [], 0.0, counters, per_round, history, source)


def _queue_variant(g, source, dist, parent, counters, trace):
    n = g.n
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    from collections import deque
    q, queued = deque([source]), [False] * n
    queued[source] = True
    times = [0] * n                                     # wie oft ein Knoten in die Warteschlange kam; n Mal = Zyklus
    times[source] = 1
    counters.update({"pops": 0, "requeues": 0})
    history = []
    stride = 1 if n <= 40 else max(1, n // 60)          # große Netze: nur etwa 60 Zwischenstände (Speicher), die Änderungen dazwischen werden gesammelt
    changed = []
    while q:
        u = q.popleft()
        queued[u] = False
        counters["pops"] += 1
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            counters["checks"] += 1
            if dist[u] + w[k] < dist[v]:
                if trace:
                    changed.append((u, v, dist[v], dist[u] + w[k]))
                dist[v], parent[v] = dist[u] + w[k], u
                counters["improvements"] += 1
                if not queued[v]:
                    times[v] += 1
                    if times[v] > 1:
                        counters["requeues"] += 1
                    if times[v] >= n:
                        counters["rounds"] = counters["pops"]
                        cyc, cost = _find_cycle(parent, v, n, g)
                        return BellmanFord(np.array(dist), np.array(parent), True, cyc, cost, counters, [], history, source)
                    q.append(v)
                    queued[v] = True
        if trace and (counters["pops"] % stride == 0 or not q):
            history.append((list(dist), list(parent), changed))
            changed = []
    counters["rounds"] = counters["pops"]
    return BellmanFord(np.array(dist), np.array(parent), False, [], 0.0, counters, [], history, source)


# --- Dijkstra als Gegenstück ------------------------------------------------------------------------------------------------------------------

@dataclass
class Dijkstra:
    dist: np.ndarray                                   # bei festgelegten Knoten endgültig (mit negativen Kanten nicht mehr!)
    parent: np.ndarray
    order: list = field(default_factory=list)          # festgelegte Knoten in Reihenfolge
    alarms: list = field(default_factory=list)         # (Knoten, festgelegter Wert, besserer Wert): ein festgelegter Knoten hätte sich verbessern lassen - nur mit negativen Kanten
    counters: dict = field(default_factory=dict)       # checks (Kantenprüfungen), pops

    def route(self, target):
        if not np.isfinite(self.dist[target]):
            return []
        path = [int(target)]
        while self.parent[path[-1]] >= 0:
            path.append(int(self.parent[path[-1]]))
        return path[::-1]


def dijkstra(g, source):
    """Dijkstra, wie er gelehrt wird: jeder Knoten wird genau einmal festgelegt; findet sich danach eine bessere Route zu ihm (nur mit negativen Kanten möglich), zählt das als Alarm und wird NICHT nachgebessert."""
    n = g.n
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    q = QUEUES["lazy"](n)
    dist, parent, settled = [INF] * n, [-1] * n, [False] * n
    dist[int(source)] = 0.0
    q.push_or_decrease(int(source), 0.0)
    order, alarms, checks = [], [], 0
    while len(q):
        d, u = q.pop_min()
        if settled[u]:
            continue
        settled[u] = True
        order.append(u)
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            checks += 1
            nd = d + w[k]
            if nd < dist[v]:
                if settled[v]:
                    alarms.append((v, dist[v], nd))
                    continue
                dist[v], parent[v] = nd, u
                q.push_or_decrease(v, nd)
    return Dijkstra(np.array(dist), np.array(parent), order, alarms, {"checks": checks, "pops": len(order)})


def hop_limited(g, source, k):
    """Billigste Kosten mit höchstens k Kanten (Vergleichsrechnung für die Tests und die Ansicht "höchstens k Etappen"): k-mal die Werte der vorigen Runde lesen."""
    src, dst, w = _edges(g)
    cur = np.full(g.n, INF)
    cur[int(source)] = 0.0
    for _ in range(k):
        nxt = cur.copy()
        ok = np.isfinite(cur[src])
        np.minimum.at(nxt, dst[ok], cur[src[ok]] + w[ok])
        cur = nxt
    return cur
