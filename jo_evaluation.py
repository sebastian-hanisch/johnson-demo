"""Johnson gegen Floyd-Warshall, n-mal Bellman-Ford und n-mal Dijkstra: Kennzahlen für ein Netz, Paarwahl, Bildfolge für die Ansicht, Experimente (Aufwand gegen Dichte und Größe, wenige Startknoten, die Konstanten-Falle),
Urteil für die App.

Der Aufwand wird in Zählern gemessen (Kantenprüfungen, Vergleiche; plattformfest). Die Zähler sind kein gemeinsames Maß (ein Vergleich in Floyd-Warshalls Matrixoperation ist billiger als eine Kantenprüfung mit Warteschlange);
Laufzeiten stehen nur als Messwerte in der App."""

import time
from dataclasses import dataclass

import numpy as np

import jo_algorithm as alg
import jo_constants as C
import jo_fw as fwm
import jo_sp
from jo_graph import from_arcs
from jo_scenario import build_random, ev_network, make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    jo: alg.Johnson
    method: str
    s: int
    t: int
    metrics: dict
    seconds: dict
    exact: object = None                   # n x n richtige Matrix (Floyd-Warshall), None bei negativem Zyklus


def pick_pair(net, dist_row_fn, distance_pct=60, seed=C.DEFAULT_SEED):
    """Start und Ziel der Routen-Ansicht: bei den kleinen Netzen die feste Aufgabe (Depot -> Hafen); sonst ein Start (Netze mit Karte: nahe bei 30 % Breite und 50 % Höhe, sonst zufällig) und als Ziel der Knoten, dessen
    richtige Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei `distance_pct` Prozent liegt. `dist_row_fn(s)` liefert die richtigen Entfernungen ab s."""
    g = net.graph
    if g.names:
        return 0, g.n - 1
    if net.geometric:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        s = int(np.argmin(np.hypot(*(g.xy - (lo + (hi - lo) * np.array([0.3, 0.5]))).T)))
    else:
        s = int(np.random.default_rng([int(seed), 808]).integers(0, g.n))
    d = dist_row_fn(s)
    reachable = np.where(np.isfinite(d))[0]
    reachable = reachable[reachable != s]
    if not len(reachable):
        return s, s
    order = reachable[np.argsort(d[reachable], kind="stable")]
    return s, int(order[min(len(order) - 1, int(round(distance_pct / 100.0 * (len(order) - 1))))])


def analyse(net, method=C.DEFAULT_METHOD, distance=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED):
    g = net.graph
    n = g.n
    t0 = time.perf_counter()
    jo = alg.johnson(g, None, method)
    t_jo = time.perf_counter() - t0
    right = jo if method == "johnson" else alg.johnson(g, None, "johnson")            # die richtige Matrix (bei der Falle zum Vergleich)
    cyc = right.negative_cycle
    t0 = time.perf_counter()
    _, bf_checks, bf_cycles = fwm.all_pairs_by_bellman_ford(g)
    t_bf = time.perf_counter() - t0
    neg = int((g.weight < 0).sum())
    m = {"n": n, "m": g.m, "neg_edges": neg, "neg_share": neg / max(g.m, 1), "cycle": cyc, "cycle_nodes": len(set(right.cycle)) if cyc else 0, "cycle_cost": right.cycle_cost,
         "bf_checks": right.counters["bf_checks"], "bf_rounds": right.counters["bf_rounds"], "dijkstra_checks": right.counters["dijkstra_checks"], "total": right.counters["checks_total"],
         "fw_comparisons": n ** 3, "nbf_checks": bf_checks, "nbf_cycle_sources": bf_cycles, "cells": n * n, "method": method}
    exact = None
    if not cyc:
        t0 = time.perf_counter()
        exact = right.dist
        fin = np.isfinite(exact)
        naive, _ = fwm.all_pairs_by_dijkstra(g)
        fw = fwm.floyd_warshall(g, "numpy")
        t_fw = time.perf_counter() - t0
        both = np.isfinite(naive) & fin
        m.update({"exact_vs_fw": bool(np.array_equal(np.isfinite(fw.dist), fin) and np.allclose(fw.dist[fin], exact[fin])), "naive_wrong": int((both & ~np.isclose(naive, exact)).sum()), "pairs": n * (n - 1),
                  "tight_edges": right.tight_edges, "tight_share": right.tight_edges / max(g.m, 1), "h_min": float(right.h.min()), "h_nonzero": int((right.h != 0).sum()),
                  "neg_after": int((right.reweighted.weight < 0).sum()), "reachable_share": float(fin[~np.eye(n, dtype=bool)].mean()) if n > 1 else 1.0, "fw_seconds": t_fw})
    s, t = pick_pair(net, (lambda src: exact[src]) if exact is not None else (lambda src: np.zeros(n)), distance, seed)
    m["s"], m["t"] = s, t
    if exact is not None:
        m["cost_exact"] = float(exact[s, t])
        m["route_exact"] = right.route(s, t)
        if method == "constant":
            rep = alg.constant_shift_report(g, exact)
            m.update({"const_wrong": rep["wrong"], "const_share_wrong": rep["share_wrong"], "const_excess": rep["mean_excess"], "shift": rep["shift"], "cost_found": float(jo.dist[s, t]), "route_found": jo.route(s, t)})
    return Analysis(net, jo, method, s, t, m, {"jo": t_jo, "bf": t_bf}, exact)


def verdict(a):
    """Code für die App: cycle / trap_wrong / trap_ok / no_negative / johnson_ok."""
    m = a.metrics
    if m["cycle"]:
        return "cycle"
    if a.method == "constant":
        return "trap_wrong" if m["const_wrong"] > 0 else "trap_ok"
    return "no_negative" if m["neg_edges"] == 0 else "johnson_ok"


def frames(a, max_dijkstra=40):
    """Die Bildfolge der Ansicht als Liste von (Art, Zahl): ("orig", 0), ("bf", Runde) ..., ("reweighted", 0) oder ("shift", 0), ("dijkstra", k) ..., ("result", 0) - bei einem negativen Zyklus endet sie mit ("cycle", 0).
    Bei großen Netzen zeigt die Ansicht höchstens `max_dijkstra` Zwischenstände der Festlegung."""
    jo = a.jo
    out = [("orig", 0)]
    if a.method == "johnson":
        rounds = len(jo.bf.history)
        out += [("bf", k) for k in range(1, rounds + 1)]
        if jo.negative_cycle:
            return out + [("cycle", 0)]
        out.append(("reweighted", 0))
    else:
        out.append(("shift", 0))
    settled = len(jo.order[jo.row(a.s)])
    ks = sorted({int(round(x)) for x in np.linspace(1, settled, min(settled, max_dijkstra))}) if settled else []
    out += [("dijkstra", k) for k in ks]
    out.append(("result", 0))
    return out


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def _row(g):
    r = alg.johnson(g)
    assert not r.negative_cycle
    _, nbf, cyc = fwm.all_pairs_by_bellman_ford(g)
    assert cyc == 0
    return {"n": g.n, "m": g.m, "johnson": r.counters["checks_total"], "bf_part": r.counters["bf_checks"], "dijkstra_part": r.counters["dijkstra_checks"], "fw": g.n ** 3, "nbf": nbf}


def _mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def effort_vs_degree(degrees=(2.0, 3.0, 6.0, 12.0), n=200, pot=C.DEFAULT_POT, seeds=C.SWEEP_SEEDS):
    """Zähler gegen die Dichte (Zufallsnetz mit Potenzialen, n Knoten): Johnson (Bellman-Ford ab q plus n-mal Dijkstra), Floyd-Warshall n³, n-mal Bellman-Ford."""
    return [{"degree": deg, **_mean_rows([_row(build_random(n, deg, pot, sd)) for sd in seeds])} for deg in degrees]


def effort_vs_size(sizes=(50, 100, 200, 400), degree=3.0, pot=C.DEFAULT_POT, seeds=C.SWEEP_SEEDS):
    return [{"size": n, **_mean_rows([_row(build_random(n, degree, pot, sd)) for sd in seeds])} for n in sizes]


def complete_graph_effort(n=60, pot=C.DEFAULT_POT, seeds=C.SWEEP_SEEDS):
    """Der Grenzfall dichtes Netz: vollständiger gerichteter Graph, Kosten 1 bis 9 plus Potenziale (also negative Kanten, nie ein Zyklus)."""
    rows = []
    for sd in seeds:
        rng = np.random.default_rng([int(sd), 909])
        p = np.random.default_rng([int(sd), 606]).integers(0, pot + 1, n)
        arcs = [(u, v, float(rng.integers(1, 10) + p[u] - p[v])) for u in range(n) for v in range(n) if u != v]
        rows.append(_row(from_arcs(n, arcs, np.zeros((n, 2)), directed=True)))
    return _mean_rows(rows)


def few_sources(ks=(1, 2, 5, 20, 100), n=200, degree=3.0, pot=C.DEFAULT_POT, seeds=C.SWEEP_SEEDS):
    """Wenige Startknoten (Zufallsnetz mit Potenzialen): Johnson (einmal Bellman-Ford ab q, dann k Dijkstra) gegen k-mal Bellman-Ford gegen Floyd-Warshall (immer n³). Die k Starts sind die ersten k Knoten."""
    rows = []
    for k in ks:
        acc = []
        for sd in seeds:
            g = build_random(n, degree, pot, sd)
            jo = alg.johnson(g, range(k))
            nbf = sum(jo_sp.bellman_ford(g, s, "early_stop").counters["checks"] for s in range(k))
            acc.append({"johnson": jo.counters["checks_total"], "bf_part": jo.counters["bf_checks"], "kbf": nbf, "fw": n ** 3})
        rows.append({"k": k, **_mean_rows(acc)})
    return rows


def constant_trap(pots=(2, 4, 8, 12, 20), n=100, degree=3.0, seeds=C.SWEEP_SEEDS):
    """Die Konstanten-Falle über die Potenzialspanne (Zufallsnetz mit Potenzialen): Anteil der Paare, deren gefundene Route nicht die billigste ist, und Anteil negativer Kanten."""
    rows = []
    for pot in pots:
        share, neg, exc = [], [], []
        for sd in seeds:
            g = build_random(n, degree, pot, sd)
            ex = alg.johnson(g).dist
            rep = alg.constant_shift_report(g, ex)
            share.append(rep["share_wrong"])
            exc.append(rep["mean_excess"])
            neg.append(float((g.weight < 0).mean()))
        rows.append({"pot": pot, "neg_share": float(np.mean(neg)), "share_wrong": float(np.mean(share)), "excess": float(np.mean(exc))})
    return rows


def constant_trap_by_hops(pot=C.DEFAULT_POT, n=100, degree=3.0, seeds=C.SWEEP_SEEDS, max_hops=6):
    """Die Falle nach der Kantenzahl der richtigen Route: je mehr Kanten, desto öfter liegt sie daneben (das Anheben bestraft jede Kante)."""
    wrong, total = np.zeros(max_hops + 1), np.zeros(max_hops + 1)
    for sd in seeds:
        g = build_random(n, degree, pot, sd)
        right = alg.johnson(g)
        fw = fwm.floyd_warshall(g, "numpy")
        hops = np.minimum(fwm.hop_matrix(fw), max_hops)
        bad = ~np.isclose(alg.johnson(g, None, "constant").dist, right.dist)
        off = ~np.eye(n, dtype=bool) & np.isfinite(right.dist)
        for h in range(1, max_hops + 1):
            sel = off & (hops == h)
            total[h] += sel.sum()
            wrong[h] += (sel & bad).sum()
    return [{"hops": h, "pairs": int(total[h]), "share_wrong": float(wrong[h] / total[h]) if total[h] else 0.0} for h in range(1, max_hops + 1)]
