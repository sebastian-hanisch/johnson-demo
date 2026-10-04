"""Unabhängiges Orakel für Johnson und die Floyd-Warshall-Bausteine: eine eigene Matrixrechnung (Floyd-Warshall) und networkx (negativer Zyklus) auf Zufallsgraphen mit Gleichständen, Nullkosten, negativen Kanten
mit und ohne Zyklus, Teilmengen von Startknoten. Geprüft werden Entfernungen, Zyklus-Meldung, Routen, Potenziale, Zähler (Bellman-Ford ab q = Runden mal (m + n), Dijkstra = Summe der Grade erreichbarer Knoten),
die Falle (gefundene Route ist eine nach dem Anheben kürzeste Route) und die Matrixzähler und betroffenen Paare von Floyd-Warshall."""

import networkx as nx
import numpy as np
import pytest

import jo_algorithm as alg
import jo_fw as fwm
from jo_graph import from_arcs, route_cost

INF = float("inf")


def oracle_fw(g):
    D = np.full((g.n, g.n), INF)
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            D[u, v] = min(D[u, v], w)
    np.fill_diagonal(D, np.minimum(np.diag(D), 0.0))
    for k in range(g.n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def reach(g):
    R = np.eye(g.n, dtype=bool)
    for u in range(g.n):
        for v in g.out(u):
            R[u, v] = True
    for k in range(g.n):
        R = R | (R[:, k:k + 1] & R[k:k + 1, :])
    return R


def random_graph(rng, it, low=-2):
    n = int(rng.integers(1, 10))
    pot = rng.integers(0, 8, n)
    arcs = []
    for _ in range(int(rng.integers(0, 3 * n + 1))):
        u, v = (int(x) for x in rng.integers(0, n, 2))
        c = [int(rng.integers(1, 6)), int(rng.integers(1, 10)) + int(pot[u]) - int(pot[v]), int(rng.integers(low, 7)), float(rng.integers(-3, 8)) / 2][it % 4]
        arcs.append((u, v, float(c)))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True, clean=bool(it % 2))


def test_johnson_equals_the_floyd_warshall_oracle_including_cycles_routes_potentials_and_counters():
    rng = np.random.default_rng(5)
    cycles = solved = 0
    for it in range(300):
        g = random_graph(rng, it)
        D = oracle_fw(g)
        cyc = bool((np.diag(D) < 0).any())
        G = nx.DiGraph()
        G.add_nodes_from(range(g.n))
        for u in range(g.n):
            for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
                if not G.has_edge(u, v) or G[u][v]["weight"] > w:
                    G.add_edge(u, v, weight=w)
        assert nx.negative_edge_cycle(G, weight="weight") == cyc                                # das Orakel stimmt mit networkx überein
        srcs = None if it % 3 else sorted({int(x) for x in rng.integers(0, g.n, 3)})
        jo = alg.johnson(g, srcs)
        assert jo.negative_cycle == cyc
        if cyc:
            cycles += 1
            c = jo.cycle
            cost = sum(float(g.weight[g.arc(a, b)]) for a, b in zip(c[:-1], c[1:]))
            assert c[0] == c[-1] and cost < 0 and cost == pytest.approx(jo.cycle_cost)
            continue
        solved += 1
        rows = list(range(g.n)) if srcs is None else srcs
        assert np.array_equal(np.isfinite(jo.dist), np.isfinite(D[rows])) and np.allclose(jo.dist[np.isfinite(D[rows])], D[rows][np.isfinite(D[rows])])
        assert (jo.reweighted.weight >= -1e-9).all() and np.allclose(jo.h, np.minimum(0, D.min(axis=0)))
        assert jo.tight_edges == int((jo.reweighted.weight == 0).sum())
        assert jo.counters["bf_checks"] == jo.counters["bf_rounds"] * (g.m + g.n)
        R, deg = reach(g), g.degree()
        assert jo.counters["dijkstra_checks"] == sum(int(deg[R[s]].sum()) for s in rows)
        for s in rows:
            for t in range(g.n):
                route = jo.route(s, t)
                if np.isfinite(D[s, t]):
                    assert route[0] == s and route[-1] == t and route_cost(g, route) == pytest.approx(D[s, t])
                else:
                    assert route == []
    assert cycles > 10 and solved > 100


def test_the_constant_shift_trap_finds_a_route_that_is_optimal_after_the_shift_but_never_cheaper_than_the_truth():
    rng = np.random.default_rng(6)
    for it in range(120):
        g = random_graph(rng, it)
        D = oracle_fw(g)
        if (np.diag(D) < 0).any() or g.n < 2:
            continue
        const = alg.johnson(g, None, "constant")
        shifted = oracle_fw(from_arcs(g.n, [(u, int(v), float(w) + const.shift) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u))], np.zeros((g.n, 2)), directed=True, clean=False))
        for s in range(g.n):
            for t in range(g.n):
                if s == t:
                    continue
                if not np.isfinite(D[s, t]):
                    assert not np.isfinite(const.dist[s, t])
                    continue
                route = const.route(s, t)
                assert route_cost(g, route) + const.shift * (len(route) - 1) == pytest.approx(shifted[s, t])
                assert route_cost(g, route) == pytest.approx(const.dist[s, t]) and const.dist[s, t] >= D[s, t] - 1e-9


def test_floyd_warshall_variants_counters_and_affected_pairs_equal_the_oracle():
    rng = np.random.default_rng(7)
    for it in range(150):
        g = random_graph(rng, it)
        D = oracle_fw(g)
        neg = np.diag(D) < 0
        for variant in fwm.VARIANTS:
            fw = fwm.floyd_warshall(g, variant)
            assert fw.negative_cycle == bool(neg.any())
            if neg.any():
                R = reach(g)
                want = np.array([[any(R[i, x] and R[x, j] and neg[x] for x in range(g.n)) for j in range(g.n)] for i in range(g.n)])
                assert np.array_equal(fwm.affected_pairs(fw), want)
                continue
            assert np.array_equal(fw.dist, D) and (variant == "skip" or fw.counters["comparisons"] == g.n ** 3)
            d, _ = fwm.initial_matrix(g)
            total = 0
            for k in range(g.n):
                total += int(np.isfinite(d[:, k]).sum()) * int(np.isfinite(d[k, :]).sum())
                d = np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
            assert variant == "classic" or fw.counters["skip_comparisons"] == total
            hop = fwm.hop_matrix(fw)
            for i in range(g.n):
                for j in range(g.n):
                    route = fw.route(i, j)
                    if np.isfinite(D[i, j]):
                        assert route[0] == i and route[-1] == j and route_cost(g, route) == pytest.approx(D[i, j]) and hop[i, j] == len(route) - 1
                    else:
                        assert route == [] and hop[i, j] == -1
        if not neg.any():
            assert all(np.array_equal(fwm.floyd_warshall_order(g, o), D) for o in ("kij", "kji"))
