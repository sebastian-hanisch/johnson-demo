"""Property-basierte Ergänzung zum festen Orakeltest (test_oracle_johnson.py): dasselbe Orakel (eigene Floyd-Warshall-Matrixrechnung, networkx für den negativen Zyklus, Erreichbarkeitsmatrix für die
Zähler), aber mit Hypothesis erzeugten gerichteten Graphen (2-8 Knoten, ganz- und halbzahlige Kosten, negative Kanten mit und ohne Zyklus, Nullkosten, Parallelkanten, Schleifen, Teilmengen von Startknoten)
und automatisch verkleinerten Gegenbeispielen. Deterministisch für die CI (derandomize, keine Beispieldatenbank)."""

import numpy as np
import pytest

pytest.importorskip("hypothesis")
nx = pytest.importorskip("networkx")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import jo_algorithm as alg  # noqa: E402
import jo_fw as fwm  # noqa: E402
from jo_graph import from_arcs, route_cost  # noqa: E402

CI = settings(max_examples=100, deadline=None, derandomize=True, database=None, suppress_health_check=[HealthCheck.too_slow])

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


@st.composite
def graphs(draw, only_acyclic_costs=False):
    """Gerichteter Graph; `free` (Kosten -3..7 in Halbschritten) enthält oft negative Zyklen, `potential` hat negative Kanten ohne Zyklus, `nonneg` Kosten >= 0 (auch Gleitkomma)."""
    n = draw(st.integers(2, 8))
    node = st.integers(0, n - 1)
    kind = draw(st.sampled_from(("potential", "nonneg", "float") if only_acyclic_costs else ("free", "potential", "nonneg", "float")))
    pot = draw(st.lists(st.integers(0, 8), min_size=n, max_size=n))
    arcs = []
    for u, v in draw(st.lists(st.tuples(node, node), max_size=3 * n)):
        if kind == "free":
            w = draw(st.integers(-6, 14)) / 2
        elif kind == "potential":
            w = float(draw(st.integers(1, 9)) + pot[u] - pot[v])
        elif kind == "float":
            w = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        else:
            w = float(draw(st.integers(0, 5)))
        arcs.append((u, v, w))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True, clean=draw(st.booleans()))


@CI
@given(g=graphs(), srcs=st.one_of(st.none(), st.lists(st.integers(0, 7), min_size=1, max_size=3)))
def test_johnson_equals_the_floyd_warshall_oracle_including_cycles_routes_potentials_and_counters(g, srcs):
    if srcs is not None:
        srcs = sorted({s % g.n for s in srcs})
    D = oracle_fw(g)
    cyc = bool((np.diag(D) < 0).any())
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            if not G.has_edge(u, v) or G[u][v]["weight"] > w:
                G.add_edge(u, v, weight=w)
    assert nx.negative_edge_cycle(G, weight="weight") == cyc                                    # das Orakel stimmt mit networkx überein
    jo = alg.johnson(g, srcs)
    assert jo.negative_cycle == cyc
    if cyc:
        c = jo.cycle
        cost = sum(float(g.weight[g.arc(a, b)]) for a, b in zip(c[:-1], c[1:]))
        assert c[0] == c[-1] and cost < 0 and cost == pytest.approx(jo.cycle_cost)
        return
    rows = list(range(g.n)) if srcs is None else srcs
    fin = np.isfinite(D[rows])
    assert np.array_equal(np.isfinite(jo.dist), fin) and np.allclose(jo.dist[fin], D[rows][fin], rtol=1e-9, atol=1e-9)
    assert (jo.reweighted.weight >= -1e-9).all() and np.allclose(jo.h, np.minimum(0, D.min(axis=0)))
    assert jo.tight_edges == int((jo.reweighted.weight == 0).sum())
    assert jo.counters["bf_checks"] == jo.counters["bf_rounds"] * (g.m + g.n)
    R, deg = reach(g), g.degree()
    assert jo.counters["dijkstra_checks"] == sum(int(deg[R[s]].sum()) for s in rows)
    for s in rows:
        for t in range(g.n):
            route = jo.route(s, t)
            if np.isfinite(D[s, t]):
                assert route[0] == s and route[-1] == t and route_cost(g, route) == pytest.approx(D[s, t], rel=1e-9, abs=1e-9)
            else:
                assert route == []


@CI
@given(g=graphs())
def test_the_constant_shift_trap_finds_a_route_that_is_optimal_after_the_shift_but_never_cheaper_than_the_truth(g):
    D = oracle_fw(g)
    if (np.diag(D) < 0).any():
        return
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
            assert route_cost(g, route) + const.shift * (len(route) - 1) == pytest.approx(shifted[s, t], rel=1e-9, abs=1e-9)
            assert route_cost(g, route) == pytest.approx(const.dist[s, t], rel=1e-9, abs=1e-9) and const.dist[s, t] >= D[s, t] - 1e-9


@CI
@given(g=graphs(), variant=st.sampled_from(fwm.VARIANTS))
def test_floyd_warshall_variants_counters_and_affected_pairs_equal_the_oracle(g, variant):
    D = oracle_fw(g)
    neg = np.diag(D) < 0
    fw = fwm.floyd_warshall(g, variant)
    assert fw.negative_cycle == bool(neg.any())
    if neg.any():
        R = reach(g)
        want = np.array([[any(R[i, x] and R[x, j] and neg[x] for x in range(g.n)) for j in range(g.n)] for i in range(g.n)])
        assert np.array_equal(fwm.affected_pairs(fw), want)
        return
    assert np.allclose(fw.dist[np.isfinite(D)], D[np.isfinite(D)], rtol=1e-9, atol=1e-9) and np.array_equal(np.isfinite(fw.dist), np.isfinite(D))
    assert variant == "skip" or fw.counters["comparisons"] == g.n ** 3
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
                assert route[0] == i and route[-1] == j and route_cost(g, route) == pytest.approx(D[i, j], rel=1e-9, abs=1e-9) and hop[i, j] == len(route) - 1
            else:
                assert route == [] and hop[i, j] == -1
    for o in ("kij", "kji"):
        other = fwm.floyd_warshall_order(g, o)
        assert np.array_equal(np.isfinite(other), np.isfinite(D)) and np.allclose(other[np.isfinite(D)], D[np.isfinite(D)], rtol=1e-9, atol=1e-9)
