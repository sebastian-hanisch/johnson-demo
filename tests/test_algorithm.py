"""Johnson (Potenziale aus Bellman-Ford, Umgewichten, n-mal Dijkstra, Zurückrechnen) und die Konstanten-Falle gegen Floyd-Warshall, networkx und unabhängige Rechnungen."""

import networkx as nx
import numpy as np
import pytest

import jo_algorithm as alg
import jo_fw as fwm
import jo_scenario as sc
from jo_graph import from_arcs, route_cost

INF = float("inf")


def _random_arcs(n, m, seed, kind):
    """kind: 'positive', 'potential' (negative Kanten, nie ein Zyklus), 'free' (beliebige Kosten, meist mit negativen Zyklen)."""
    rng = np.random.default_rng(seed)
    pot = rng.integers(0, 15, n)
    arcs = {}
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u == v or (u, v) in arcs:
            continue
        c = int(rng.integers(1, 10))
        if kind == "potential":
            c = c + int(pot[u]) - int(pot[v])
        elif kind == "free":
            c = int(rng.integers(-3, 10))
        arcs[(u, v)] = c
    return [(u, v, float(c)) for (u, v), c in arcs.items()]


def _graph(n, arcs):
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True)


def _nx(n, arcs):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v, c in arcs:
        G.add_edge(u, v, weight=c)
    return G


CASES = [(n, m, seed, kind) for kind in ("positive", "potential", "free") for n, m, seed in ((6, 12, 1), (12, 30, 2), (20, 40, 3), (25, 30, 4), (10, 9, 5))]


@pytest.mark.parametrize("n,m,seed,kind", CASES)
def test_matches_floyd_warshall_and_networkx_and_detects_the_same_cycles(n, m, seed, kind):
    arcs = _random_arcs(n, m, seed, kind)
    g, G = _graph(n, arcs), _nx(n, arcs)
    r = alg.johnson(g)
    assert r.negative_cycle == nx.negative_edge_cycle(G) == fwm.floyd_warshall(g).negative_cycle
    if r.negative_cycle:
        assert r.cycle[0] == r.cycle[-1] and route_cost(g, r.cycle) == pytest.approx(r.cycle_cost) and r.cycle_cost < 0 and r.dist is None
        return
    ref = np.array(nx.floyd_warshall_numpy(G, weight="weight"))
    assert np.array_equal(np.isfinite(ref), np.isfinite(r.dist)) and np.allclose(ref[np.isfinite(ref)], r.dist[np.isfinite(ref)])
    for s in range(n):
        for t in range(n):
            route = r.route(s, t)
            if np.isfinite(ref[s, t]):
                assert route[0] == s and route[-1] == t and route_cost(g, route) == pytest.approx(ref[s, t])
            else:
                assert route == []


@pytest.mark.parametrize("kind", ("positive", "potential"))
@pytest.mark.parametrize("seed", range(4))
def test_potentials_are_nonpositive_feasible_and_make_every_cost_nonnegative(kind, seed):
    g = _graph(18, _random_arcs(18, 50, seed, kind))
    r = alg.johnson(g)
    assert (r.h <= 0).all() and alg.is_feasible(g, r.h) and (r.reweighted.weight >= 0).all()
    assert r.tight_edges == int((r.reweighted.weight == 0).sum())
    if kind == "positive":
        assert (r.h == 0).all() and np.array_equal(r.reweighted.weight, g.weight) and r.tight_edges == 0     # nichts zu tun: h = 0, keine enge Kante
    else:
        assert (r.h < 0).any() and r.tight_edges > 0                                                        # negative Kanten: der Kürzeste-Wege-Baum von q besteht aus engen Kanten


@pytest.mark.parametrize("seed", range(4))
def test_every_route_changes_its_cost_by_the_same_constant(seed):
    g = _graph(15, _random_arcs(15, 45, seed, "potential"))
    r = alg.johnson(g)
    fw = fwm.floyd_warshall(g)
    for s in range(15):
        for t in range(15):
            route = fw.route(s, t)
            if len(route) > 1:
                assert route_cost(r.reweighted, route) == pytest.approx(route_cost(g, route) + r.h[s] - r.h[t])


def test_any_feasible_potential_gives_the_same_matrix_including_the_hidden_ones_of_the_generator():
    n, span, seed = 60, 8, 3
    g = sc.build_random(n, 3.0, span, seed)
    hidden = -sc.hidden_potentials(n, span, seed)
    assert alg.is_feasible(g, hidden) and (alg.reweight(g, hidden).weight >= 1).all()                      # das versteckte Potenzial liefert die ursprünglichen Kosten 1 bis 9
    ref = alg.johnson(g)
    assert not np.allclose(ref.h, hidden)                                                                   # ein anderes Potenzial als das von Bellman-Ford
    rw = alg.reweight(g, hidden)
    import jo_sp
    for s in range(0, n, 7):
        d = jo_sp.dijkstra(rw, s).dist
        fin = np.isfinite(d)
        assert np.allclose(d[fin] - hidden[s] + hidden[np.where(fin)[0]], ref.dist[s][fin])


def test_a_subset_of_sources_gives_exactly_those_rows_and_costs_fewer_checks():
    g = _graph(20, _random_arcs(20, 60, 7, "potential"))
    full, part = alg.johnson(g), alg.johnson(g, [3, 11, 17])
    assert part.sources == [3, 11, 17] and np.array_equal(part.dist, full.dist[[3, 11, 17]])
    assert part.counters["bf_checks"] == full.counters["bf_checks"] and part.counters["dijkstra_checks"] < full.counters["dijkstra_checks"]
    assert part.route(3, 11) == full.route(3, 11)


def test_counters_add_up():
    g = _graph(16, _random_arcs(16, 40, 2, "potential"))
    r = alg.johnson(g)
    assert r.counters["bf_checks"] == r.counters["bf_rounds"] * (g.m + g.n)                                  # jede Runde prüft alle Kanten des erweiterten Graphen
    import jo_sp
    assert r.counters["dijkstra_checks"] == sum(jo_sp.dijkstra(g, s).counters["checks"] for s in range(16))                         # die Struktur (erreichbare Kanten) ändert sich nicht
    assert r.counters["checks_total"] == r.counters["bf_checks"] + r.counters["dijkstra_checks"]


def test_virtual_source_reaches_everything_and_keeps_the_distances():
    g = _graph(10, _random_arcs(10, 25, 1, "potential"))
    aug = alg.add_virtual_source(g)
    assert aug.n == 11 and aug.m == g.m + 10 and aug.degree()[10] == 10 and (aug.indices == 10).sum() == 0
    named = sc.small_network()
    assert alg.add_virtual_source(named.graph).names[-1] == "q"


def test_unreachable_pairs_zero_edges_single_node():
    g = _graph(4, [(0, 1, 0.0), (1, 0, 0.0), (1, 2, 4.0)])
    r = alg.johnson(g)
    assert list(r.dist[0]) == [0.0, 0.0, 4.0, INF] and r.route(0, 3) == [] and r.route(0, 2) == [0, 1, 2] and r.route(2, 2) == [2]
    solo = alg.johnson(_graph(1, []))
    assert solo.dist.tolist() == [[0.0]] and not solo.negative_cycle


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        alg.johnson(_graph(2, []), method="astar")


# --- Die Falle: alle Kosten um eine Konstante anheben ------------------------------------------------------------------------------------------

def test_constant_shift_is_wrong_on_a_built_counterexample_and_johnson_is_right():
    # A: 0 -> 1 -> 2 -> 3 kostet 1 + 1 - 1 = 1 (drei Kanten); B: 0 -> 3 direkt kostet 2. Um M = 1 angehoben: A = 4, B = 3 - die Falle wählt B.
    g = _graph(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, -1.0), (0, 3, 2.0)])
    right, wrong = alg.johnson(g), alg.johnson(g, method="constant")
    assert right.dist[0, 3] == 1.0 and wrong.dist[0, 3] == 2.0 and wrong.shift == 1.0 and right.route(0, 3) == [0, 1, 2, 3] and wrong.route(0, 3) == [0, 3]


def test_constant_shift_report_and_no_shift_without_negative_edges():
    g = _graph(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, -1.0), (0, 3, 2.0)])
    exact = alg.johnson(g).dist
    rep = alg.constant_shift_report(g, exact)
    assert rep["wrong"] >= 1 and 0 < rep["share_wrong"] < 1 and rep["mean_excess"] > 0
    pos = _graph(10, _random_arcs(10, 30, 1, "positive"))
    ok = alg.constant_shift_report(pos, alg.johnson(pos).dist)
    assert ok["shift"] == 0.0 and ok["wrong"] == 0                                                          # ohne negative Kanten ist die "Falle" gar keine


def test_constant_shift_costs_are_real_costs_of_the_found_route():
    g = _graph(20, _random_arcs(20, 60, 5, "potential"))
    wrong = alg.johnson(g, method="constant")
    exact = alg.johnson(g).dist
    assert (wrong.dist[np.isfinite(exact)] >= exact[np.isfinite(exact)] - 1e-9).all()                       # nie besser als das Optimum
    for s, t in ((0, 5), (3, 9), (7, 2)):
        route = wrong.route(s, t)
        if route:
            assert route_cost(g, route) == pytest.approx(wrong.dist[s, t])
