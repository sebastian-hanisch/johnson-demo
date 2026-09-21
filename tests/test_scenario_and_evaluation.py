"""Netze (kleines Netz, Zyklus-Variante, E-Lieferwagen, Zufallsnetz, Stadtnetz), Zielwahl, Kennzahlen, Bildfolge, Experimente."""

import networkx as nx
import numpy as np
import pytest

import jo_algorithm as alg
import jo_constants as C
import jo_evaluation as ev
import jo_fw as fwm
import jo_scenario as sc


def _nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


def _strong(g):
    return nx.number_strongly_connected_components(_nx(g))


# --- Netze -----------------------------------------------------------------------------------------------------------------------------------

def test_small_network_is_the_one_from_the_floyd_warshall_demo():
    net = sc.small_network()
    g = net.graph
    assert g.n == 6 and g.m == 9 and g.directed and _strong(g) == 1
    assert [(g.names[u], g.names[int(v)], float(w)) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u)) if w < 0] == [("Ost", "Nord", -3.0)]


def test_cycle_variant_has_the_negative_cycle_between_sued_and_west():
    net = sc.small_network(True)
    r = alg.johnson(net.graph)
    assert r.negative_cycle and sorted({net.graph.names[i] for i in r.cycle}) == ["Süd", "West"] and r.cycle_cost == -2.0


@pytest.mark.parametrize("seed", range(3))
def test_ev_network_never_has_a_negative_cycle(seed):
    net = sc.ev_network(7, 40, 90, seed)
    assert not alg.johnson(net.graph).negative_cycle and (net.graph.weight < 0).any() and _strong(net.graph) == 1


def test_random_network_hidden_potentials_reproduce_the_original_costs():
    n, span, seed = 80, 8, 2
    g = sc.build_random(n, 3.0, span, seed)
    h = -sc.hidden_potentials(n, span, seed)
    assert (alg.reweight(g, h).weight >= 1).all() and (alg.reweight(g, h).weight <= 9).all() and (g.weight < 0).any()
    assert (sc.hidden_potentials(n, 0, seed) == 0).all()


def test_make_network_rejects_unknown_nets():
    with pytest.raises(ValueError):
        sc.make_network("ring")


# --- Kennzahlen -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
@pytest.mark.parametrize("method", C.METHODS)
def test_analysis_invariants_for_every_net_and_method(key, method):
    net = sc.make_network(key, side=5, nodes=40)
    a = ev.analyse(net, method)
    m, g = a.metrics, net.graph
    assert m["cycle"] == nx.negative_edge_cycle(_nx(g)) == (key == "small_cycle") and m["fw_comparisons"] == g.n ** 3 and m["cells"] == g.n ** 2
    assert ev.verdict(a) in ("cycle", "johnson_ok", "no_negative", "trap_wrong", "trap_ok")
    if m["cycle"]:
        assert ev.verdict(a) == "cycle" and m["cycle_nodes"] >= 2 and a.exact is None
        return
    ref = np.array(nx.floyd_warshall_numpy(_nx(g), weight="weight"))
    assert np.array_equal(np.isfinite(ref), np.isfinite(a.exact)) and np.allclose(ref[np.isfinite(ref)], a.exact[np.isfinite(ref)]) and m["exact_vs_fw"]
    assert m["neg_after"] == 0 and m["total"] == m["bf_checks"] + m["dijkstra_checks"] and m["cost_exact"] == pytest.approx(ref[a.s, a.t])
    if method == "constant":
        assert m["const_wrong"] >= 0 and m["cost_found"] >= m["cost_exact"] - 1e-9
    if key == "city":
        assert m["neg_edges"] == 0 and m["h_nonzero"] == 0 and ev.verdict(a) in ("no_negative", "trap_ok")


def test_verdicts_of_the_presets():
    kinds = {}
    for name, p in C.PRESETS.items():
        net = sc.make_network(p["net"], p["side"], p["hill"], p["eta"], p["reach"], p["spread"], p["nodes"], p["degree"], p["pot"], p["seed"])
        kinds[p["net"]] = ev.verdict(ev.analyse(net, p["method"], p["distance"], p["seed"]))
    assert kinds == {"small": "johnson_ok", "small_cycle": "cycle", "ev": "johnson_ok", "random": "johnson_ok", "city": "no_negative"}


def test_pick_pair_uses_the_fixed_task_for_small_nets_and_the_distance_rank_otherwise():
    small = sc.small_network()
    assert ev.pick_pair(small, lambda s: np.zeros(6), 10, 1) == (0, 5) == ev.pick_pair(small, lambda s: np.zeros(6), 99, 5)
    net = sc.make_network("city", side=8)
    exact = alg.johnson(net.graph).dist
    pairs = {p: ev.pick_pair(net, lambda s: exact[s], p, 7) for p in (10, 50, 100)}
    s = pairs[10][0]
    assert len({a for a, _ in pairs.values()}) == 1 and exact[s, pairs[10][1]] < exact[s, pairs[50][1]] <= exact[s, pairs[100][1]] and exact[s, pairs[100][1]] == np.max(exact[s])


# --- Bildfolge ---------------------------------------------------------------------------------------------------------------------------------

def test_frames_for_johnson_the_trap_and_a_cycle():
    a = ev.analyse(sc.small_network())
    f = ev.frames(a)
    kinds = [k for k, _ in f]
    assert kinds[0] == "orig" and kinds[-1] == "result" and kinds.count("reweighted") == 1 and kinds.count("bf") == a.metrics["bf_rounds"] and kinds.count("dijkstra") == 6
    assert [k for k, i in f if k == "dijkstra"] == ["dijkstra"] * 6 and [i for k, i in f if k == "dijkstra"] == list(range(1, 7))
    trap = ev.frames(ev.analyse(sc.small_network(), "constant"))
    assert [k for k, _ in trap][:2] == ["orig", "shift"] and "bf" not in [k for k, _ in trap] and trap[-1][0] == "result"
    cyc = ev.frames(ev.analyse(sc.small_network(True)))
    assert cyc[-1] == ("cycle", 0) and "dijkstra" not in [k for k, _ in cyc]


def test_large_nets_show_at_most_forty_dijkstra_frames_ending_at_the_full_order():
    a = ev.analyse(sc.make_network("ev", side=12))
    f = [(k, i) for k, i in ev.frames(a) if k == "dijkstra"]
    assert len(f) <= 40 and f[-1][1] == 144 and [i for _, i in f] == sorted({i for _, i in f})


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

def test_effort_rows_order_the_counters_as_expected():
    rows = ev.effort_vs_degree(degrees=(2.0, 6.0), n=60, seeds=C.SWEEP_SEEDS[:2])
    for r in rows:
        assert r["fw"] == r["n"] ** 3 and r["fw"] > r["nbf"] > r["johnson"] and r["johnson"] == r["bf_part"] + r["dijkstra_part"] and r["bf_part"] < 0.15 * r["johnson"]
    sizes = ev.effort_vs_size(sizes=(20, 40), seeds=C.SWEEP_SEEDS[:2])
    assert sizes[1]["fw"] == 8 * sizes[0]["fw"] and sizes[1]["fw"] / sizes[1]["johnson"] > sizes[0]["fw"] / sizes[0]["johnson"]


def test_complete_graph_makes_floyd_warshall_win_narrowly():
    r = ev.complete_graph_effort(n=30, seeds=C.SWEEP_SEEDS[:2])
    assert r["dijkstra_part"] == 30 * 30 * 29 and r["fw"] == 30 ** 3 and r["johnson"] > r["fw"]


def test_few_sources_rows():
    rows = ev.few_sources(ks=(1, 5), n=60, seeds=C.SWEEP_SEEDS[:2])
    assert rows[0]["bf_part"] == rows[1]["bf_part"] and rows[1]["johnson"] < rows[1]["kbf"] and rows[1]["johnson"] > rows[0]["johnson"] and rows[0]["fw"] == 60 ** 3


def test_constant_trap_rows():
    pots = ev.constant_trap(pots=(2, 12), n=60, seeds=C.SWEEP_SEEDS[:2])
    assert pots[0]["share_wrong"] < pots[1]["share_wrong"] and pots[0]["neg_share"] < pots[1]["neg_share"]
    hops = ev.constant_trap_by_hops(n=60, seeds=C.SWEEP_SEEDS[:2], max_hops=5)
    assert [r["hops"] for r in hops] == [1, 2, 3, 4, 5] and hops[0]["share_wrong"] == 0 and hops[-1]["share_wrong"] > hops[1]["share_wrong"]
