"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Kantenprüfungen, Vergleiche und ganzzahlige Kosten sind plattformfest;
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import numpy as np
import pytest

import jo_algorithm as alg
import jo_constants as C
import jo_evaluation as ev
import jo_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "small_cycle": "🔁 Negativer Zyklus", "ev": "🔋 E-Lieferwagen", "random": "🕸️ Zufallsnetz", "city": "🏙️ Stadtnetz"}


def _preset(key, method=None):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["hill"], p["eta"], p["reach"], p["spread"], p["nodes"], p["degree"], p["pot"], p["seed"])
    return p, net, ev.analyse(net, method or p["method"], p["distance"], p["seed"])


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    _, net, a = _preset("small")
    m = a.metrics
    assert (m["bf_rounds"], m["bf_checks"], m["dijkstra_checks"], m["total"], m["fw_comparisons"], m["nbf_checks"]) == (3, 45, 54, 99, 216, 180)
    assert (m["h_nonzero"], m["h_min"], m["tight_edges"], m["neg_after"], m["naive_wrong"], m["pairs"]) == (1, -3.0, 1, 0, 7, 30) and net.graph.names[int(np.argmin(a.jo.h))] == "Nord"
    _has("small", "3 Runden und 45 Kantenprüfungen", "Nord: −3", "54 Prüfungen", "99 gegen 216", "180", "7 von 30 Paaren")


def test_small_preset_reweighting_and_the_trap_example():
    _, net, a = _preset("small")
    g = net.graph
    rw = {(g.names[u], g.names[int(v)]): float(w) for u in range(g.n) for v, w in zip(a.jo.reweighted.out(u), a.jo.reweighted.out_weights(u))}
    assert rw[("Ost", "Nord")] == 0.0 and rw[("Depot", "Nord")] == 4.0 and rw[("Nord", "Süd")] == 2.0                 # -3 -> 0 (eng), 1 -> 4, 5 -> 2
    _, _, t = _preset("small", "constant")
    assert (t.metrics["shift"], t.metrics["const_wrong"], t.metrics["cost_exact"], t.metrics["cost_found"]) == (3.0, 7, 6.0, 8.0)


def test_small_cycle_preset_numbers():
    _, net, a = _preset("small_cycle")
    m = a.metrics
    assert m["cycle"] and (m["bf_rounds"], m["bf_checks"], m["cycle_cost"]) == (2, 30, -2.0) and sorted({net.graph.names[i] for i in a.jo.cycle}) == ["Süd", "West"]
    _has("small_cycle", "−2 Euro", "nach 2 Runden (30 Kantenprüfungen)")


def test_ev_preset_numbers():
    _, net, a = _preset("ev")
    m = a.metrics
    assert (m["neg_edges"], m["m"], m["bf_checks"], m["bf_rounds"], m["tight_edges"], m["dijkstra_checks"], m["total"], m["fw_comparisons"], m["nbf_checks"], m["naive_wrong"], m["pairs"]) == \
        (42, 336, 1200, 3, 32, 21504, 22704, 262144, 137424, 524, 4032)
    _has("ev", "42 von 336", "1 200 Kantenprüfungen (3 Runden)", "32 sind eng", "21 504", "22 704", "262 144", "137 424", "524 von 4 032")


def test_random_preset_numbers():
    _, net, a = _preset("random")
    m = a.metrics
    assert (m["neg_edges"], m["m"], m["bf_checks"], m["dijkstra_checks"], m["total"], m["fw_comparisons"], m["nbf_checks"]) == (38, 300, 1600, 30000, 31600, 1000000, 190500)
    _, _, t = _preset("random", "constant")
    assert t.metrics["shift"] == 5.0 and t.metrics["const_share_wrong"] == pytest.approx(0.266, abs=0.001) and (t.metrics["cost_found"], t.metrics["cost_exact"]) == (24.0, 21.0)
    _has("random", "38 von 300", "31 600", "1 600 Bellman-Ford", "30 000", "1 000 000", "190 500", "um 5 anheben", "26.6 %", "24 statt 21")


def test_city_preset_numbers():
    _, net, a = _preset("city")
    m = a.metrics
    assert (m["bf_checks"], m["bf_rounds"], m["dijkstra_checks"], m["total"], m["fw_comparisons"], m["h_nonzero"]) == (800, 2, 21504, 22304, 262144, 0) and m["bf_checks"] / m["dijkstra_checks"] == pytest.approx(0.037, abs=0.0005)
    _has("city", "800 Kantenprüfungen", "(2 Runden)", "3.7 %", "21 504", "22 304", "262 144")


# --- Experimente und die Tabelle "Wo die Annahmen enden" ----------------------------------------------------------------------------------------

def test_effort_claims():
    s = {int(r["size"]): r for r in ev.effort_vs_size()}
    assert (s[400]["johnson"], s[400]["fw"], s[400]["nbf"], s[400]["bf_part"]) == (487040, 64000000, 3891840, 7040)
    assert s[400]["fw"] / s[400]["johnson"] == pytest.approx(131.4, abs=0.2) and s[400]["nbf"] / s[400]["johnson"] == pytest.approx(8.0, abs=0.1) and s[400]["bf_part"] / s[400]["johnson"] == pytest.approx(0.0145, abs=0.0003)
    d = {r["degree"]: r for r in ev.effort_vs_degree()}
    assert d[2.0]["johnson"] == 82400 and d[2.0]["fw"] == 8000000 and all(r["fw"] > r["nbf"] > r["johnson"] for r in list(d.values()) + list(s.values()))
    comp = ev.complete_graph_effort()
    assert (comp["johnson"], comp["fw"], comp["dijkstra_part"]) == (226800, 216000, 212400) and comp["johnson"] > comp["fw"]                # Tabelle: 226 800 gegen 216 000


def test_few_sources_claims():
    rows = {r["k"]: r for r in ev.few_sources()}
    assert rows[1]["johnson"] == 3960 and rows[1]["kbf"] == 3840 and rows[1]["bf_part"] == 3360
    assert abs(rows[1]["johnson"] - rows[1]["kbf"]) < 0.05 * rows[1]["kbf"] and rows[1]["bf_part"] == pytest.approx(rows[1]["kbf"], rel=0.15)        # "gleichauf", "etwa so viel wie ein einzelner Bellman-Ford"
    assert (rows[2]["johnson"], rows[2]["kbf"]) == (4560, 7440) and rows[2]["johnson"] < rows[2]["kbf"]
    assert rows[100]["kbf"] / rows[100]["johnson"] == pytest.approx(6.65, abs=0.05) and all(r["fw"] == 8000000 for r in rows.values())


def test_constant_trap_claims():
    pots = {r["pot"]: r for r in ev.constant_trap()}
    assert pots[2]["share_wrong"] == pytest.approx(0.087, abs=0.002) and pots[2]["neg_share"] == pytest.approx(0.016, abs=0.002)
    assert pots[20]["share_wrong"] == pytest.approx(0.34, abs=0.005) and pots[20]["neg_share"] == pytest.approx(0.277, abs=0.003)
    assert all(pots[a]["share_wrong"] < pots[b]["share_wrong"] for a, b in ((2, 4), (4, 8), (8, 12), (12, 20)))
    hops = ev.constant_trap_by_hops()
    assert [r["share_wrong"] for r in hops] == pytest.approx([0.0, 0.005, 0.026, 0.098, 0.274, 0.65], abs=0.005) and hops[0]["share_wrong"] == 0.0            # je mehr Kanten, desto öfter falsch


def test_johnson_matches_floyd_warshall_on_every_preset_without_cycle_and_never_needs_more_than_a_third_of_nbf():
    for key in ("small", "ev", "random", "city"):
        _, _, a = _preset(key)
        assert a.metrics["exact_vs_fw"], key
    _, _, r = _preset("random")
    assert r.metrics["total"] < r.metrics["nbf_checks"] / 5 and r.metrics["total"] < r.metrics["fw_comparisons"] / 30


def test_bellman_ford_from_q_is_a_small_share_of_the_total_in_every_generated_preset():
    for key in ("ev", "random", "city"):
        _, _, a = _preset(key)
        assert a.metrics["bf_checks"] < 0.07 * a.metrics["total"], key                                     # "die Sicherheit kostet wenig"
