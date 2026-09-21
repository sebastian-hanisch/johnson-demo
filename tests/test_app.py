"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, beide Umgewichtungen, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import jo_constants as C
from jo_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
# Urteil je Preset: success = Johnson liefert die Matrix, n-mal Dijkstra ohne Umgewichten läge daneben; warning = negativer Zyklus; info = keine negativen Kanten (gemessen, siehe test_claims)
EXPECTED_KIND = {"🔀 Kleines Netz": "success", "🔁 Negativer Zyklus": "warning", "🔋 E-Lieferwagen": "success", "🕸️ Zufallsnetz": "success", "🏙️ Stadtnetz": "info"}


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _kinds(at):
    return {"success": len(at.success), "warning": len(at.warning), "info": len(at.info)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception():
    at = _run()
    assert any("Johnson in Aktion" in m.value for m in at.markdown)
    assert _kinds(at) == {"success": 1, "warning": 0, "info": 0}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    want = {"success": 0, "warning": 0, "info": 0}
    want[EXPECTED_KIND[name]] = 1
    assert _kinds(at) == want


@pytest.mark.parametrize("method", C.METHODS)
@pytest.mark.parametrize("net", C.NETS)
def test_every_method_renders_on_every_net_with_exactly_one_verdict(method, net):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["method_select"] = method
        at.session_state["side_slider"] = 6
        at.session_state["nodes_slider"] = 60
    at = _run(setup)
    assert sum(_kinds(at).values()) == 1
    if net == "small_cycle":
        assert len(at.warning) == 1 and "Negativer Zyklus" in at.warning[0].value


def test_the_trap_warns_on_the_random_preset_and_names_the_shift():
    def setup(at):
        _apply(at, C.PRESETS["🕸️ Zufallsnetz"])
        at.session_state["method_select"] = "constant"
    at = _run(setup)
    assert len(at.warning) == 1 and "Die Falle" in at.warning[0].value and "angehoben" in at.warning[0].value


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big_ev(at):
        at.session_state["net_select"] = "ev"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["hill_slider"] = C.HILL_MAX
        at.session_state["eta_slider"] = C.ETA_MAX

    def dense(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = 100
        at.session_state["degree_slider"] = C.DEGREE_MAX
        at.session_state["pot_slider"] = C.POT_MAX

    def flat(at):
        at.session_state["net_select"] = "ev"
        at.session_state["hill_slider"] = C.HILL_MIN
    for setup in (small, big_ev, dense, flat):
        at = _run(setup)
        assert at.slider(key="jo_step").value == at.slider(key="jo_step").max


def test_hidden_controls_follow_the_net():
    small, small_c, city, ev, rnd = (_labels(_run(net=n)) for n in ("small", "small_cycle", "city", "ev", "random"))
    assert small == small_c == {"Netz", "Umgewichtung"}                                                 # feste Aufgabe: kein Paar, kein Seed
    common = {"Netz", "Umgewichtung", "Entfernung Start–Ziel [%]", "Zufalls-Seed"}
    assert city == common | {"Kreuzungen je Seite", "Reichweite der Straßen [Blocklängen]", "Streuung der Kosten"}
    assert ev == common | {"Kreuzungen je Seite", "Hügel [m Höhenunterschied]", "Rückgewinnung bergab [%]"}
    assert rnd == common | {"Knoten", "Mittlerer Grad", "Potenzialspanne"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="ev")
    at.session_state["hill_slider"] = 12
    at.run()
    at.session_state["net_select"] = "small"
    at.run()
    at.session_state["net_select"] = "ev"
    at.run()
    assert not at.exception and at.slider(key="hill_slider").value == 12


def test_step_slider_returns_to_the_last_step_when_anything_changes():
    at = _run(net="ev")
    at.slider(key="jo_step").set_value(2)
    at.run()
    assert at.slider(key="jo_step").value == 2
    at.session_state["method_select"] = "constant"
    at.run()
    assert not at.exception and at.slider(key="jo_step").value == at.slider(key="jo_step").max


def test_every_step_of_the_small_nets_renders_for_both_methods():
    for net in ("small", "small_cycle"):
        for method in C.METHODS:
            def setup(at):
                at.session_state["net_select"] = net
                at.session_state["method_select"] = method
            at = _run(setup)
            for k in range(0, int(at.slider(key="jo_step").max) + 1):
                at.slider(key="jo_step").set_value(k)
                at.run()
                assert not at.exception, (net, method, k)


def test_play_renders_several_frames_without_duplicate_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    for setup in (lambda a: None, lambda a: a.session_state.__setitem__("net_select", "ev"), lambda a: a.session_state.__setitem__("net_select", "random"), lambda a: a.session_state.__setitem__("net_select", "small_cycle")):
        at = _run(setup)
        _play(at)
        assert not at.exception, [e.value for e in at.exception]
    at = _run(net="ev")
    at.session_state["side_slider"] = 12
    at.run()
    _play(at)
    assert not at.exception


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "random"
    at.query_params["nodes"] = "999999"
    at.query_params["method"] = "astar"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "random"
    assert at.slider(key="nodes_slider").value == C.NODES_MAX and at.selectbox(key="method_select").value == C.DEFAULT_METHOD


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 Netze" in c.value for c in at.caption)
    for key in ("effort_start", "few_start", "trap_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Der einmalige Bellman-Ford ab q macht davon nur", "Deshalb ist Johnson schon bei k = 1 gleichauf", "Anheben bestraft **jede Kante**"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # Das Netz steht in der Play-Schleife (kleine und große Netze an je einer Stelle): sein Schlüssel trägt den Schritt
    assert sorted(keys) == sorted(["net_chart", "net_chart", "effort_degree_chart", "effort_size_chart", "few_chart", "trap_pot_chart", "trap_hops_chart"]), keys
    assert all('_{current}"' in c for c in calls if 'key=f"net_chart' in c) and sum('key=f"' in c for c in calls) == 2
    viz = (ROOT / "jo_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
