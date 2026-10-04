"""Johnson - Umgewichten statt neu erfinden - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Johnson - und lässt stattdessen das Beispiel wachsen.
Siebtes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Konvergenz von Bellman-Ford und Floyd-Warshall: einmal Bellman-Ford für Potenziale, dann n-mal Dijkstra auf umgewichteten Kanten.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import jo_constants as C
import jo_evaluation as ev
from jo_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from jo_scenario import make_network
from jo_visualization import (
    build_effort_degree,
    build_effort_size,
    build_few_sources,
    build_network,
    build_trap_hops,
    build_trap_pot,
    frame_title,
    table_rows,
)

st.set_page_config(page_title="Johnson – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


def _cost(net, x):
    return f"{x:,.0f} {net.unit}".replace(",", ".") if float(x).is_integer() else f"{x:g} {net.unit}"


def _pct(x, digits=0):
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _analysis(params, method, distance):
    return ev.analyse(_network(params), method, distance, params[-1])


@st.cache_data(show_spinner=False)
def _effort_degree():
    return ev.effort_vs_degree()


@st.cache_data(show_spinner=False)
def _effort_size():
    return ev.effort_vs_size()


@st.cache_data(show_spinner=False)
def _complete():
    return ev.complete_graph_effort()


@st.cache_data(show_spinner=False)
def _few():
    return ev.few_sources()


@st.cache_data(show_spinner=False)
def _trap_pot():
    return ev.constant_trap()


@st.cache_data(show_spinner=False)
def _trap_hops():
    return ev.constant_trap_by_hops()


st.title("⚖️ Johnson – Umgewichten statt neu erfinden")
st.markdown(
    """
Dijkstra ist schnell, darf aber keine negativen Kanten sehen; Bellman-Ford verträgt sie, braucht aber für **alle** Starts n-mal so viel Arbeit; Floyd-Warshall liefert alle Paare, braucht dafür $n^3$ Vergleiche.
**Johnson** verbindet die beiden ersten: **einmal Bellman-Ford** berechnet **Potenziale** $h$, mit denen jede Kante umgewichtet wird, $c'(u,v) = c(u,v) + h(u) - h(v) \\ge 0$. Danach darf **n-mal Dijkstra** laufen, und das Ergebnis wird zurückgerechnet: $d(s,t) = d'(s,t) - h(s) + h(t)$.
Der Trick: jede Route von $s$ nach $t$ ändert ihre Kosten um **dieselbe Konstante** $h(s) - h(t)$ - die Potenziale der Zwischenknoten heben sich auf. Die kürzesten Routen bleiben also dieselben.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Konvergenz aus Bellman-Ford und Floyd-Warshall - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwächen von Johnson sind der eine Bellman-Ford (er scheitert an negativen Zyklen) und dichte Netze, in denen sich der Vorteil gegenüber Floyd-Warshall verliert. Das Verfahren geht auf Johnson (1977) zurück; "
    "alle Netze und Zahlen dieser Demo sind eigene Graphen und Messungen."
)

with st.expander("So funktioniert Johnson", expanded=True):
    st.markdown(
        """
1. **Hilfsknoten:** ein neuer Knoten $q$ mit einer Kante der Kosten 0 zu jedem Knoten. Von $q$ aus ist alles erreichbar, und $q$ liegt auf keinem Zyklus.
2. **Potenziale:** Bellman-Ford ab $q$ liefert $h(v) = d(q,v) \\le 0$. Findet Bellman-Ford einen negativen Zyklus, gibt es keine kürzesten Routen, und Johnson bricht hier ab.
3. **Umgewichten:** $c'(u,v) = c(u,v) + h(u) - h(v)$. Wegen $h(v) \\le h(u) + c(u,v)$ (Dreiecksungleichung der Entfernungen ab $q$) ist $c' \\ge 0$. Kanten mit $c' = 0$ heißen **eng**: sie liegen auf kürzesten Routen von $q$ aus.
4. **n-mal Dijkstra** auf $c'$ (alle Kosten nichtnegativ).
5. **Zurückrechnen:** $d(s,t) = d'(s,t) - h(s) + h(t)$. Die Routen sind dieselben Kanten im Originalnetz.
6. **Die Falle:** "einfach alle Kosten um eine Konstante anheben" ändert die Kosten einer Route um (Zahl der Kanten) × Konstante - Routen mit vielen Kanten werden bestraft, Dijkstra findet nicht mehr die billigste (Schalter in der Seitenleiste).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (Liefernetz mit einer negativen Kante, dasselbe mit negativem Zyklus) oder erzeugt (E-Lieferwagen mit Rekuperation, Zufallsnetz mit Potenzialen, Stadtnetz ohne negative Kanten). Alle Kosten sind ganze Zahlen; höchstens 400 Knoten.",
    )
    if net_key in C.SIZED_NETS:
        seed_widget("side_slider")
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider", help="Größe des Rasters: n = Seite² Knoten.")
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "ev":
        seed_widget("hill_slider")
        hill = st.slider("Hügel [m Höhenunterschied]", *bounds("hill_slider"), key="hill_slider", help="Höhenunterschied zwischen Tal und Spitze: je höher, desto mehr negative Kanten (bei 0 m gibt es keine).")
        st.session_state[KEPT["hill_slider"]] = hill
        seed_widget("eta_slider")
        eta = st.slider("Rückgewinnung bergab [%]", *bounds("eta_slider"), key="eta_slider", help="Wie viel Lageenergie der Wagen beim Bergabfahren zurückgewinnt. Unter 100 % entsteht nie ein negativer Zyklus.")
        st.session_state[KEPT["eta_slider"]] = eta
    else:
        hill = int(st.session_state.get(KEPT["hill_slider"], C.DEFAULT_HILL))
        eta = int(st.session_state.get(KEPT["eta_slider"], C.DEFAULT_ETA))
    if net_key == "city":
        seed_widget("reach_slider")
        reach = st.slider("Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1, help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster).")
        st.session_state[KEPT["reach_slider"]] = reach
        seed_widget("spread_slider")
        spread = st.slider("Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25, help="Kosten einer Straße = Länge × (1 + Streuung × Zufall), gerundet auf ganze Meter.")
        st.session_state[KEPT["spread_slider"]] = spread
    else:
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
    if net_key == "random":
        seed_widget("nodes_slider")
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10, help="Anzahl der Knoten n.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        seed_widget("degree_slider")
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5, help="Kanten je Knoten (ausgehend). Je dichter das Netz, desto mehr Arbeit haben die n Läufe von Dijkstra - und desto näher rückt Floyd-Warshall (im vollständigen Netz gleichauf).")
        st.session_state[KEPT["degree_slider"]] = degree
        seed_widget("pot_slider")
        pot = st.slider("Potenzialspanne", *bounds("pot_slider"), key="pot_slider",
                        help="Kosten = 1 bis 9 plus Potenzial(Start) − Potenzial(Ziel), Potenziale zufällig zwischen 0 und dieser Spanne. Je größer, desto mehr negative Kanten - nie ein negativer Zyklus. 0 = keine negativen Kanten.")
        st.session_state[KEPT["pot_slider"]] = pot
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
        pot = int(st.session_state.get(KEPT["pot_slider"], C.DEFAULT_POT))
    method = st.selectbox("Umgewichtung", C.METHODS, key="method_select", format_func=lambda k: C.METHOD_LABELS[k],
                          help="Johnson: Potenziale aus Bellman-Ford ab q, das Ergebnis ist exakt. Falle: statt der Potenziale alle Kosten um |kleinster Wert| anheben und n-mal Dijkstra rechnen - die gefundenen Routen sind oft nicht die billigsten. "
                               "Ohne negative Kanten sind beide dasselbe.")
    if net_key not in C.SMALL_NETS:
        seed_widget("distance_slider")
        distance = st.slider("Entfernung Start–Ziel [%]", *bounds("distance_slider"), key="distance_slider",
                             help="Welches Paar die Ansicht zeigt: das Ziel ist der Knoten, dessen Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei diesem Prozentwert liegt (100 = der am weitesten entfernte). Die Rechnung enthält alle Paare.")
        st.session_state[KEPT["distance_slider"]] = distance
    else:
        distance = int(st.session_state.get(KEPT["distance_slider"], C.DEFAULT_DISTANCE))
    if net_key in ("city", "ev", "random"):
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "hill_slider": int(hill), "eta_slider": int(eta), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2),
                   "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "pot_slider": int(pot), "distance_slider": int(distance), "method_select": method, "seed_input": int(seed)})

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, hill=C.DEFAULT_HILL, eta=C.DEFAULT_ETA, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, pot=C.DEFAULT_POT, seed=C.DEFAULT_SEED)
if net_key == "ev":
    d.update(side=int(side), hill=int(hill), eta=int(eta), seed=int(seed))
elif net_key == "city":
    d.update(side=int(side), reach=round(float(reach), 1), spread=round(float(spread), 2), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), pot=int(pot), seed=int(seed))
params = (net_key, d["side"], d["hill"], d["eta"], d["reach"], d["spread"], d["nodes"], d["degree"], d["pot"], d["seed"])
distance_used = C.DEFAULT_DISTANCE if net_key in C.SMALL_NETS else int(distance)
with st.spinner("Rechne ..."):
    a = _analysis(params, method, distance_used)
net, m, jo, g = a.net, a.metrics, a.jo, a.net.graph
small = bool(g.names)
n = g.n
frames = ev.frames(a)
last_step = len(frames) - 1
view_key = (params, method, distance_used)

# --- Johnson in Aktion -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Johnson in Aktion")
if st.session_state.get("jo_step_owner") != view_key:
    st.session_state["jo_step"] = last_step
    st.session_state["jo_step_owner"] = view_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Bild der Rechnung", 0, last_step, key="jo_step",
                     help="Die Rechnung in Bildern: Ausgangsnetz, Bellman-Ford ab q (je Runde), umgewichtetes Netz, Dijkstra ab dem Start (bei großen Netzen etwa 40 Zwischenstände), Ergebnis. Ganz rechts ist die Rechnung fertig.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(current):
    frame = frames[current]
    with view_slot.container():
        st.markdown(f"**{frame_title(a, frame)}**")
        if small:
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_network(a, frame, height=400), width="stretch", key=f"net_chart_{current}")
            c2.markdown(f"**Potenziale und Entfernungen ab {g.names[a.s]}**")
            c2.dataframe(pd.DataFrame(table_rows(a, frame)), hide_index=True, width="stretch")
        else:
            st.plotly_chart(build_network(a, frame), width="stretch", key=f"net_chart_{current}")
        kind = frame[0]
        if kind == "result" and not jo.negative_cycle:
            if a.method == "johnson" and m.get("cost_exact") is not None:
                if a.s == a.t:
                    st.markdown("Start und Ziel sind derselbe Knoten.")
                else:
                    who = (g.names[a.s], g.names[a.t]) if small else (a.s, a.t)
                    hs, ht = jo.h[a.s], jo.h[a.t]
                    dprime = jo.dist_rw[jo.row(a.s), a.t]
                    st.markdown(f"**Zurückgerechnet {who[0]} → {who[1]}:** d = d' − h(Start) + h(Ziel) = {_cost(net, dprime)} − ({_fmt_h(hs)}) + ({_fmt_h(ht)}) = **{_cost(net, m['cost_exact'])}**, "
                                f"{len(jo.route(a.s, a.t)) - 1} Kanten.")
            elif a.method == "constant" and m.get("cost_exact") is not None:
                who = (g.names[a.s], g.names[a.t]) if small else (a.s, a.t)
                st.markdown(f"**Falle {who[0]} → {who[1]}:** die gefundene Route kostet {_cost(net, m['cost_found'])} ({len(jo.route(a.s, a.t)) - 1} Kanten), die billigste {_cost(net, m['cost_exact'])} ({len(m['route_exact']) - 1} Kanten).")


def _fmt_h(x):
    return f"{x:g}"


if auto_play:
    n_frames = min(last_step + 1, 40)
    for k in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames)}):
        _render(k)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)

st.markdown("---")

# --- Umgewichten - und was es spart ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Umgewichten – und was es spart")
st.caption("**Zähler** = Schritte, die ein Verfahren ausführt: Johnson und n-mal Bellman-Ford / Dijkstra zählen Kantenprüfungen, Floyd-Warshall zählt Vergleiche in der Matrix. Die Zähler sind plattformfest, aber kein gemeinsames Maß "
           "(ein Vergleich in einer Matrixoperation ist billiger als eine Kantenprüfung mit Warteschlange). Laufzeiten stehen nur als Messwerte im Vergleich unten.")
code = ev.verdict(a)
p1, p2, p3, p4 = st.columns(4)
if m["cycle"]:
    p1.metric("Johnson", "bricht ab", delta=f"Zyklus mit {m['cycle_cost']:g} {net.unit}", delta_color="off", help="Bellman-Ford ab q meldet einen negativen Zyklus: es gibt keine Potenziale.")
else:
    p1.metric("Johnson", _num(m["total"]), delta=f"Bellman-Ford ab q {_num(m['bf_checks'])} + n-mal Dijkstra {_num(m['dijkstra_checks'])}", delta_color="off",
              help="Kantenprüfungen insgesamt: einmal Bellman-Ford ab dem Hilfsknoten q, dann ein Dijkstra-Lauf je Start.")
p2.metric("Floyd-Warshall", _num(m["fw_comparisons"]), delta=(f"{m['fw_comparisons'] / max(m['total'], 1):.1f}× so viele wie Johnson" if m["fw_comparisons"] > m["total"] else "nicht mehr als Johnson"), delta_color="off",
          help="n³ Vergleiche, egal wie dünn das Netz ist.")
p3.metric("n-mal Bellman-Ford", _num(m["nbf_checks"]), delta=(f"{m['nbf_checks'] / max(m['total'], 1):.1f}× so viele wie Johnson" if m["nbf_checks"] > m["total"] else "nicht mehr als Johnson"), delta_color="off",
          help="Kantenprüfungen aller n Läufe mit frühem Abbruch.")
if m["cycle"]:
    p4.metric("Negative Kanten", f"{m['neg_edges']} von {m['m']}", help="Vor dem Umgewichten; Johnson kommt nicht bis dorthin.")
else:
    p4.metric("Negative Kanten", f"{m['neg_edges']} → {m['neg_after']}", delta=f"{m['tight_edges']} enge Kanten ({m['tight_share']:.0%})", delta_color="off",
              help="Zahl der Kanten mit negativen Kosten vor und nach dem Umgewichten; \"eng\" sind Kanten mit c' = 0.")
if code == "cycle":
    st.warning(f"🔁 **Negativer Zyklus:** Bellman-Ford ab q meldet {m['cycle_nodes']} Knoten auf einem Zyklus der Kosten {m['cycle_cost']:g} {net.unit} nach {m['bf_rounds']} Runden ({_num(m['bf_checks'])} Kantenprüfungen). "
               "Ohne Potenziale kann Johnson nicht umgewichten; für Paare, deren Routen den Zyklus berühren können, gibt es keine kürzeste Route." + (" Die Falle merkt davon nichts: sie liefert trotzdem Routen und Zahlen." if a.method == "constant" else ""))
elif code == "johnson_ok":
    st.success(f"✅ Johnson liefert dieselbe Matrix wie Floyd-Warshall ({'bestätigt' if m['exact_vs_fw'] else 'ABWEICHUNG'}) mit {_num(m['total'])} statt {_num(m['fw_comparisons'])} Schritten. Der einmalige Bellman-Ford macht davon nur {m['bf_checks'] / m['total']:.1%} aus. "
               f"n-mal Dijkstra ohne Umgewichten läge an {m['naive_wrong']} von {m['pairs']} Paaren ({m['naive_wrong'] / m['pairs']:.0%}) falsch.")
elif code == "no_negative":
    st.info(f"ℹ️ Keine negativen Kanten: alle Potenziale sind 0, die umgewichteten Kosten sind die ursprünglichen. Die Sicherheit kostet {_num(m['bf_checks'])} Kantenprüfungen für Bellman-Ford ab q ({m['bf_checks'] / max(m['dijkstra_checks'], 1):.1%} des n-mal Dijkstra: {_num(m['dijkstra_checks'])}) - "
            "hier würde n-mal Dijkstra allein genügen.")
elif code == "trap_wrong":
    st.warning(f"⚠️ **Die Falle:** alle Kosten um {m['shift']:g} {net.unit} angehoben - bei {m['const_wrong']} von {m['pairs']} Paaren ({m['const_share_wrong']:.1%}) ist die gefundene Route nicht die billigste, im Mittel {m['const_excess']:.1f} {net.unit} zu teuer. "
               "Johnson wäre bei allen exakt.")
else:
    st.info("ℹ️ Die Falle liegt in diesem Netz zufällig nirgends daneben - keine Garantie (es gibt hier keine negativen Kanten oder keine Route, die durch das Anheben benachteiligt würde).")
if not m["cycle"] and a.s != a.t and m.get("cost_exact") is not None:
    who = (g.names[a.s], g.names[a.t]) if small else (a.s, a.t)
    pair = f"**Gezeigtes Paar** ({who[0]} → {who[1]}): billigste Route {_cost(net, m['cost_exact'])} mit {len(m['route_exact']) - 1} Kanten"
    if a.method == "constant":
        pair += f"; die Falle findet {_cost(net, m['cost_found'])} mit {len(jo.route(a.s, a.t)) - 1} Kanten"
    st.markdown(pair + f". Potenziale: {m['h_nonzero']} von {n} Knoten haben ein Potenzial ungleich 0, das kleinste ist {m['h_min']:g} {net.unit}.")

st.markdown("---")

# --- Vergleich -------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Johnson, Floyd-Warshall und n-mal Bellman-Ford im Vergleich"):
    rows = {"Verfahren": ["Johnson (ganze Rechnung)", "Floyd-Warshall (Matrixoperation)", "n-mal Bellman-Ford"], "Zähler": [_num(m["total"]), _num(m["fw_comparisons"]), _num(m["nbf_checks"])]}
    rows["Laufzeit [ms]"] = [f"{a.seconds['jo'] * 1000:.1f}", f"{m['fw_seconds'] * 1000:.1f}" if "fw_seconds" in m else "–", f"{a.seconds['bf'] * 1000:.1f}"]
    st.table(rows)
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (ein Lauf, schwankend). Johnson und n-mal Bellman-Ford sind reines Python, Floyd-Warshall läuft als vektorisierte Matrixoperation (in C, die Zeit enthält den Vergleich mit n-mal Dijkstra): "
               "Laufzeiten dieser Umsetzungen sind nicht als Vergleich der Verfahren zu lesen. Die Zähler oben sind es.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wann lohnt sich Johnson?")
if st.button("Aufwand gegen Dichte und Größe messen (dauert einen Moment)", key="effort_start"):
    st.session_state["effort_on"] = True
if st.session_state.get("effort_on"):
    with st.spinner("Rechne Zähler für 4 Dichten und 4 Größen × 5 Netze ..."):
        drows, srows, comp = _effort_degree(), _effort_size(), _complete()
    c1, c2 = st.columns(2)
    c1.markdown("**Gegen die Dichte** (200 Knoten)")
    c1.plotly_chart(build_effort_degree(drows), width="stretch", key="effort_degree_chart")
    c2.markdown("**Gegen die Größe** (mittlerer Grad 3)")
    c2.plotly_chart(build_effort_size(srows), width="stretch", key="effort_size_chart")
    b4, d2 = srows[-1], drows[0]
    st.caption(f"Zufallsnetze mit Potenzialen (Spanne 8, also mit negativen Kanten), Mittel über 5 Netze. Bei 400 Knoten braucht Johnson {_num(b4['johnson'])} Schritte, Floyd-Warshall {_num(b4['fw'])} ({b4['fw'] / b4['johnson']:.0f}-fach) und n-mal Bellman-Ford {_num(b4['nbf'])} ({b4['nbf'] / b4['johnson']:.0f}-fach). "
               f"Der einmalige Bellman-Ford ab q macht davon nur {b4['bf_part'] / b4['johnson']:.1%} aus. Je dichter das Netz, desto mehr Arbeit haben die n Läufe von Dijkstra: bei Grad 2 sind es {_num(d2['johnson'])} Schritte, Floyd-Warshall bleibt bei {_num(d2['fw'])}. "
               f"Im **vollständigen Netz** (60 Knoten) kehrt es sich um: Johnson {_num(comp['johnson'])} gegen Floyd-Warshall {_num(comp['fw'])} Schritte - dort lohnt sich Johnson nicht mehr.")

st.markdown("---")

st.subheader("🔬 Wenige Startknoten")
if st.button("Johnson gegen k-mal Bellman-Ford messen (dauert einen Moment)", key="few_start"):
    st.session_state["few_on"] = True
if st.session_state.get("few_on"):
    with st.spinner("Rechne 5 Werte von k × 5 Netze ..."):
        frows = _few()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_few_sources(frows), width="stretch", key="few_chart")
    c2.table({"k": [str(r["k"]) for r in frows], "Johnson": [_num(r["johnson"]) for r in frows], "k-mal Bellman-Ford": [_num(r["kbf"]) for r in frows]})
    f1, f2, f100 = frows[0], frows[1], frows[-1]
    st.caption(f"Zufallsnetze mit 200 Knoten, Grad 3, Spanne 8, Mittel über 5 Netze; die k Starts sind die ersten k Knoten. Der einmalige Bellman-Ford ab q kostet {_num(f1['bf_part'])} Kantenprüfungen - etwa so viel wie ein einzelner Bellman-Ford ab einem Start ({_num(f1['kbf'])}). "
               f"Deshalb ist Johnson schon bei k = 1 gleichauf ({_num(f1['johnson'])} gegen {_num(f1['kbf'])}), bei k = 2 klar billiger ({_num(f2['johnson'])} gegen {_num(f2['kbf'])}) und bei k = 100 {f100['kbf'] / f100['johnson']:.1f}-mal billiger. Jeder weitere Start kostet nur einen Dijkstra-Lauf. "
               "Floyd-Warshall bleibt bei jedem k bei n³ Vergleichen.")

st.markdown("---")

st.subheader("🔬 Die Falle: alle Kosten anheben")
if st.button("Wie oft liegt die Falle daneben? (dauert einen Moment)", key="trap_start"):
    st.session_state["trap_on"] = True
if st.session_state.get("trap_on"):
    with st.spinner("Rechne 5 Potenzialspannen und die Kantenzahlen × 5 Netze ..."):
        prow, hrow = _trap_pot(), _trap_hops()
    c1, c2 = st.columns(2)
    c1.markdown("**Nach Zahl der negativen Kanten**")
    c1.plotly_chart(build_trap_pot(prow), width="stretch", key="trap_pot_chart")
    c2.markdown("**Nach Kantenzahl der richtigen Route**")
    c2.plotly_chart(build_trap_hops(hrow), width="stretch", key="trap_hops_chart")
    hi = hrow[-1]
    st.caption(f"Zufallsnetze mit 100 Knoten, Grad 3, Mittel über 5 Netze. Bei Spanne {prow[0]['pot']} ({prow[0]['neg_share']:.1%} negative Kanten) liegt die Falle an {prow[0]['share_wrong']:.1%} der Paare daneben, bei Spanne {prow[-1]['pot']} ({prow[-1]['neg_share']:.0%} negative Kanten) an {prow[-1]['share_wrong']:.0%}. "
               f"Rechts der Grund: Anheben bestraft **jede Kante**. Hat die richtige Route nur eine Kante, liegt die Falle nie daneben ({hrow[0]['share_wrong']:.0%}); bei {hrow[3]['hops']} Kanten an {hrow[3]['share_wrong']:.0%} der Paare, ab {hi['hops']} Kanten an {hi['share_wrong']:.0%}. "
               "Die Potenziale dagegen ändern jede Route um dieselbe Konstante - deshalb sind sie exakt.")

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt keinen negativen Zyklus** | Bellman-Ford ab q meldet ihn, aber Potenziale gibt es dann nicht: Johnson bricht ab. Für Paare, die den Zyklus berühren, gibt es keine kürzeste Route. | (das Problem hat keine Lösung) |
| **Das Netz ist dünn** | Im vollständigen Netz mit 60 Knoten braucht Johnson 226 800 Schritte, Floyd-Warshall 216 000: der Vorteil von n·m gegen n³ verschwindet, wenn m so groß ist wie n². | **Floyd-Warshall** |
| **Es werden viele Starts gebraucht** | Für wenige Starts genügen wenige Dijkstra-Läufe; bei nur einem Start kostet ein Bellman-Ford ab dem Start etwa so viel wie der ab q (3 840 gegen 3 360 Kantenprüfungen bei 200 Knoten). | k-mal Bellman-Ford, Bellman-Ford |
| **Die Kosten ändern sich nicht** | Ändern sich Kosten, sind die Potenziale veraltet (eine Kante kann wieder negativ werden): der Bellman-Ford ab q muss neu laufen. | Verfahren mit dynamischen Potenzialen |
| **Nur eine Kostenart** | Zeit gegen Energie gleichzeitig ist ein anderes Problem. | **Mehrkriterien-Routing** |
"""
)
st.caption("Verwandtes aus der Literatur (hier nicht gebaut): dieselben \"reduzierten Kosten\" $c + h(u) - h(v) \\ge 0$ stehen hinter A* mit konsistenter Heuristik und hinter der Ungarischen Methode. "
           "Die Nachbarn der Kürzeste-Wege-Linie: Mehrkriterien-Routing (gebaut). Ebenfalls gebaut: Breitensuche, Dijkstra, bidirektionale Suche, Contraction Hierarchies, Bellman-Ford und Floyd-Warshall.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \in \mathbb{R}$, $n=|V|$, $m=|E|$. Gesucht: die Entfernungen $\delta(s,t)$ für alle $s$ (oder eine Startmenge) - oder die Feststellung, dass ein negativer Zyklus erreichbar ist.

**Potenziale.** Sei $G^{q}$ der Graph mit einem Hilfsknoten $q$ und Kanten $(q,v)$ der Kosten 0. Bellman-Ford ab $q$ liefert $h(v)=\delta_{G^{q}}(q,v)\le 0$ - oder einen negativen Zyklus (er liegt in $G$, denn in $q$ endet keine Kante).
Aus $h(v)\le h(u)+c_{uv}$ folgt $c'_{uv}:=c_{uv}+h(u)-h(v)\ge 0$: **jedes zulässige Potenzial macht alle Kosten nichtnegativ**.

**Umgewichten.** Für jede Route $P=(v_0,\dots,v_k)$ gilt $c'(P)=\sum_i \bigl(c_{v_iv_{i+1}}+h(v_i)-h(v_{i+1})\bigr)=c(P)+h(v_0)-h(v_k)$. Die Summe der Potenziale ist eine **Teleskopsumme**. Deshalb ist $P$ genau dann eine kürzeste Route für $c$, wenn sie eine für $c'$ ist,
und $\delta(s,t)=\delta'(s,t)-h(s)+h(t)$. Kreise behalten ihre Kosten (Start = Ziel): es entsteht kein neuer negativer Zyklus.

**Nicht eindeutig.** Jedes $h$ mit $h(v)\le h(u)+c_{uv}$ genügt - die Potenziale aus Bellman-Ford ($h=\delta(q,\cdot)$) sind die **größten** zulässigen Potenziale mit $h\le 0$. Die Kosten vor der Verschiebung, mit der das Zufallsnetz erzeugt wurde, sind ein anderes zulässiges Potenzial.

**Warum "alle Kosten anheben" falsch ist.** Mit $c''_e=c_e+M$ kostet eine Route mit $k$ Kanten $c(P)+kM$: die Konstante hängt von der **Kantenzahl** ab, nicht vom Paar $(s,t)$. Eine Route mit weniger Kanten kann gewinnen, obwohl sie unter $c$ teurer ist.

**Aufwand.** Ein Bellman-Ford $O(n\,m)$ (in der Praxis meist deutlich weniger, siehe Bellman-Ford-Demo) plus $n$ Läufe Dijkstra: $O\bigl(n\,m+n(m+n\log n)\bigr)$ mit Fibonacci-Heap (Lehrbuchwert) - gegen $\Theta(n^3)$ bei Floyd-Warshall. Für $m=\Theta(n^2)$ sind beide von gleicher Größenordnung.

Implementiert in `jo_graph.py` (CSR-Graph), `jo_algorithm.py` (Hilfsknoten, Potenziale, Umgewichten, Dijkstra ab jedem Start, Zurückrechnen, die Falle), `jo_sp.py` (Dijkstra und Bellman-Ford aus der Floyd-Warshall-Demo), `jo_fw.py` (Floyd-Warshall als Vergleich und Exaktheitsprobe), `jo_scenario.py` (Netze), `jo_evaluation.py` (Kennzahlen, Bildfolge, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kürzeste Wege: von der Breitensuche bis RAPTOR](https://sebastianhanisch.net/konzepte-kuerzeste-wege.html)."
)
