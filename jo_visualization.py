"""Plotly-Abbildungen: die Bildfolge der Ansicht (Ausgangsnetz, Bellman-Ford ab q, umgewichtetes Netz, Dijkstra, Ergebnis) und die Experimente. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import jo_constants as C


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _fmt(x):
    return "∞" if not np.isfinite(x) else f"{x:g}"


def _segments(g, mask=None):
    """Kanten als eine Linienspur (None trennt die Segmente); bei ungerichteten Netzen Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    keep = np.ones(len(src), dtype=bool) if mask is None else mask.copy()
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _shifted(g, u, v, amount=0.11):
    """Endpunkte der Kante u -> v, ein Stück nach rechts der Fahrtrichtung versetzt, damit Hin- und Rückrichtung getrennt zu sehen sind."""
    p, q = g.xy[u], g.xy[v]
    d = q - p
    norm = float(np.hypot(*d)) or 1.0
    off = np.array([d[1], -d[0]]) / norm * amount
    return p + off, q + off


def bf_state(a, k):
    """Bellman-Ford ab q nach Runde k (1-basiert): Entfernungen der Originalknoten (= vorläufige Potenziale) und die in dieser Runde verbesserten Kanten."""
    n = a.net.graph.n
    dist, _, changed = a.jo.bf.history[k - 1]
    return np.array(dist[:n]), changed


def frame_title(a, frame):
    """Überschrift der Ansicht für ein Bild der Bildfolge."""
    kind, k = frame
    m = a.metrics
    if kind == "orig":
        return f"Ausgangsnetz: {m['neg_edges']} von {m['m']} Kanten haben negative Kosten - Dijkstra darf hier nicht laufen."
    if kind == "bf":
        return f"Bellman-Ford ab dem Hilfsknoten q, Runde {k}: Kanten der Kosten 0 von q zu allen Knoten, die Entfernung von q ist das vorläufige Potenzial h."
    if kind == "reweighted":
        return f"Umgewichtet: c' = c + h(Start) − h(Ziel). Alle Kosten sind nichtnegativ; {a.jo.tight_edges} Kanten sind \"eng\" (c' = 0, gelb)."
    if kind == "shift":
        return f"Falle: alle Kosten um {a.jo.shift:g} angehoben, damit nichts mehr negativ ist. Das ändert die Kosten jeder Route um (Kantenzahl × {a.jo.shift:g}) - nicht um eine Konstante."
    if kind == "dijkstra":
        return f"Dijkstra ab dem Start auf den umgewichteten Kosten: {k} Knoten festgelegt."
    if kind == "cycle":
        return "Bellman-Ford meldet einen negativen Zyklus: es gibt keine Potenziale, Johnson bricht ab."
    return "Ergebnis: zurückgerechnet d = d' − h(Start) + h(Ziel)." if a.method == "johnson" else "Ergebnis der Falle: die gefundene Route (blau gestrichelt) gegen die billigste (rot)."


def build_network(a, frame, height=520):
    """Das Netz im gewählten Bild. Kleine Netze mit Kosten an den Kanten (vorher, im umgewichteten Bild nachher mit den ursprünglichen in Klammern); große Netze farbig."""
    net, g, jo = a.net, a.net.graph, a.jo
    kind, k = frame
    small = bool(g.names)
    rw = jo.reweighted
    show_rw = kind in ("reweighted", "shift", "dijkstra", "result") and rw is not None
    fig = go.Figure()
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip", showlegend=False))
    src = np.repeat(np.arange(g.n), g.degree())
    cost_now = rw.weight if show_rw else g.weight
    if small:
        both = {(int(u), int(v)) for u, v in zip(src, g.indices)}
        for i, (u, v) in enumerate(zip(src.tolist(), g.indices.tolist())):
            p, q = _shifted(g, u, v) if (v, u) in both else (g.xy[u], g.xy[v])
            neg_now = cost_now[i] < 0
            tight = show_rw and kind == "reweighted" and cost_now[i] == 0 and jo.method == "johnson"
            col = "rgba(44,160,44,0.95)" if neg_now else ("rgba(230,170,0,0.95)" if tight else "rgba(120,120,120,0.55)")
            fig.add_annotation(x=q[0], y=q[1], ax=p[0], ay=p[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2.4 if (neg_now or tight) else 1.3,
                               arrowcolor=col, standoff=13, startstandoff=13)
            mid = (p + q) / 2
            text = _fmt(cost_now[i]) if not (show_rw and cost_now[i] != g.weight[i]) else f"{_fmt(cost_now[i])} ({_fmt(g.weight[i])})"
            fig.add_annotation(x=mid[0], y=mid[1], text=text, showarrow=False, font=dict(size=12, color="#1b7f1b" if neg_now else "#555"), bgcolor="rgba(255,255,255,0.85)")
    elif net.geometric:
        neg = cost_now < 0
        if neg.any():
            nx_, ny_ = _segments(g, neg)
            fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="lines", line=dict(color="rgba(44,160,44,0.9)", width=2), name="negative Kanten", hoverinfo="skip"))
        if kind == "reweighted" and jo.method == "johnson":
            tight = cost_now == 0
            if tight.any():
                tx, ty = _segments(g, tight)
                fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color=C.COLORS["tight"], width=3), name="enge Kanten (c' = 0)", hoverinfo="skip"))
    # Knotenfarben und Beschriftung je Bild
    n = g.n
    if kind == "bf":
        h, _ = bf_state(a, k)
        value, title = h, "h"
    elif kind == "dijkstra":
        row = jo.row(a.s)
        settled = np.zeros(n, dtype=bool)
        settled[jo.order[row][:k]] = True
        value = np.where(settled, jo.dist_rw[row], np.nan)
        title = "d'"
    elif kind in ("reweighted", "shift"):
        value, title = (jo.h if jo.h is not None else np.zeros(n)), "h"
    elif kind == "result":
        row = jo.row(a.s)
        value, title = jo.dist[row], "d"
    else:
        value, title = None, ""
    finite = np.isfinite(value) if value is not None else np.zeros(n, dtype=bool)
    size = 15 if small else (3 if n > 1500 else (5 if n > 300 else 8))
    if small:
        label = [g.names[v] + (f"<br>{title} = {_fmt(value[v])}" if value is not None and finite[v] else "") for v in range(n)]
        rest, got = np.where(~finite)[0], np.where(finite)[0]
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers+text", showlegend=False, text=[label[i] for i in rest], textposition="top center", hoverinfo="skip",
                                     marker=dict(size=size, color="white", line=dict(color="gray", width=1.5))))
        if len(got):
            vals = value[got]
            fig.add_trace(go.Scatter(x=g.xy[got, 0], y=g.xy[got, 1], mode="markers+text", showlegend=False, text=[label[i] for i in got], textposition="top center", hoverinfo="skip",
                                     marker=dict(size=size, color=vals, colorscale="Viridis", showscale=False, cmin=float(vals.min()), cmax=float(vals.max()) if vals.max() > vals.min() else float(vals.min()) + 1,
                                                 line=dict(color="gray", width=1.5))))
    else:
        rest = np.where(~finite)[0]
        if len(rest):
            fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(size=size, color="rgba(200,200,200,0.9)")))
        got = np.where(finite)[0]
        if len(got):
            vmin, vmax = float(value[got].min()), float(value[got].max())
            fig.add_trace(go.Scatter(x=g.xy[got, 0], y=g.xy[got, 1], mode="markers", showlegend=False, customdata=value[got], hovertemplate=f"{title} = %{{customdata:g}} {net.unit}<extra></extra>",
                                     marker=dict(size=size, color=value[got], colorscale="Viridis", cmin=vmin, cmax=max(vmax, vmin + 1), colorbar=dict(title=f"{title}<br>[{net.unit}]", thickness=12, len=0.6))))
    # Hilfsknoten q (nur im Bellman-Ford-Bild): am Rand des Netzes, mit Strichen zu allen Knoten wäre unlesbar - nur als Marke
    if kind == "bf":
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        qx, qy = lo[0] - 0.12 * (hi[0] - lo[0]), (lo[1] + hi[1]) / 2
        fig.add_trace(go.Scatter(x=[qx], y=[qy], mode="markers+text", text=["q"], textposition="middle left", name="Hilfsknoten q (Kante 0 zu jedem Knoten)", hoverinfo="skip",
                                 marker=dict(size=16, color=C.COLORS["q"], symbol="hexagon", line=dict(color="white", width=1.5))))
    # Routen im Ergebnisbild
    if kind == "result" and not jo.negative_cycle:
        if jo.method == "johnson":
            route = jo.route(a.s, a.t)
            if route:
                pts = g.xy[route]
                fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["jo"], width=5), name="billigste Route", hoverinfo="skip"))
        else:
            right = alg_route(a)
            found = jo.route(a.s, a.t)
            for route, name, style in ((right, "billigste Route (Johnson)", dict(color=C.COLORS["jo"], width=5)), (found, "Route der Falle", dict(color=C.COLORS["wrong"], width=4, dash="dash"))):
                if route:
                    pts = g.xy[route]
                    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=style, name=name, hoverinfo="skip"))
    if kind == "cycle" and jo.cycle:
        pts = g.xy[jo.cycle]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["cycle"], width=6), name=f"negativer Zyklus ({_fmt(jo.cycle_cost)} {net.unit})", hoverinfo="skip"))
    for node, name, color, symbol in ((a.s, "Start", C.COLORS["start"], "diamond"), (a.t, "Ziel", C.COLORS["goal"], "star")):
        lab = g.names[node] if small else f"{node}"
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {lab}", hoverinfo="skip", marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if not small:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0] - (0.12 * (hi[0] - lo[0]) if kind == "bf" else 0), hi[0] + pad[0]])
        fig.update_yaxes(range=[lo[1] - pad[1], hi[1] + pad[1] + 0.05 * (hi[1] - lo[1])])
    return _base(fig, height)


def alg_route(a):
    """Die billigste Route des gezeigten Paares (aus der richtigen Rechnung), auch wenn die Ansicht die Falle zeigt."""
    return a.metrics.get("route_exact", [])


def table_rows(a, frame):
    """Kleine Netze: Ort, Potenzial h (im Bellman-Ford-Bild der vorläufige Wert) und Entfernung d' -> d ab dem Start, soweit sie im Bild schon feststehen."""
    g = a.net.graph
    jo = a.jo
    kind, k = frame
    rows = []
    for v in range(g.n):
        if kind == "bf":
            h = bf_state(a, k)[0][v]
            rows.append({"Ort": g.names[v], "h": _fmt(h), "d' → d": "–"})
        elif kind in ("orig", "cycle"):
            rows.append({"Ort": g.names[v], "h": "–", "d' → d": "–"})
        else:
            hv = jo.h[v] if jo.h is not None else None
            row = jo.row(a.s)
            settled = set(jo.order[row][:k]) if kind == "dijkstra" else (set(jo.order[row]) if kind == "result" else set())
            if v in settled and jo.h is not None:
                dprime, dtrue = jo.dist_rw[row, v], jo.dist[row, v]
                cell = f"{_fmt(dprime)} → {_fmt(dtrue)}" if kind in ("dijkstra", "result") else "–"
            elif v in settled:
                cell = f"{_fmt(jo.dist_rw[row, v])} → {_fmt(jo.dist[row, v])}"
            else:
                cell = "–"
            rows.append({"Ort": g.names[v], "h": _fmt(hv) if hv is not None else "–", "d' → d": cell})
    return rows


def build_effort_degree(rows, height=340):
    x = [r["degree"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("fw", "Floyd-Warshall (n³ Vergleiche)", "#d62728", "solid"), ("nbf", "n-mal Bellman-Ford (Kantenprüfungen)", "#2ca02c", "solid"), ("johnson", "Johnson (Bellman-Ford ab q + n-mal Dijkstra)", "#1f77b4", "solid"),
                                   ("dijkstra_part", "davon n-mal Dijkstra", "#9ecae1", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="mittlerer Ausgangsgrad (Kanten je Knoten)", type="log"), yaxis=dict(title="Zähler", type="log"))
    return _base(fig, height)


def build_effort_size(rows, height=340):
    x = [r["size"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("fw", "Floyd-Warshall (n³)", "#d62728", "solid"), ("nbf", "n-mal Bellman-Ford", "#2ca02c", "solid"), ("johnson", "Johnson", "#1f77b4", "solid"), ("bf_part", "davon Bellman-Ford ab q", "#9467bd", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Zähler", type="log"))
    return _base(fig, height)


def build_few_sources(rows, height=340):
    x = [r["k"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("kbf", "k-mal Bellman-Ford", "#2ca02c", "solid"), ("johnson", "Johnson (einmal Bellman-Ford ab q, dann k Dijkstra)", "#1f77b4", "solid"), ("bf_part", "davon der einmalige Bellman-Ford", "#9467bd", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="Zahl der Startknoten k", type="log"), yaxis=dict(title="Kantenprüfungen", type="log"))
    return _base(fig, height)


def build_trap_pot(rows, height=300):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["pot"]) for r in rows], y=[r["neg_share"] * 100 for r in rows], name="negative Kanten [%]", marker_color=C.COLORS["negative"]))
    fig.add_trace(go.Bar(x=[str(r["pot"]) for r in rows], y=[r["share_wrong"] * 100 for r in rows], name="Paare mit falscher Route bei der Falle [%]", marker_color=C.COLORS["wrong"]))
    fig.update_layout(barmode="group", xaxis_title="Potenzialspanne (je größer, desto mehr negative Kanten)", yaxis_title="Anteil [%]")
    return _base(fig, height)


def build_trap_hops(rows, height=300):
    fig = go.Figure(go.Bar(x=[("≥ " if r["hops"] == rows[-1]["hops"] else "") + str(r["hops"]) for r in rows], y=[r["share_wrong"] * 100 for r in rows], marker_color=C.COLORS["wrong"],
                           hovertemplate="%{x} Kanten: %{y:.1f} % falsch<extra></extra>"))
    fig.update_layout(xaxis_title="Kanten der richtigen Route", yaxis_title="Paare mit falscher Route bei der Falle [%]")
    return _base(fig, height)
