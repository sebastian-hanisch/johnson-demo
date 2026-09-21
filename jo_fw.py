"""Floyd-Warshall (Floyd 1962, Warshall 1962, Roy 1959): kürzeste Routen zwischen ALLEN Knotenpaaren mit drei Schleifen.

d_k(i, j) = billigste Route von i nach j, die nur über die ersten k Knoten (Nummern 0 .. k-1) als Zwischenknoten läuft:
    d_k(i, j) = min( d_{k-1}(i, j),  d_{k-1}(i, k-1) + d_{k-1}(k-1, j) )

Varianten (`variant`):
- "classic": dreifache Schleife in reinem Python, jeder Vergleich wird gezählt (immer genau n^3);
- "skip":    wie classic, aber nur Paare (i, j) mit endlichem d(i, k) und d(k, j) - zählt weniger, das Ergebnis ist dasselbe;
- "numpy":   je k eine Matrixoperation (Spalte plus Zeile), zählt dieselben n^3 Vergleiche wie classic und zusätzlich, wie viele "skip" gebraucht hätte.

Die Nachfolger-Matrix `nxt[i][j]` (erster Schritt einer kürzesten Route von i nach j) packt jede Route in O(Länge) aus.
Alles ist eigene Umsetzung auf dem CSR-Graphen aus jo_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

from dataclasses import dataclass, field

import numpy as np

INF = float("inf")
VARIANTS = ("classic", "skip", "numpy")
LOOP_ORDERS = ("kij", "kji", "ikj", "jki", "ijk", "jik")     # nur k als äußerste Schleife ist richtig (kij, kji)


@dataclass
class FloydWarshall:
    dist: np.ndarray                                   # n x n; bei negativem Zyklus der Stand beim Ende (unbrauchbar)
    nxt: np.ndarray                                    # n x n; erster Knoten nach i auf der Route nach j, -1 = keine Route
    negative_cycle: bool                               # ein Wert auf der Diagonale ist negativ
    counters: dict = field(default_factory=dict)       # comparisons (Vergleiche), improvements (Verbesserungen), skip_comparisons (nur Paare mit endlichen d(i,k), d(k,j))
    history: list = field(default_factory=list)        # nur mit trace=True: (k erledigt, dist-Kopie, nxt-Kopie, bool-Matrix der seit dem letzten Eintrag geänderten Zellen); bei n > 100 nur etwa 40 Einträge

    @property
    def n(self):
        return len(self.dist)

    def route(self, i, j):
        """Kürzeste Route i -> j als Knotenfolge; leer, wenn es keine gibt oder wenn ein negativer Zyklus die Rekonstruktion unmöglich macht."""
        i, j = int(i), int(j)
        if self.nxt[i, j] < 0 or self.negative_cycle:
            return []
        path, cur = [i], i
        while cur != j:
            cur = int(self.nxt[cur, j])
            path.append(cur)
            if len(path) > self.n:
                return []
        return path

    def diagonal_negative(self):
        return np.where(np.diag(self.dist) < 0)[0]


def initial_matrix(g):
    """Ausgangsmatrix: 0 auf der Diagonale, Kantenkosten (bei Parallelkanten die billigste), sonst unendlich; dazu die Nachfolger-Matrix."""
    n = g.n
    d = np.full((n, n), INF)
    nxt = np.full((n, n), -1, dtype=np.int64)
    src = np.repeat(np.arange(n), g.degree())
    for u, v, w in zip(src.tolist(), g.indices.tolist(), g.weight.tolist()):
        if w < d[u, v]:
            d[u, v], nxt[u, v] = w, v
    idx = np.arange(n)
    d[idx, idx] = np.minimum(d[idx, idx], 0.0)
    nxt[idx, idx] = idx
    return d, nxt


def floyd_warshall(g, variant="numpy", trace=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    n = g.n
    d, nxt = initial_matrix(g)
    counters = {"comparisons": 0, "improvements": 0, "skip_comparisons": 0}
    history = []
    stride = 1 if n <= 100 else max(1, n // 40)         # große Matrizen: nur etwa 40 Zwischenstände (Speicher), die Änderungen dazwischen werden gesammelt
    acc = np.zeros((n, n), dtype=bool)
    if variant == "numpy":
        for k in range(n):
            col, row = d[:, k], d[k, :]
            fin_c, fin_r = np.isfinite(col), np.isfinite(row)
            counters["skip_comparisons"] += int(fin_c.sum()) * int(fin_r.sum())
            counters["comparisons"] += n * n
            cand = col[:, None] + row[None, :]
            mask = cand < d
            imp = int(mask.sum())
            counters["improvements"] += imp
            if imp:
                nxt = np.where(mask, nxt[:, k][:, None], nxt)
                d = np.where(mask, cand, d)
            if trace:
                acc |= mask
                if (k + 1) % stride == 0 or k == n - 1:
                    history.append((k + 1, d.copy(), nxt.astype(np.int16), acc.copy()))
                    acc[:] = False
    else:
        dl = d.tolist()
        nl = nxt.tolist()
        for k in range(n):
            dk = dl[k]
            changed = []
            js = [j for j in range(n) if dk[j] != INF] if variant == "skip" else range(n)
            for i in range(n):
                dik = dl[i][k]
                if variant == "skip" and dik == INF:
                    continue
                di, ni = dl[i], nl[i]
                for j in js:
                    counters["comparisons"] += 1
                    cand = dik + dk[j]
                    if cand < di[j]:
                        di[j], ni[j] = cand, nl[i][k]
                        counters["improvements"] += 1
                        if trace:
                            changed.append((i, j))
            if variant == "classic":
                counters["skip_comparisons"] = 0
            if trace:
                for i, j in changed:
                    acc[i, j] = True
                if (k + 1) % stride == 0 or k == n - 1:
                    history.append((k + 1, np.array(dl), np.array(nl, dtype=np.int16), acc.copy()))
                    acc[:] = False
        d, nxt = np.array(dl), np.array(nl, dtype=np.int64)
        if variant == "skip":
            counters["skip_comparisons"] = counters["comparisons"]
    return FloydWarshall(d, nxt, bool((np.diag(d) < 0).any()), counters, history)


def route_from(nxt, i, j):
    """Route aus einem Nachfolger-Snapshot (wie FloydWarshall.route, aber ohne Zyklus-Prüfung: bricht bei einer Schleife ab und gibt dann [] zurück)."""
    i, j = int(i), int(j)
    if nxt[i, j] < 0:
        return []
    path, cur = [i], i
    while cur != j:
        cur = int(nxt[cur, j])
        if cur < 0:
            return []
        path.append(cur)
        if len(path) > len(nxt):
            return []
    return path


def hop_matrix(fw):
    """Zahl der Kanten der kürzesten Route für jedes Paar (-1 = keine Route), vektorisiert über die Nachfolger-Matrix; nur ohne negativen Zyklus."""
    n = fw.n
    nxt = fw.nxt
    cur = np.tile(np.arange(n)[:, None], (1, n))
    tgt = np.tile(np.arange(n)[None, :], (n, 1))
    hops = np.where(nxt >= 0, 0, -1)
    active = (nxt >= 0) & (cur != tgt)
    for _ in range(n):
        if not active.any():
            break
        cur = np.where(active, nxt[cur, tgt], cur)
        hops = np.where(active, hops + 1, hops)
        active = active & (cur != tgt)
    return hops


def floyd_warshall_order(g, order="kij"):
    """Lehr-Falle: dieselben drei Schleifen in anderer Reihenfolge (nur Entfernungen, ohne Zähler). Nur wenn k die äußerste Schleife ist, sind alle Zwischenknoten rechtzeitig berücksichtigt."""
    if order not in LOOP_ORDERS:
        raise ValueError(order)
    n = g.n
    d, _ = initial_matrix(g)
    dl = d.tolist()
    rng = range(n)
    loops = {"k": rng, "i": rng, "j": rng}
    idx = dict(k=0, i=0, j=0)
    for a in loops[order[0]]:
        idx[order[0]] = a
        for b in loops[order[1]]:
            idx[order[1]] = b
            for c in loops[order[2]]:
                idx[order[2]] = c
                i, j, k = idx["i"], idx["j"], idx["k"]
                if dl[i][k] + dl[k][j] < dl[i][j]:
                    dl[i][j] = dl[i][k] + dl[k][j]
    return np.array(dl)


def affected_pairs(fw):
    """Bei negativen Zyklen: Paare (i, j), für die es keine kürzeste Route gibt, weil eine Route von i nach j über einen Knoten auf einem negativen Zyklus führen kann (boolesche n x n Matrix).
    Die Erreichbarkeit stimmt auch bei negativen Zyklen (endlich heißt: es gibt eine Route)."""
    d = fw.dist
    bad = np.where(np.diag(d) < 0)[0]
    if not len(bad):
        return np.zeros(d.shape, dtype=bool)
    fin = np.isfinite(d)
    return (fin[:, bad].astype(np.int64) @ fin[bad, :].astype(np.int64)) > 0


def restricted_reference(g, k):
    """Unabhängige Vergleichsrechnung für die Tests: billigste Kosten i -> j, wenn nur die Knoten 0 .. k-1 Zwischenknoten sein dürfen (für jeden Start Bellman-Ford auf den erlaubten Kanten; nur ohne negative Zyklen)."""
    n = g.n
    src = np.repeat(np.arange(n), g.degree()).tolist()
    dst, w = g.indices.tolist(), g.weight.tolist()
    out = np.full((n, n), INF)
    for s in range(n):
        dist = [INF] * n
        dist[s] = 0.0
        for _ in range(n):
            changed = False
            for u, v, c in zip(src, dst, w):
                if (u == s or u < k) and dist[u] + c < dist[v]:
                    dist[v] = dist[u] + c
                    changed = True
            if not changed:
                break
        out[s] = dist
    return out


# --- Vergleich: n-mal einzeln (Zähler) ---------------------------------------------------------------------------------------------------------

def all_pairs_by_dijkstra(g):
    """n-mal Dijkstra (nur ohne negative Kanten richtig): Entfernungsmatrix und Kantenprüfungen insgesamt."""
    import jo_sp
    d = np.full((g.n, g.n), INF)
    checks = 0
    for s in range(g.n):
        r = jo_sp.dijkstra(g, s)
        d[s] = r.dist
        checks += r.counters["checks"]
    return d, checks


def all_pairs_by_bellman_ford(g, variant="early_stop"):
    """n-mal Bellman-Ford: Entfernungsmatrix, Kantenprüfungen insgesamt, Zahl der Startknoten mit negativem Zyklus."""
    import jo_sp
    d = np.full((g.n, g.n), INF)
    checks = cycles = 0
    for s in range(g.n):
        r = jo_sp.bellman_ford(g, s, variant)
        checks += r.counters["checks"]
        if r.negative_cycle:
            cycles += 1
        else:
            d[s] = r.dist
    return d, checks, cycles
