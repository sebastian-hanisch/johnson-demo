"""Prioritätswarteschlange (Lazy-Heap mit Zählern) für Dijkstra.

Schnittstelle: `push_or_decrease(node, key)`, `pop_min() -> (key, node)`, `len(queue)`."""

import heapq


class _Counters:
    def __init__(self):
        self.pushes = self.decrease_keys = self.pops = self.work = 0

    def as_dict(self):
        return {"pushes": self.pushes, "decrease_keys": self.decrease_keys, "pops": self.pops, "work": self.work}




class LazyHeapQueue(_Counters):
    """Binärheap ohne Decrease-Key (`heapq`): ein gesenkter Schlüssel legt einen zweiten Eintrag an, der alte wird beim Entnehmen übersprungen (faule Löschung).
    Schlüsselvergleiche macht `heapq` in C, sie sind hier nicht zählbar (work = 0)."""

    def __init__(self, n=0, max_weight=None):
        super().__init__()
        self.heap, self.best = [], {}
        self.stale_pops = 0

    def __len__(self):
        return len(self.best)

    def push_or_decrease(self, node, key):
        if node in self.best:
            self.decrease_keys += 1
        else:
            self.pushes += 1
        self.best[node] = key
        heapq.heappush(self.heap, (key, node))

    def pop_min(self):
        while True:
            key, node = heapq.heappop(self.heap)
            if self.best.get(node) == key:
                del self.best[node]
                self.pops += 1
                return key, node
            self.stale_pops += 1



QUEUES = {"lazy": LazyHeapQueue}
