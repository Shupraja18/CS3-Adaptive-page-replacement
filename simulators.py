"""Page-replacement simulators + two adaptive policies (CS3).

All online policies share one implementation (MetaCache) that keeps metadata for
every rule (load time, last use, frequency, reference bit). The eviction *rule*
can therefore be switched at any moment, which is what the adaptive policies do.
OPT (Belady's optimal) is offline: it needs the whole future reference string,
so it is only used as a lower bound, never as an adaptive candidate.
"""
import time
from collections import Counter

RULES = ("fifo", "lru", "lfu", "clock", "mru")   # candidate eviction rules
REQUIRED = ("FIFO", "LRU", "LFU", "Clock")        # baselines demanded by the case study


class MetaCache:
    def __init__(self, frames, rule="lru"):
        self.frames, self.rule = frames, rule
        self.t = 0            # logical time (reference count)
        self.qc = 0           # counter for clock queue order
        self.faults = 0
        self.loaded, self.last, self.freq, self.ref, self.qt = {}, {}, {}, {}, {}

    def access(self, p):
        """Return True on page fault."""
        self.t += 1
        if p in self.loaded:
            self.last[p] = self.t
            self.freq[p] += 1
            self.ref[p] = 1
            return False
        self.faults += 1
        if len(self.loaded) >= self.frames:
            v = self._victim()
            for d in (self.loaded, self.last, self.freq, self.ref, self.qt):
                del d[v]
        self.qc += 1
        self.loaded[p] = self.t
        self.last[p] = self.t
        self.freq[p] = 1
        self.ref[p] = 1
        self.qt[p] = self.qc
        return True

    def age(self):
        """Halve frequency counts so stale history cannot pin pages (LFU aging)."""
        for q in self.freq:
            self.freq[q] = max(1, self.freq[q] >> 1)

    def _victim(self):
        r = self.rule
        if r == "fifo":
            return min(self.loaded, key=self.loaded.get)
        if r == "lru":
            return min(self.last, key=self.last.get)
        if r == "mru":
            return max(self.last, key=self.last.get)
        if r == "lfu":   # least frequently used (count while resident), tie -> LRU
            return min(self.loaded, key=lambda q: (self.freq[q], self.last[q]))
        if r == "clock":  # second-chance / clock: hand sweeps in load order
            while True:
                c = min(self.qt, key=self.qt.get)
                if self.ref[c]:
                    self.ref[c] = 0
                    self.qc += 1
                    self.qt[c] = self.qc   # hand moves past it
                else:
                    return c
        raise ValueError(r)


def simulate_opt(refs, frames):
    """Belady's optimal: evict the page whose next use is farthest in the future."""
    n, INF = len(refs), float("inf")
    nxt, seen = [INF] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = seen.get(refs[i], INF)
        seen[refs[i]] = i
    res, faults = {}, 0
    for i, p in enumerate(refs):
        if p in res:
            res[p] = nxt[i]
            continue
        faults += 1
        if len(res) >= frames:
            del res[max(res, key=res.get)]
        res[p] = nxt[i]
    return faults


# ----------------------------------------------------------------- adaptivity
DEFAULT_PARAMS = dict(fit=0.9, loop_seq=0.5, loop_reuse=0.3, scan_reuse=0.1,
                      skew=0.45, lru_reuse=0.5, loop_rule="mru", aging=False)


def window_features(win, frames):
    """Observable locality features of one window of references."""
    W = len(win)
    distinct = len(set(win))
    seq = sum(1 for a, b in zip(win, win[1:]) if b == a + 1) / max(1, W - 1)
    cnt = sorted(Counter(win).values(), reverse=True)
    k = max(1, distinct // 5)
    return dict(ws_ratio=distinct / frames,          # working set / memory
                reuse=1 - distinct / W,              # fraction of repeated refs
                seq=seq,                             # sequential (stride +1) fraction
                skew=sum(cnt[:k]) / W)               # share of refs hitting top-20% pages


def choose_rule(f, P=DEFAULT_PARAMS):
    if f["ws_ratio"] <= P["fit"]:
        return "lru"                                  # working set fits: recency is safe
    if f["reuse"] < P["scan_reuse"]:
        return "fifo"                                 # pure scan: nothing helps, be cheap
    if f["seq"] >= P["loop_seq"] and f["reuse"] >= P["loop_reuse"]:
        return P["loop_rule"]                         # cyclic loop larger than memory
    if f["skew"] >= P["skew"]:
        return "lfu"                                  # strong frequency skew / scan pollution
    return "lru" if f["reuse"] >= P["lru_reuse"] else "clock"


class AdaptiveFeature(MetaCache):
    """Adaptive-F: every window, measure locality features and pick the rule."""
    def __init__(self, frames, window=None, params=None):
        super().__init__(frames, "lru")
        self.W = window or 4 * frames
        self.P = params or DEFAULT_PARAMS
        self.buf, self.log = [], []

    def access(self, p):
        r = super().access(p)
        self.buf.append(p)
        if len(self.buf) == self.W:
            f = window_features(self.buf, self.frames)
            self.rule = choose_rule(f, self.P)
            if self.P["aging"]:
                self.age()
            self.log.append((self.t, self.rule))
            self.buf = []
        return r


class AdaptiveShadow(MetaCache):
    """Adaptive-S: ghost caches replay every candidate rule on the recent window;
    the real cache adopts the rule with the fewest faults (5 % hysteresis)."""
    def __init__(self, frames, window=None, hysteresis=0.05, aging=False):
        super().__init__(frames, "lru")
        self.aging = aging
        self.W = window or 4 * frames
        self.h = hysteresis
        self.ghosts = {r: MetaCache(frames, r) for r in RULES}
        self.base = {r: 0 for r in RULES}
        self.log = []

    def access(self, p):
        r = super().access(p)
        for g in self.ghosts.values():
            g.access(p)
        if self.t % self.W == 0:
            wf = {k: g.faults - self.base[k] for k, g in self.ghosts.items()}
            best = min(wf, key=wf.get)
            if wf[best] < wf[self.rule] * (1 - self.h):
                self.rule = best
            self.base = {k: g.faults for k, g in self.ghosts.items()}
            if self.aging:
                self.age()
                for g in self.ghosts.values():
                    g.age()
            self.log.append((self.t, self.rule))
        return r


# ------------------------------------------------------------------- runner
def run_policy(name, refs, frames):
    """Run one policy; return dict(faults, hit_ratio, time_s, switches, log)."""
    refs = list(map(int, refs))
    t0 = time.perf_counter()
    log, sw = [], 0
    if name == "OPT":
        faults = simulate_opt(refs, frames)
    else:
        if name == "Adaptive-F":
            c = AdaptiveFeature(frames)
        elif name == "Adaptive-S":
            c = AdaptiveShadow(frames)
        else:
            c = MetaCache(frames, name.lower())
        for p in refs:
            c.access(p)
        faults = c.faults
        log = getattr(c, "log", [])
        sw = sum(1 for a, b in zip(log, log[1:]) if a[1] != b[1])
    dt = time.perf_counter() - t0
    return dict(faults=faults, hit_ratio=1 - faults / len(refs), time_s=dt,
                switches=sw, log=log)
