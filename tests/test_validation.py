"""Validation on controlled reference strings (run: python tests/test_validation.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulators import run_policy
from workloads import PATTERNS

BELADY = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]  # Silberschatz et al.

def f(name, refs, fr): return run_policy(name, refs, fr)["faults"]

def stack_distance_lru(refs, fr):          # independent LRU implementation
    stack, faults = [], 0
    for p in refs:
        if p in stack:
            d = len(stack) - stack.index(p)
            faults += d > fr
            stack.remove(p)
        else:
            faults += 1
        stack.append(p)
    return faults

checks = [
    ("FIFO textbook, 3 frames",  f("FIFO", TEXTBOOK, 3), 15),
    ("LRU  textbook, 3 frames",  f("LRU", TEXTBOOK, 3), 12),
    ("OPT  textbook, 3 frames",  f("OPT", TEXTBOOK, 3), 9),
    ("FIFO Belady, 3 frames",    f("FIFO", BELADY, 3), 9),
    ("FIFO Belady, 4 frames",    f("FIFO", BELADY, 4), 10),   # Belady's anomaly
    ("LRU  Belady, 3 frames",    f("LRU", BELADY, 3), 10),
    ("LRU  Belady, 4 frames",    f("LRU", BELADY, 4), 8),
    ("OPT  Belady, 3 frames",    f("OPT", BELADY, 3), 7),
    ("OPT  Belady, 4 frames",    f("OPT", BELADY, 4), 6),
]
ok = True
for name, got, exp in checks:
    s = "PASS" if got == exp else "FAIL"; ok &= got == exp
    print(f"{s}  {name}: got {got}, expected {exp}")

rng = np.random.default_rng(1)
for pat, g in PATTERNS.items():
    refs = list(map(int, g(3000, 16, rng)))
    res = {n: f(n, refs, 16) for n in ["OPT", "FIFO", "LRU", "LFU", "Clock", "MRU", "Adaptive-F", "Adaptive-S"]}
    c1 = res["OPT"] == min(res.values())
    c2 = all(v >= len(set(refs)) for v in res.values())
    c3 = res["LRU"] == stack_distance_lru(refs, 16)
    s = "PASS" if (c1 and c2 and c3) else "FAIL"; ok &= (c1 and c2 and c3)
    print(f"{s}  property checks on '{pat}' (OPT minimal, >= cold misses, LRU == stack-distance LRU)")

c = all(f(n, TEXTBOOK, 8) == len(set(TEXTBOOK)) for n in ["FIFO", "LRU", "LFU", "Clock", "OPT"])
print("PASS" if c else "FAIL", " frames >= distinct pages -> faults == distinct pages"); ok &= c
print("\nALL VALIDATION CHECKS PASSED" if ok else "\nSOME CHECKS FAILED"); sys.exit(0 if ok else 1)
