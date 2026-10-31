"""Principal experiment: 6 reference patterns x 3 operating points x R seeds x 8 policies.
Usage:  python run_experiments.py            (full, R=10)
        python run_experiments.py --quick    (smoke test, tiny sizes)
Output: results/raw_results.csv, results/config.json, results/selection_log_phase_mix.csv
"""
import argparse, csv, json, os, platform, sys, time
import numpy as np
from simulators import run_policy
from workloads import PATTERNS

POLICIES = ["OPT", "FIFO", "LRU", "LFU", "Clock", "MRU", "Adaptive-F", "Adaptive-S"]
OPERATING_POINTS = [(10_000, 16), (50_000, 32), (100_000, 64)]   # (references N, frames F)

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
points = [(2_000, 8), (4_000, 16), (6_000, 32)] if a.quick else OPERATING_POINTS
reps = 2 if a.quick else a.reps

os.makedirs("results", exist_ok=True)
json.dump(dict(policies=POLICIES, operating_points=points, reps=reps, seeds=list(range(reps)),
               patterns=list(PATTERNS), python=sys.version, platform=platform.platform(),
               numpy=np.__version__), open("results/config.json", "w"), indent=2)

t0 = time.time()
with open("results/raw_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["pattern", "n_refs", "frames", "seed", "policy", "faults", "hit_ratio",
                "fault_rate", "time_s", "rule_switches"])
    for n, F in points:
        for pat, gen in PATTERNS.items():
            for seed in range(reps):
                refs = gen(n, F, np.random.default_rng(seed))
                for pol in POLICIES:
                    r = run_policy(pol, refs, F)
                    w.writerow([pat, n, F, seed, pol, r["faults"], f"{r['hit_ratio']:.6f}",
                                f"{1 - r['hit_ratio']:.6f}", f"{r['time_s']:.6f}", r["switches"]])
                    if pat == "phase_mix" and seed == 0 and pol.startswith("Adaptive") and n == points[1][0]:
                        with open(f"results/selection_log_{pol}.csv", "w") as lf:
                            lf.write("ref_index,rule\n")
                            lf.writelines(f"{t},{rule}\n" for t, rule in r["log"])
                fh.flush()
            print(f"done N={n} F={F} pattern={pat}  [{time.time() - t0:.0f}s]", flush=True)
print("finished")
