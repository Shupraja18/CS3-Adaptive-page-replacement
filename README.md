# CS3 – Adaptive Page Replacement (BCSE303L Operating Systems)

## Layout
| File | Purpose |
|---|---|
| `simulators.py` | FIFO, LRU, LFU, Clock, Optimal + MRU (extra) and the two adaptive policies |
| `workloads.py` | Six reference-string generators (random, zipf, working_set, loop, scan_mix, phase_mix) |
| `tests/test_validation.py` | Validation on controlled strings (textbook, Belady's anomaly, property checks) |
| `run_experiments.py` | Principal experiment → `results/raw_results.csv`, `results/config.json` |
| `analyze.py` | Mean/std/95 % CI, Wilcoxon tests, plots → `results/summary.csv`, `results/significance.csv`, `plots/` |

## Reproduce
```bash
pip install numpy pandas matplotlib scipy
python tests/test_validation.py      # must print ALL VALIDATION CHECKS PASSED
python run_experiments.py            # ~10-15 min on one core (use --quick for a smoke test)
python analyze.py
```

## Design summary
* One `MetaCache` stores load-time, last-use, frequency and reference-bit for every resident page,
  so the eviction rule can be switched at any reference. Fixed policies = fixed rule.
* **Adaptive-F (feature-based)**: every window of W = 4·frames references it computes
  working-set ratio, reuse ratio, sequential-stride fraction and frequency skew, then picks a rule
  (`choose_rule`). Thresholds were tuned on seeds 1000+, evaluated on seeds 0-9 (no leakage).
* **Adaptive-S (shadow)**: ghost caches replay every candidate rule; the real cache adopts the rule
  with the fewest faults in the last window (5 % hysteresis). Costs ~6× simulation time.
* OPT is offline (needs the future) – a lower bound only, never an adaptive candidate.
* MRU is an *extra* candidate/baseline (useful for loops larger than memory).
* LFU counts references while a page is resident (counts reset on eviction); ties broken by LRU.
* Clock = second-chance with a sweeping hand; reference bit set on load and on hit.
