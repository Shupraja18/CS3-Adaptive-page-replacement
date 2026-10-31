"""Statistical summaries + plots from results/raw_results.csv.
Outputs: results/summary.csv, results/significance.csv, plots/*.png
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("results/raw_results.csv")
REQ = ["FIFO", "LRU", "LFU", "Clock"]
ORDER = ["OPT", "FIFO", "LRU", "LFU", "Clock", "MRU", "Adaptive-F", "Adaptive-S"]
pats = list(dict.fromkeys(df.pattern))
points = sorted(df.n_refs.unique())
os.makedirs("plots", exist_ok=True)

# ---------- summary (mean, std, 95% CI over seeds) ----------
def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

g = df.groupby(["pattern", "n_refs", "frames", "policy"])
summ = g.agg(faults_mean=("faults", "mean"), faults_std=("faults", "std"),
             hit_mean=("hit_ratio", "mean"), hit_std=("hit_ratio", "std"),
             hit_ci95=("hit_ratio", ci95),
             time_mean_s=("time_s", "mean"), time_std_s=("time_s", "std"),
             switches_mean=("rule_switches", "mean")).reset_index()
summ.to_csv("results/summary.csv", index=False)

# ---------- significance: paired Wilcoxon on fault counts, per seed ----------
rows = []
for (pat, n, F), d in df.groupby(["pattern", "n_refs", "frames"]):
    piv = d.pivot(index="seed", columns="policy", values="faults")
    best_fixed = piv[REQ].mean().idxmin()
    for ad in ["Adaptive-F", "Adaptive-S"]:
        for ref in dict.fromkeys([best_fixed, "LRU"]):
            diff = piv[ad] - piv[ref]
            try:
                p = stats.wilcoxon(piv[ad], piv[ref]).pvalue if (diff != 0).any() else 1.0
            except ValueError:
                p = 1.0
            rows.append(dict(pattern=pat, n_refs=n, adaptive=ad, versus=ref,
                             is_best_required_fixed=(ref == best_fixed),
                             mean_fault_change_pct=100 * (piv[ad].mean() / piv[ref].mean() - 1),
                             wilcoxon_p=p, significant_p05=p < 0.05))
        rows.append(dict(pattern=pat, n_refs=n, adaptive=ad, versus="OPT",
                         is_best_required_fixed=False,
                         mean_fault_change_pct=100 * (piv[ad].mean() / piv["OPT"].mean() - 1),
                         wilcoxon_p=np.nan, significant_p05=np.nan))
sig = pd.DataFrame(rows)
sig.to_csv("results/significance.csv", index=False)

COL = {"OPT": "#999999", "FIFO": "#d62728", "LRU": "#1f77b4", "LFU": "#ff7f0e", "Clock": "#2ca02c",
       "MRU": "#8c564b", "Adaptive-F": "#9467bd", "Adaptive-S": "#000000"}

# ---------- Fig 1: hit ratio per pattern at each operating point ----------
fig, axes = plt.subplots(1, len(points), figsize=(5.2 * len(points), 3.8), sharey=True)
for ax, n in zip(np.atleast_1d(axes), points):
    s = summ[summ.n_refs == n]
    w = 0.1
    for i, pol in enumerate(ORDER):
        v = [s[(s.pattern == p) & (s.policy == pol)].hit_mean.values[0] for p in pats]
        e = [s[(s.pattern == p) & (s.policy == pol)].hit_ci95.values[0] for p in pats]
        ax.bar(np.arange(len(pats)) + (i - 3.5) * w, v, w, yerr=e, color=COL[pol], label=pol, capsize=1)
    F = s.frames.iloc[0]
    ax.set_title(f"N={n:,} refs, {F} frames", fontsize=10)
    ax.set_xticks(range(len(pats))); ax.set_xticklabels(pats, rotation=30, ha="right", fontsize=8)
np.atleast_1d(axes)[0].set_ylabel("Hit ratio (mean, 95% CI)")
np.atleast_1d(axes)[-1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("plots/fig1_hit_ratio.png", dpi=200); plt.close()

# ---------- Fig 2: scalability of page faults (fault rate vs N) ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, pat in zip(axes.ravel(), pats):
    for pol in ORDER:
        s = summ[(summ.pattern == pat) & (summ.policy == pol)].sort_values("n_refs")
        ax.plot(s.n_refs, 1 - s.hit_mean, marker="o", ms=3, color=COL[pol], label=pol,
                lw=2 if pol.startswith("Adaptive") else 1)
    ax.set_title(pat, fontsize=10); ax.set_xscale("log"); ax.set_xlabel("references N"); ax.set_ylabel("page-fault rate")
axes[0, 0].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig("plots/fig2_fault_rate_scaling.png", dpi=200); plt.close()

# ---------- Fig 3: execution time ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
for pol in ORDER:
    s = summ[summ.policy == pol].groupby("n_refs").time_mean_s.mean()
    ax.plot(s.index, s.values, marker="o", color=COL[pol], label=pol, lw=2 if pol.startswith("Adaptive") else 1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("references N"); ax.set_ylabel("simulation time (s), mean over patterns")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig("plots/fig3_execution_time.png", dpi=200); plt.close()

# ---------- Fig 4: gap to OPT (%) -- middle operating point ----------
n = points[len(points) // 2]
s = summ[summ.n_refs == n]
fig, ax = plt.subplots(figsize=(7, 3.8))
cands = [p for p in ORDER if p != "OPT"]
for i, pol in enumerate(cands):
    v = [100 * (s[(s.pattern == p) & (s.policy == pol)].faults_mean.values[0] /
                s[(s.pattern == p) & (s.policy == "OPT")].faults_mean.values[0] - 1) for p in pats]
    ax.bar(np.arange(len(pats)) + (i - 3) * 0.12, v, 0.12, color=COL[pol], label=pol)
ax.set_xticks(range(len(pats))); ax.set_xticklabels(pats, rotation=20, fontsize=8)
ax.set_ylabel("extra faults vs OPT (%)"); ax.set_title(f"Gap to optimal, N={n:,}", fontsize=10); ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("plots/fig4_gap_to_opt.png", dpi=200); plt.close()

# ---------- Fig 5: adaptive rule timeline on phase_mix ----------
for ad in ["Adaptive-F", "Adaptive-S"]:
    f = f"results/selection_log_{ad}.csv"
    if os.path.exists(f):
        lg = pd.read_csv(f)
        rules = ["fifo", "lru", "lfu", "clock", "mru"]
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.scatter(lg.ref_index, lg.rule.map({r: i for i, r in enumerate(rules)}), s=6, c="k")
        ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules); ax.set_xlabel("reference index")
        ax.set_title(f"{ad}: rule chosen per window (phase_mix, seed 0)", fontsize=10)
        plt.tight_layout(); plt.savefig(f"plots/fig5_timeline_{ad}.png", dpi=200); plt.close()

# ---------- console report ----------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
t = summ.pivot_table(index=["n_refs", "pattern"], columns="policy", values="hit_mean")[ORDER]
print("MEAN HIT RATIO\n", t.round(3))
print("\nSIGNIFICANCE vs best required fixed policy\n",
      sig[sig.is_best_required_fixed][["pattern", "n_refs", "adaptive", "versus", "mean_fault_change_pct", "wilcoxon_p"]].round(3).to_string(index=False))
