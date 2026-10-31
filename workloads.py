"""Reference-string generators. F = number of frames. All take an np.random.Generator."""
import numpy as np


def random_uniform(n, F, rng):
    return rng.integers(0, 4 * F, n)


def zipf_skew(n, F, rng, a=1.1):
    U = 8 * F
    p = np.arange(1, U + 1) ** -a
    p /= p.sum()
    return rng.permutation(U)[rng.choice(U, n, p=p)]


def working_set(n, F, rng, phases=10, hot_frac=0.75, p_hot=0.9):
    U, out = 16 * F, []
    per = n // phases
    for i in range(phases):
        m = per if i < phases - 1 else n - per * (phases - 1)
        hot = rng.choice(U, max(2, int(hot_frac * F)), replace=False)
        use_hot = rng.random(m) < p_hot
        out.append(np.where(use_hot, rng.choice(hot, m), rng.integers(0, U, m)))
    return np.concatenate(out)


def loop(n, F, rng, mult=1.25, noise=0.02):
    L = int(mult * F)
    x = (np.arange(n) + rng.integers(0, L)) % L
    flip = rng.random(n) < noise
    return np.where(flip, rng.integers(0, 4 * F, n), x)


def scan_mix(n, F, rng, hot_frac=0.5, p_hot=0.5):
    """Hot pages interleaved with a never-repeating sequential scan (cache pollution)."""
    hot = np.arange(max(2, int(hot_frac * F)))
    is_hot = rng.random(n) < p_hot
    scan = 10_000 + np.cumsum(~is_hot)
    return np.where(is_hot, rng.choice(hot, n), scan)


def phase_mix(n, F, rng, segments=8):
    """Behaviour changes over time -- the case an adaptive policy is built for."""
    gens = [loop, working_set, zipf_skew, scan_mix, random_uniform]
    per, out = n // segments, []
    for i in range(segments):
        m = per if i < segments - 1 else n - per * (segments - 1)
        out.append(gens[rng.integers(len(gens))](m, F, rng))
    return np.concatenate(out)


PATTERNS = dict(random=random_uniform, zipf=zipf_skew, working_set=working_set,
                loop=loop, scan_mix=scan_mix, phase_mix=phase_mix)
