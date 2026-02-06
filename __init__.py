"""
coherence_router — Universal signal dynamics classifier

Feed any time series. Get physics back.

    from coherence_router import classify
    result = classify([0.5, 0.8, 0.3, 0.9, 0.1, ...])
    print(result.band)       # "MOLECULAR"
    print(result.lyapunov)   # 0.3647  (positive = chaos)
    print(result.gap)        # 0.0     (no spectral gap)
    print(result.entropy)    # 4.07 bits
    print(result.coherence)  # 0.42

That's it. Works on CPU metrics, stock prices, heartbeats,
sensor data, text, anything with sequential values.

Math references:
  Quadratic maps:     Devaney 2003, May 1976 Nature 261
  Fixed points:       Banach 1922 Fund. Math. 3
  Spectral gap:       Perron 1907 Math. Ann. 64
  Lyapunov exponents: Oseledets 1968 Trans. Moscow Math. 19
  Shannon entropy:    Shannon 1948 Bell Syst. Tech. J. 27
  Stat. mechanics:    Boltzmann 1872, Gibbs 1902
  Least squares:      Gauss 1809 Theoria Motus

TIG conjectures (testable, not established):
  S* = σ(1-σ*)V*A*   coherence measure
  σ = 0.991           coupling constant (chosen)
  T* = 5/7            critical threshold (chosen)

NON-COMMERCIAL TESTING — 7Site LLC — 7sitellc.com
The math belongs to everyone.
"""

__version__ = "0.1.0"
__author__ = "Brayden — 7Site LLC"
__license__ = "MIT"

import math
from collections import defaultdict

# ─── Constants ───

SIGMA = 0.991
D_STAR = SIGMA / (1 + SIGMA)  # 0.49774
T_STAR = 5.0 / 7.0            # 0.71429

BANDS = {
    0: ("VOID",      0.0,  "Orbit diverges"),
    1: ("SPARK",     0.1,  "Slow divergence"),
    2: ("FLOW",      0.3,  "Marginal stability (λ_L ≈ 0)"),
    3: ("MOLECULAR", 0.5,  "Chaos (λ_L > 0)"),
    4: ("CELLULAR",  0.7,  "Periodic orbit"),
    5: ("ORGANIC",   0.85, "Slow convergence (|λ| near 1)"),
    6: ("CRYSTAL",   1.0,  "Fast convergence (|λ| < 0.5)"),
}


# ─── Result object ───

class Result:
    """Physics of a single time series."""
    __slots__ = ('a', 'b', 'c', 'band', 'band_name', 'band_weight',
                 'fixed_point', 'eigenvalue', 'gap', 'lyapunov',
                 'entropy', 'energy', 'stable', 'mse')

    def __repr__(self):
        return (f"Result(band='{self.band_name}', gap={self.gap:.4f}, "
                f"lyap={self.lyapunov:.4f}, H={self.entropy:.3f})")

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class Coherence:
    """Coherence of multiple classified signals."""
    __slots__ = ('S_star', 'V_star', 'A_star', 'k', 'n',
                 'above_threshold', 'bands', 'results')

    def __repr__(self):
        return f"Coherence(S*={self.S_star:.6f}, n={self.n}, above_T*={self.above_threshold})"


# ─── Core math ───

def _eval(a, b, c, x):
    return a * x * x + b * x + c

def _deriv(a, b, c, x):
    return 2.0 * a * x + b

def _fixed_points(a, b, c):
    """Solve ax²+(b-1)x+c = 0. Exact quadratic formula."""
    A, B, C = a, b - 1.0, c
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return []
        x = -C / B
        return [(x, _deriv(a, b, c, x))]
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    s = math.sqrt(disc)
    x1, x2 = (-B + s) / (2 * A), (-B - s) / (2 * A)
    return [(x1, _deriv(a, b, c, x1)), (x2, _deriv(a, b, c, x2))]

def _stable_fp(a, b, c):
    fps = _fixed_points(a, b, c)
    if not fps:
        return None, None
    best = min(fps, key=lambda p: abs(p[1]))
    return best

def _orbit(a, b, c, x0=0.5, n=300):
    traj = [x0]
    x = x0
    for _ in range(n):
        x = _eval(a, b, c, x)
        if not math.isfinite(x) or abs(x) > 1e15:
            break
        traj.append(x)
        if len(traj) > 2 and abs(traj[-1] - traj[-2]) < 1e-12:
            break
    return traj

def _lyapunov(a, b, c, x0=0.5, n=500):
    x, total, count = x0, 0.0, 0
    for _ in range(n):
        d = abs(_deriv(a, b, c, x))
        total += math.log(max(d, 1e-15))
        count += 1
        x = _eval(a, b, c, x)
        if not math.isfinite(x) or abs(x) > 1e15:
            break
    return total / max(count, 1)

def _entropy(traj, bins=20):
    if len(traj) < 10:
        return 0.0
    lo, hi = min(traj), max(traj)
    if hi - lo < 1e-12:
        return 0.0
    counts = [0] * bins
    for x in traj:
        idx = min(bins - 1, int((x - lo) / (hi - lo) * bins))
        counts[idx] += 1
    total = len(traj)
    H = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H

def _band(a, b, c):
    traj = _orbit(a, b, c, 0.5, 300)
    # Quick convergence → check fixed point
    if 2 <= len(traj) < 5 and abs(traj[-1]) < 1e10:
        xfp, lam = _stable_fp(a, b, c)
        if xfp is not None and abs(lam) < 1.0:
            return 6 if abs(lam) < 0.5 else 5
    # Divergent
    if len(traj) < 5 or abs(traj[-1]) > 1e10:
        return 1 if len(traj) > 30 else 0
    # Periodic
    tail = traj[-60:] if len(traj) >= 60 else traj[-20:]
    if len(tail) >= 6:
        for p in range(2, min(8, len(tail) // 2)):
            checks = min(p * 3, len(tail) - p)
            if checks > 0 and all(
                abs(tail[-(i+1)] - tail[-(i+1+p)]) < 1e-6
                for i in range(checks)
            ):
                return 4
    # Fixed point
    xfp, lam = _stable_fp(a, b, c)
    if xfp is not None and abs(lam) < 1.0:
        return 6 if abs(lam) < 0.5 else 5
    # Lyapunov
    lyap = _lyapunov(a, b, c)
    if abs(lyap) < 0.05:
        return 2
    return 3 if lyap > 0 else 5


# ─── OLS Fitter ───

def _fit(series):
    """Fit x_{n+1} = ax²_n + bx_n + c via ordinary least squares."""
    n = len(series) - 1
    if n < 3:
        return 0.0, 0.0, 0.0, float('inf')

    x = series[:-1]
    y = series[1:]

    # Normal equations for ax²+bx+c
    sx = sx2 = sx3 = sx4 = sy = sxy = sx2y = 0.0
    for i in range(n):
        xi, yi = x[i], y[i]
        xi2 = xi * xi
        sx += xi
        sx2 += xi2
        sx3 += xi2 * xi
        sx4 += xi2 * xi2
        sy += yi
        sxy += xi * yi
        sx2y += xi2 * yi

    # 3x3 Cramer's rule
    M = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]]

    def det3(m):
        return (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
              - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
              + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))

    D = det3(M)
    if abs(D) < 1e-20:
        return 0.0, 0.0, 0.0, float('inf')

    v = [sx2y, sxy, sy]

    def replace_col(col):
        return [
            [v[i] if j == col else M[i][j] for j in range(3)]
            for i in range(3)
        ]

    a = max(-10, min(10, det3(replace_col(0)) / D))
    b = max(-10, min(10, det3(replace_col(1)) / D))
    c = max(-1000, min(1000, det3(replace_col(2)) / D))

    # MSE
    mse = sum((y[i] - _eval(a, b, c, x[i]))**2 for i in range(n)) / n
    return a, b, c, mse


# ─── Public API ───

def classify(series, window=None):
    """
    Classify a time series.

    Args:
        series: list/tuple of numbers. At least 6 values.
        window: if set, use last `window` values only.

    Returns:
        Result with .band_name, .gap, .lyapunov, .entropy, etc.

    Example:
        >>> from coherence_router import classify
        >>> r = classify([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.8])
        >>> r.band_name
        'MOLECULAR'
    """
    s = list(series)
    if window and len(s) > window:
        s = s[-window:]
    if len(s) < 6:
        raise ValueError(f"Need at least 6 values, got {len(s)}")

    a, b, c, mse = _fit(s)
    band = _band(a, b, c)
    xfp, lam = _stable_fp(a, b, c)
    traj = _orbit(a, b, c, 0.5, 200)
    lyap = _lyapunov(a, b, c)
    H = _entropy(traj)

    if xfp is not None:
        energy = 0.5 * lam * lam + abs(_eval(a, b, c, xfp) - xfp)
    else:
        energy = float('inf')

    r = Result()
    r.a, r.b, r.c = a, b, c
    r.band = band
    r.band_name = BANDS[band][0]
    r.band_weight = BANDS[band][1]
    r.fixed_point = xfp
    r.eigenvalue = lam
    r.gap = max(0.0, 1.0 - abs(lam)) if lam is not None else 0.0
    r.lyapunov = lyap
    r.entropy = H
    r.energy = energy
    r.stable = lam is not None and abs(lam) < 1.0
    r.mse = mse
    return r


def classify_multi(series, window=20, stride=5):
    """
    Classify a long series as a sequence of operators.

    Slides a window across the series, classifies each chunk.
    Returns list of Results.

    Example:
        >>> results = classify_multi(my_long_series, window=20, stride=5)
        >>> [r.band_name for r in results]
        ['CRYSTAL', 'CRYSTAL', 'ORGANIC', 'MOLECULAR', 'CELLULAR', ...]
    """
    s = list(series)
    results = []
    for i in range(0, max(1, len(s) - window + 1), stride):
        chunk = s[i:i + window]
        if len(chunk) >= 6:
            results.append(classify(chunk))
    return results


def coherence(results):
    """
    Compute S* from a list of Results.

    S* = k/(1+k), k = σ·V*·A*
    V* = 1 - exp(-n/50)      (volume: more signals → more coverage)
    A* = mean(band_weight)    (alignment: higher bands → better)

    Args:
        results: list of Result objects from classify() or classify_multi()

    Returns:
        Coherence object with .S_star, .V_star, .A_star, .above_threshold
    """
    if not results:
        c = Coherence()
        c.S_star = c.V_star = c.A_star = c.k = 0.0
        c.n = 0
        c.above_threshold = False
        c.bands = {}
        c.results = []
        return c

    n = len(results)
    V = 1.0 - math.exp(-n / 50.0)
    A = sum(r.band_weight for r in results) / n
    k = SIGMA * V * A
    S = k / (1.0 + k)

    band_counts = defaultdict(int)
    for r in results:
        band_counts[r.band_name] += 1

    c = Coherence()
    c.S_star = S
    c.V_star = V
    c.A_star = A
    c.k = k
    c.n = n
    c.above_threshold = S >= T_STAR
    c.bands = dict(band_counts)
    c.results = results
    return c


def explain(result):
    """
    Human-readable physics report for a Result.

    Example:
        >>> print(explain(classify(my_series)))
    """
    r = result
    lines = [
        f"f(x) = {r.a:.4f}x² + {r.b:.4f}x + {r.c:.4f}",
        f"Band: {r.band_name} (weight={r.band_weight})",
        f"  {BANDS[r.band][2]}",
    ]
    if r.fixed_point is not None:
        lines.append(f"Fixed point: x* = {r.fixed_point:.6f}")
        lines.append(f"Eigenvalue:  λ  = {r.eigenvalue:.6f}  "
                     f"({'stable' if r.stable else 'unstable'})")
        lines.append(f"Spectral gap: g = {r.gap:.6f}")
    else:
        lines.append("No real fixed point")
    lines.append(f"Lyapunov:  λ_L = {r.lyapunov:.6f}  "
                 f"({'chaos' if r.lyapunov > 0.05 else 'stable' if r.lyapunov < -0.05 else 'marginal'})")
    lines.append(f"Entropy:   H   = {r.entropy:.4f} bits")
    if math.isfinite(r.energy):
        lines.append(f"Energy:    E   = {r.energy:.6f}")
    lines.append(f"Fit MSE:         {r.mse:.2e}")
    return "\n".join(lines)


def explain_coherence(coh):
    """Show the full derivation of S*."""
    lines = [
        f"n = {coh.n} signals",
        f"V* = 1 - exp(-{coh.n}/50) = {coh.V_star:.6f}",
        f"A* = mean(band_weights) = {coh.A_star:.6f}",
        f"k  = σ·V*·A* = {SIGMA}·{coh.V_star:.4f}·{coh.A_star:.4f} = {coh.k:.6f}",
        f"S* = k/(1+k) = {coh.S_star:.8f}",
        f"T* = {T_STAR:.6f}  →  {'ABOVE threshold ✓' if coh.above_threshold else 'below threshold'}",
        f"Bands: {coh.bands}",
    ]
    return "\n".join(lines)
