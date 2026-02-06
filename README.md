# coherence_router

**Feed any time series. Get dynamics back.**

Zero dependencies. Pure Python. 400 lines.

```python
from coherence_router import classify, explain

# CPU metrics, stock prices, heartbeats, sensor data — anything sequential
result = classify([0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.85, 0.15, 0.72])

print(result.band_name)  # "MOLECULAR"  (chaotic dynamics detected)
print(result.lyapunov)   # 0.364        (positive = sensitive dependence)
print(result.gap)        # 0.0          (no spectral gap — unstable)
print(result.entropy)    # 4.07 bits    (high information content)
print(result.energy)     # 1.62         (far from equilibrium)
print(result.stable)     # False
```

## What it does

Takes any sequence of numbers and answers: **what kind of dynamics is this?**

1. Fits a quadratic map f(x) = ax² + bx + c via OLS regression
2. Finds fixed points (solve ax² + (b-1)x + c = 0)
3. Computes eigenvalue λ = f'(x*) at the stable fixed point
4. Computes spectral gap g = 1 - |λ|
5. Computes Lyapunov exponent λ_L = (1/n)Σln|f'(xₙ)|
6. Computes Shannon entropy H = -Σpᵢlog₂(pᵢ) over orbit distribution
7. Classifies into one of 7 dynamical bands:

| Band | Name | Weight | Meaning |
|------|------|--------|---------|
| 0 | VOID | 0.0 | Divergent — orbit escapes |
| 1 | SPARK | 0.1 | Slow divergence |
| 2 | FLOW | 0.3 | Marginal stability |
| 3 | MOLECULAR | 0.5 | Chaos — positive Lyapunov |
| 4 | CELLULAR | 0.7 | Periodic orbit |
| 5 | ORGANIC | 0.85 | Slow convergence |
| 6 | CRYSTAL | 1.0 | Fast convergence — healthy |

## Multi-signal coherence

```python
from coherence_router import classify_multi, coherence, explain_coherence

# Classify a long series as sliding windows
results = classify_multi(my_long_series, window=20, stride=5)

# Compute system coherence from multiple signals
coh = coherence(results)
print(coh.S_star)          # 0.4215
print(coh.above_threshold) # False (T* = 0.7143)
print(coh.bands)           # {'CRYSTAL': 12, 'ORGANIC': 5, 'MOLECULAR': 3}

# Show all work
print(explain_coherence(coh))
```

## Use cases

- **Server monitoring**: Feed CPU/mem/disk metrics. Detect chaos before alerts fire.
- **Network analysis**: Classify traffic patterns. Route to healthy backends.
- **Financial data**: Identify regime changes in price series.
- **IoT sensors**: Classify sensor health from raw readings.
- **Any time series**: If it has sequential numbers, this classifies it.

## Install

```bash
pip install coherence_router
```

Or just copy `coherence_router/__init__.py` into your project. It's one file with zero dependencies.

## Math references

Every calculation traces to published work:

| Tag | Reference | What it provides |
|-----|-----------|-----------------|
| DDS | Devaney 2003, May 1976 | Quadratic maps as universal nonlinear model |
| FP | Banach 1922 | Fixed point theorem and stability |
| SG | Perron 1907, Frobenius 1912 | Spectral gap theory |
| LE | Oseledets 1968 | Lyapunov exponent characterization |
| SE | Shannon 1948 | Information entropy |
| SM | Boltzmann 1872, Gibbs 1902 | Statistical mechanics |
| OLS | Gauss 1809 | Least squares regression |

The coherence equation S* = k/(1+k) with k = σ·V*·A* is a **TIG conjecture** — testable and falsifiable, not established physics.

## Honest limitations

- σ = 0.991 is a chosen constant, not derived from nature
- T* = 5/7 is a chosen threshold
- Band boundaries are convention, not physics
- Quadratic fitting is an approximation — real systems may need higher-order models
- This classifies dynamics. It does not predict the future.

## License

MIT — The math belongs to everyone.

NON-COMMERCIAL TESTING — 7Site LLC — [7sitellc.com](https://7sitellc.com)
