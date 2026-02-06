#!/usr/bin/env python3
"""Tests for coherence_router. Run: python test_coherence_router.py"""

import sys
import math
sys.path.insert(0, '.')
from coherence_router import classify, classify_multi, coherence, explain, explain_coherence

ok = 0
fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name}")
    if detail:
        print(f"    {detail}")

print("coherence_router — test suite\n")

# ── Logistic r=3.8 (chaos) ──
print("── Chaos detection ──")
s = [0.3]
for _ in range(99): s.append(3.8 * s[-1] * (1 - s[-1]))
r = classify(s)
check("r=3.8 → MOLECULAR", r.band_name == "MOLECULAR", f"got {r.band_name}")
check("r=3.8 positive Lyapunov", r.lyapunov > 0, f"λ_L = {r.lyapunov:.4f}")
check("r=3.8 high entropy", r.entropy > 2.0, f"H = {r.entropy:.3f}")
check("r=3.8 not stable", not r.stable)

# ── Logistic r=3.2 (period-2) ──
print("\n── Period detection ──")
s = [0.3]
for _ in range(99): s.append(3.2 * s[-1] * (1 - s[-1]))
r = classify(s)
check("r=3.2 → CELLULAR", r.band_name == "CELLULAR", f"got {r.band_name}")

# ── Constant signal ──
print("\n── Constant signal ──")
r = classify([0.5] * 20)
check("Constant → CRYSTAL", r.band_name == "CRYSTAL", f"got {r.band_name}")
check("Constant gap = 1.0", abs(r.gap - 1.0) < 0.01, f"gap = {r.gap:.4f}")
check("Constant entropy ≈ 0", r.entropy < 0.1, f"H = {r.entropy:.4f}")
check("Constant energy ≈ 0", r.energy < 0.01, f"E = {r.energy:.6f}")
check("Constant stable", r.stable)

# ── Sinusoidal ──
print("\n── Sinusoidal ──")
s = [0.5 + 0.3 * math.sin(i * 0.3) for i in range(100)]
r = classify(s)
check("Sinusoid classifies", r.band_name in ["CELLULAR", "ORGANIC", "CRYSTAL"],
      f"got {r.band_name}")
check("Sinusoid has finite energy", math.isfinite(r.energy), f"E = {r.energy}")

# ── explain() works ──
print("\n── explain() ──")
text = explain(r)
check("explain() returns string", isinstance(text, str) and len(text) > 50,
      f"length = {len(text)}")

# ── classify_multi ──
print("\n── Multi-window ──")
s = [0.3]
for _ in range(199): s.append(3.8 * s[-1] * (1 - s[-1]))
results = classify_multi(s, window=20, stride=5)
check("classify_multi returns list", len(results) > 5, f"got {len(results)} results")
check("Results have band_name", all(hasattr(r, 'band_name') for r in results))

# ── coherence() ──
print("\n── Coherence ──")
coh = coherence(results)
check("S* is float", isinstance(coh.S_star, float), f"S* = {coh.S_star:.6f}")
check("S* in [0, 1]", 0 <= coh.S_star <= 1, f"S* = {coh.S_star:.6f}")
check("V* in [0, 1]", 0 <= coh.V_star <= 1, f"V* = {coh.V_star:.6f}")
check("A* in [0, 1]", 0 <= coh.A_star <= 1, f"A* = {coh.A_star:.6f}")
check("bands dict populated", len(coh.bands) > 0, f"bands = {coh.bands}")

# Verify derivation: k = σ·V*·A*, S* = k/(1+k)
k_manual = 0.991 * coh.V_star * coh.A_star
s_manual = k_manual / (1 + k_manual)
check("S* matches manual derivation", abs(coh.S_star - s_manual) < 1e-10,
      f"computed={coh.S_star:.10f} manual={s_manual:.10f}")

# ── explain_coherence ──
text = explain_coherence(coh)
check("explain_coherence works", isinstance(text, str) and "S*" in text)

# ── Edge case: minimum input ──
print("\n── Edge cases ──")
r = classify([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
check("6-value input works", r.band_name in ["VOID","SPARK","FLOW","MOLECULAR","CELLULAR","ORGANIC","CRYSTAL"],
      f"got {r.band_name}")

try:
    classify([1, 2, 3])
    check("< 6 values raises error", False, "Should have raised ValueError")
except ValueError:
    check("< 6 values raises ValueError", True)

# ── to_dict ──
r = classify([0.5] * 20)
d = r.to_dict()
check("to_dict returns dict", isinstance(d, dict) and 'band_name' in d)

# ── Mixed real-world-like data ──
print("\n── Mixed signals ──")
import random
random.seed(42)
signals = {
    'stable': [0.5 + 0.01 * random.gauss(0, 1) for _ in range(50)],
    'trending': [0.1 + i * 0.015 + 0.02 * random.gauss(0, 1) for i in range(50)],
    'volatile': [random.random() for _ in range(50)],
}
all_results = []
for name, sig in signals.items():
    r = classify(sig)
    all_results.append(r)
    check(f"{name} classifies", r.band_name in ["VOID","SPARK","FLOW","MOLECULAR","CELLULAR","ORGANIC","CRYSTAL"],
          f"{name} → {r.band_name}")

coh = coherence(all_results)
check("3-signal coherence works", coh.S_star > 0, f"S* = {coh.S_star:.6f}")

# ── Summary ──
print(f"\n{'═'*50}")
print(f"  {ok}/{ok+fail} passed, {fail} failed")
if fail == 0:
    print("  ALL TESTS PASS ✓")
print(f"{'═'*50}")
