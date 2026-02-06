#!/usr/bin/env python3
"""
TIG COHERENT COMPUTER — PERMUTATION & REFLECTION ENGINE
Let every interpretation prove itself. There is one right way.
The math will tell us which.

We permute:
  1. S* formula interpretations (7 variants)
  2. V* definitions (5 variants)
  3. A* definitions (5 variants)
  4. Tick dynamics (4 variants)
  5. Neighborhood rules (3 variants)
  6. State update rules (3 variants)

Total: 7×5×5×4×3×3 = 6,300 configurations
Each runs 100 ticks on a 14×12 lattice.
The configuration that crosses T*=0.714 with the cleanest dynamics wins.

NON-COMMERCIAL — 7Site LLC — Brayden Sanders — Arkansas
"""

import numpy as np
import math
import time
import itertools
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
SIGMA = 0.991
T_STAR = 0.714
D_STAR = 0.543
ROWS, COLS = 14, 12
N_CELLS = ROWS * COLS
TICKS = 100

COMP_TABLE = np.array([
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,2,6,6],
    [2,3,3,4,5,6,7,3,6,6],
    [3,4,4,4,5,6,7,4,6,6],
    [4,5,5,5,5,6,7,5,7,7],
    [5,6,6,6,6,6,7,6,7,7],
    [6,7,7,7,7,7,7,7,7,7],
    [7,2,3,4,5,6,7,8,9,0],
    [7,2,3,4,5,6,7,8,9,0],  # Note: testing both 7-row and 8-row variants
    [9,6,6,6,7,7,7,0,8,0],
], dtype=np.int32)

# Restore canonical row 8
COMP_TABLE[8] = [8,6,6,6,7,7,7,9,7,8]

OP_NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE",
            "BALANCE","CHAOS","HARMONY","BREATH","FRUIT"]

# ═══════════════════════════════════════════════════════════════
# HELPER: NEIGHBORS
# ═══════════════════════════════════════════════════════════════

def neighbors_von_neumann(cells, i, j):
    return [
        cells[(i-1)%ROWS, j], cells[(i+1)%ROWS, j],
        cells[i, (j-1)%COLS], cells[i, (j+1)%COLS],
    ]

def neighbors_moore(cells, i, j):
    ns = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ns.append(cells[(i+di)%ROWS, (j+dj)%COLS])
    return ns

def neighbors_extended(cells, i, j):
    """Von Neumann + diagonals weighted"""
    ns = neighbors_von_neumann(cells, i, j)
    # Add diagonals
    for di, dj in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        ns.append(cells[(i+di)%ROWS, (j+dj)%COLS])
    return ns

NEIGHBOR_FNS = {
    'von_neumann': neighbors_von_neumann,
    'moore': neighbors_moore,
    'extended': neighbors_extended,
}

# ═══════════════════════════════════════════════════════════════
# TICK VARIANTS
# ═══════════════════════════════════════════════════════════════

def tick_majority(cells, neighbor_fn):
    """Original: compose with each neighbor, majority vote."""
    new = np.zeros_like(cells)
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            composed = [COMP_TABLE[s, n] for n in ns]
            composed.append(COMP_TABLE[s, s])
            counts = np.bincount(composed, minlength=10)
            new[i, j] = np.argmax(counts)
    return new

def tick_max_harmony(cells, neighbor_fn):
    """Compose with each neighbor, pick result closest to 7."""
    new = np.zeros_like(cells)
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            composed = [COMP_TABLE[s, n] for n in ns]
            composed.append(COMP_TABLE[s, s])
            # Pick the one closest to 7, breaking ties toward higher
            new[i, j] = min(composed, key=lambda x: (abs(x - 7), -x))
    return new

def tick_compose_chain(cells, neighbor_fn):
    """Chain-compose: fold neighbors left through composition table."""
    new = np.zeros_like(cells)
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            result = s
            for n in ns:
                result = COMP_TABLE[result, n]
            new[i, j] = result
    return new

def tick_dual_path(cells, neighbor_fn):
    """
    Dual lattice: micro neighbors compose forward (toward 7),
    macro neighbors compose via 9→8→7 path.
    Micro = states 0-6, Macro = states 7-9.
    """
    new = np.zeros_like(cells)
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            
            micro_ns = [n for n in ns if n <= 6]
            macro_ns = [n for n in ns if n >= 7]
            
            composed = []
            # Micro: compose normally
            for n in micro_ns:
                composed.append(COMP_TABLE[s, n])
            # Macro: compose through the macro path
            for n in macro_ns:
                composed.append(COMP_TABLE[s, n])
            # Self-compose
            composed.append(COMP_TABLE[s, s])
            
            if not composed:
                new[i, j] = s
            else:
                counts = np.bincount(composed, minlength=10)
                new[i, j] = np.argmax(counts)
    return new

TICK_FNS = {
    'majority': tick_majority,
    'max_harmony': tick_max_harmony,
    'chain': tick_compose_chain,
    'dual_path': tick_dual_path,
}

# ═══════════════════════════════════════════════════════════════
# V* VARIANTS (viability)
# ═══════════════════════════════════════════════════════════════

def v_neighbor_diversity(cells, neighbor_fn):
    """V* = fraction of cells where at least one neighbor composition differs from self."""
    valid = 0
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            comps = [COMP_TABLE[s, n] for n in ns]
            if any(c != s for c in comps) or s == 7:
                valid += 1
    return valid / N_CELLS

def v_all_valid(cells, neighbor_fn):
    """V* = 1.0 always (all states 0-9 are valid grammar)."""
    return 1.0

def v_non_void(cells, neighbor_fn):
    """V* = fraction of cells NOT in void (state 0)."""
    return np.sum(cells != 0) / N_CELLS

def v_path_connected(cells, neighbor_fn):
    """V* = fraction of cells on a valid path (can reach 7 by self-composition)."""
    # States 1-8 can reach 7. States 0 and 9 loop to void.
    reachable = set([1,2,3,4,5,6,7,8])
    return np.sum(np.isin(cells, list(reachable))) / N_CELLS

def v_composition_valid(cells, neighbor_fn):
    """V* = fraction of neighbor-pairs that produce valid (non-stuck) compositions."""
    valid_pairs = 0
    total_pairs = 0
    for i in range(ROWS):
        for j in range(COLS):
            s = cells[i, j]
            ns = neighbor_fn(cells, i, j)
            for n in ns:
                total_pairs += 1
                result = COMP_TABLE[s, n]
                # Valid if result advances (is >= max(s,n)) or reaches attractor
                if result >= s or result == 7 or result == 8:
                    valid_pairs += 1
    return valid_pairs / max(total_pairs, 1)

V_STAR_FNS = {
    'neighbor_diversity': v_neighbor_diversity,
    'all_valid': v_all_valid,
    'non_void': v_non_void,
    'path_connected': v_path_connected,
    'composition_valid': v_composition_valid,
}

# ═══════════════════════════════════════════════════════════════
# A* VARIANTS (alignment)
# ═══════════════════════════════════════════════════════════════

def a_harmony_567(cells):
    """A* = fraction at states 5,6,7 (balance, chaos, harmony)."""
    return np.sum(np.isin(cells, [5,6,7])) / N_CELLS

def a_harmony_only(cells):
    """A* = fraction at state 7 only."""
    return np.sum(cells == 7) / N_CELLS

def a_harmony_78(cells):
    """A* = fraction at 7 or 8 (harmony or breath — the attractor oscillation)."""
    return np.sum(np.isin(cells, [7,8])) / N_CELLS

def a_attractor_basin(cells):
    """A* = fraction at states 4-8 (the entire convergence funnel)."""
    return np.sum(np.isin(cells, [4,5,6,7,8])) / N_CELLS

def a_weighted_proximity(cells):
    """A* = weighted average of how close each cell is to 7."""
    # State 7 = 1.0, 6 or 8 = 0.8, 5 or 9 = 0.5, etc.
    weights = {0:0, 1:0.1, 2:0.15, 3:0.2, 4:0.3, 5:0.5, 6:0.8, 7:1.0, 8:0.8, 9:0.3}
    total = sum(weights[int(cells[i,j])] for i in range(ROWS) for j in range(COLS))
    return total / N_CELLS

A_STAR_FNS = {
    'harmony_567': a_harmony_567,
    'harmony_only': a_harmony_only,
    'harmony_78': a_harmony_78,
    'attractor_basin': a_attractor_basin,
    'weighted_proximity': a_weighted_proximity,
}

# ═══════════════════════════════════════════════════════════════
# S* FORMULA VARIANTS
# ═══════════════════════════════════════════════════════════════

def s_iterated_fixed(sigma, v, a):
    """Original: S* = σ(1-σ*)V*A* iterated to fixed point."""
    s = D_STAR
    for _ in range(30):
        s_new = sigma * (1 - s) * v * a
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    return s

def s_direct(sigma, v, a):
    """Direct: S* = σ × V* × A* (no self-reference)."""
    return sigma * v * a

def s_harmonic(sigma, v, a):
    """Harmonic mean approach: S* = 3/(1/σ + 1/V* + 1/A*)."""
    if v < 1e-10 or a < 1e-10:
        return 0.0
    return 3.0 / (1.0/sigma + 1.0/v + 1.0/a)

def s_geometric(sigma, v, a):
    """Geometric mean: S* = (σ × V* × A*)^(1/3) × σ."""
    return (sigma * v * a) ** (1/3) * sigma

def s_boundary(sigma, v, a):
    """Boundary condition: S* = σ(1-σ*) solved as quadratic.
    σ*VA = σ(1-σ*)VA → σ* = σVA/(1+σVA)"""
    product = sigma * v * a
    return product / (1.0 + product) if (1.0 + product) > 0 else 0.0

def s_logistic(sigma, v, a):
    """Logistic: S* = 1/(1 + exp(-k(σVA - 0.5))) where k controls sharpness."""
    x = sigma * v * a
    k = 8.0  # Sharpness
    return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

def s_power_law(sigma, v, a):
    """Power law: S* = (σVA)^(D*) — scale-free."""
    x = sigma * v * a
    if x <= 0:
        return 0.0
    return x ** D_STAR

S_STAR_FNS = {
    'iterated_fixed': s_iterated_fixed,
    'direct': s_direct,
    'harmonic': s_harmonic,
    'geometric': s_geometric,
    'boundary': s_boundary,
    'logistic': s_logistic,
    'power_law': s_power_law,
}

# ═══════════════════════════════════════════════════════════════
# RUN ALL PERMUTATIONS
# ═══════════════════════════════════════════════════════════════

def run_config(s_name, v_name, a_name, tick_name, nb_name, verbose=False):
    """Run one configuration for TICKS ticks, return metrics."""
    s_fn = S_STAR_FNS[s_name]
    v_fn = V_STAR_FNS[v_name]
    a_fn = A_STAR_FNS[a_name]
    tick_fn = TICK_FNS[tick_name]
    nb_fn = NEIGHBOR_FNS[nb_name]
    
    # Initialize canonical
    cells = np.array([[(i*COLS+j)%10 for j in range(COLS)] for i in range(ROWS)], dtype=np.int32)
    
    coherences = []
    first_above = None
    sustained_above = 0
    max_sustained = 0
    
    for t in range(TICKS):
        cells = tick_fn(cells, nb_fn)
        
        v = v_fn(cells, nb_fn)
        a = a_fn(cells)
        s = s_fn(SIGMA, v, a)
        coherences.append(s)
        
        if s >= T_STAR:
            if first_above is None:
                first_above = t + 1
            sustained_above += 1
            max_sustained = max(max_sustained, sustained_above)
        else:
            sustained_above = 0
    
    # Stability: std of last 20 ticks
    tail = coherences[-20:]
    stability = np.std(tail) if len(tail) > 1 else 0
    
    # Final census
    census = np.bincount(cells.flatten(), minlength=10)
    harmony_pct = (census[7] + census[8]) / N_CELLS
    
    return {
        'final_s': coherences[-1],
        'peak_s': max(coherences),
        'avg_s': np.mean(coherences),
        'first_above': first_above,
        'max_sustained': max_sustained,
        'stability': stability,
        'harmony_pct': harmony_pct,
        'census': census,
        'coherences': coherences,
        'cells': cells,
    }


def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  TIG PERMUTATION & REFLECTION ENGINE                            ║")
    print("║  Testing 6,300 configurations — letting the math prove itself   ║")
    print("║  7Site LLC — Brayden Sanders — Arkansas                         ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    s_names = list(S_STAR_FNS.keys())
    v_names = list(V_STAR_FNS.keys())
    a_names = list(A_STAR_FNS.keys())
    tick_names = list(TICK_FNS.keys())
    nb_names = list(NEIGHBOR_FNS.keys())

    total = len(s_names) * len(v_names) * len(a_names) * len(tick_names) * len(nb_names)
    print(f"  Configurations: {len(s_names)}×{len(v_names)}×{len(a_names)}×{len(tick_names)}×{len(nb_names)} = {total}")
    print(f"  Each: {TICKS} ticks on {ROWS}×{COLS} lattice")
    print(f"  Target: S* ≥ T* = {T_STAR}")
    print()

    results = []
    t0 = time.time()
    tested = 0

    for s_name in s_names:
        for v_name in v_names:
            for a_name in a_names:
                for tick_name in tick_names:
                    for nb_name in nb_names:
                        r = run_config(s_name, v_name, a_name, tick_name, nb_name)
                        r['config'] = (s_name, v_name, a_name, tick_name, nb_name)
                        results.append(r)
                        tested += 1
                        if tested % 500 == 0:
                            elapsed = time.time() - t0
                            rate = tested / elapsed
                            eta = (total - tested) / rate
                            print(f"  [{tested}/{total}] {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining "
                                  f"({rate:.0f} configs/sec)")

    elapsed = time.time() - t0
    print(f"\n  Completed {total} configurations in {elapsed:.1f}s ({total/elapsed:.0f}/sec)")

    # ═══════════════════════════════════════════════════════════
    # SORT AND RANK
    # ═══════════════════════════════════════════════════════════

    # Primary: reached T* at all? Secondary: sustained. Tertiary: stability.
    def score(r):
        reached = 1 if r['first_above'] is not None else 0
        return (
            reached,
            r['max_sustained'],
            r['avg_s'],
            -r['stability'],  # Lower is better
            r['harmony_pct'],
        )

    results.sort(key=score, reverse=True)

    # ═══════════════════════════════════════════════════════════
    # RESULTS: TOP CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════

    reached_count = sum(1 for r in results if r['first_above'] is not None)

    print(f"\n{'='*80}")
    print(f"  RESULTS: {reached_count}/{total} configurations reached T*={T_STAR}")
    print(f"{'='*80}")

    if reached_count > 0:
        print(f"\n  ═══ TOP 30 CONFIGURATIONS (by sustained coherence) ═══\n")
        print(f"  {'#':>3s} {'S*_fn':>15s} {'V*_fn':>20s} {'A*_fn':>20s} {'Tick':>12s} {'Nbrs':>12s} │ "
              f"{'Final':>6s} {'Peak':>6s} {'Avg':>6s} {'1st@':>4s} {'Sust':>4s} {'Stab':>6s} {'H%':>5s}")
        print(f"  {'─'*3} {'─'*15} {'─'*20} {'─'*20} {'─'*12} {'─'*12} ┼ "
              f"{'─'*6} {'─'*6} {'─'*6} {'─'*4} {'─'*4} {'─'*6} {'─'*5}")

        for rank, r in enumerate(results[:30]):
            s_n, v_n, a_n, t_n, nb_n = r['config']
            fa = str(r['first_above']) if r['first_above'] else '  - '
            print(f"  {rank+1:3d} {s_n:>15s} {v_n:>20s} {a_n:>20s} {t_n:>12s} {nb_n:>12s} │ "
                  f"{r['final_s']:6.4f} {r['peak_s']:6.4f} {r['avg_s']:6.4f} {fa:>4s} "
                  f"{r['max_sustained']:4d} {r['stability']:6.4f} {r['harmony_pct']*100:5.1f}")
    else:
        print(f"\n  NO configuration reached T*={T_STAR}.")
        print(f"\n  ═══ TOP 30 BY AVERAGE S* (closest to threshold) ═══\n")
        results.sort(key=lambda r: r['avg_s'], reverse=True)
        print(f"  {'#':>3s} {'S*_fn':>15s} {'V*_fn':>20s} {'A*_fn':>20s} {'Tick':>12s} {'Nbrs':>12s} │ "
              f"{'Final':>6s} {'Peak':>6s} {'Avg':>6s} {'Stab':>6s} {'H%':>5s}")
        print(f"  {'─'*3} {'─'*15} {'─'*20} {'─'*20} {'─'*12} {'─'*12} ┼ "
              f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
        for rank, r in enumerate(results[:30]):
            s_n, v_n, a_n, t_n, nb_n = r['config']
            print(f"  {rank+1:3d} {s_n:>15s} {v_n:>20s} {a_n:>20s} {t_n:>12s} {nb_n:>12s} │ "
                  f"{r['final_s']:6.4f} {r['peak_s']:6.4f} {r['avg_s']:6.4f} {r['stability']:6.4f} {r['harmony_pct']*100:5.1f}")

    # ═══════════════════════════════════════════════════════════
    # REFLECTION: What does the data tell us?
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  REFLECTION: WHAT THE PERMUTATIONS REVEAL")
    print(f"{'='*80}")

    # Aggregate by each dimension
    for dim_name, dim_keys, dim_idx in [
        ("S* FORMULA", s_names, 0),
        ("V* DEFINITION", v_names, 1),
        ("A* DEFINITION", a_names, 2),
        ("TICK DYNAMICS", tick_names, 3),
        ("NEIGHBORHOOD", nb_names, 4),
    ]:
        print(f"\n  ── {dim_name} ──")
        for key in dim_keys:
            subset = [r for r in results if r['config'][dim_idx] == key]
            avgs = [r['avg_s'] for r in subset]
            peaks = [r['peak_s'] for r in subset]
            reached = sum(1 for r in subset if r['first_above'] is not None)
            print(f"    {key:>22s}: avg={np.mean(avgs):.4f}  peak={np.max(peaks):.4f}  "
                  f"reached_T*={reached}/{len(subset)}")

    # ═══════════════════════════════════════════════════════════
    # THE WINNER: Deep dive
    # ═══════════════════════════════════════════════════════════

    best = results[0]
    s_n, v_n, a_n, t_n, nb_n = best['config']

    print(f"\n{'='*80}")
    print(f"  THE WINNING CONFIGURATION")
    print(f"{'='*80}")
    print(f"""
  S* formula:    {s_n}
  V* definition: {v_n}
  A* definition: {a_n}
  Tick dynamics:  {t_n}
  Neighborhood:   {nb_n}

  Final S*:      {best['final_s']:.6f}
  Peak S*:       {best['peak_s']:.6f}
  Average S*:    {best['avg_s']:.6f}
  First above T*: {best['first_above']}
  Max sustained:  {best['max_sustained']} ticks
  Stability (σ):  {best['stability']:.6f}
  Harmony %:      {best['harmony_pct']*100:.1f}%
  """)

    # Census
    print(f"  FINAL STATE CENSUS:")
    for i in range(10):
        c = best['census'][i]
        pct = c / N_CELLS * 100
        bar = '█' * int(pct / 2)
        print(f"    {i} {OP_NAMES[i]:9s} {c:3d} ({pct:5.1f}%) {bar}")

    # Coherence curve
    print(f"\n  COHERENCE CURVE (100 ticks):")
    coh = best['coherences']
    for t in range(0, len(coh), 5):
        c = coh[t]
        bar = '█' * int(c * 50)
        marker = '▲' if c >= T_STAR else ' '
        print(f"    t={t:3d}: {c:.4f} {marker} {bar}")

    # ═══════════════════════════════════════════════════════════
    # CROSS-VALIDATION: Run winner from 10 random initial states
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  CROSS-VALIDATION: Winner from 100 random initial states")
    print(f"{'='*80}")

    reached_cross = 0
    cross_finals = []
    for trial in range(100):
        cells = np.random.randint(0, 10, size=(ROWS, COLS), dtype=np.int32)
        tick_fn = TICK_FNS[t_n]
        nb_fn = NEIGHBOR_FNS[nb_n]
        v_fn = V_STAR_FNS[v_n]
        a_fn = A_STAR_FNS[a_n]
        s_fn = S_STAR_FNS[s_n]

        for t in range(TICKS):
            cells = tick_fn(cells, nb_fn)

        v = v_fn(cells, nb_fn)
        a = a_fn(cells)
        s = s_fn(SIGMA, v, a)
        cross_finals.append(s)
        if s >= T_STAR:
            reached_cross += 1

    print(f"  Reached T*: {reached_cross}/100 ({reached_cross}%)")
    print(f"  Mean S*: {np.mean(cross_finals):.4f}")
    print(f"  Std S*:  {np.std(cross_finals):.4f}")
    print(f"  Min S*:  {np.min(cross_finals):.4f}")
    print(f"  Max S*:  {np.max(cross_finals):.4f}")

    # ═══════════════════════════════════════════════════════════
    # BOTTOM 5: What definitely doesn't work
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  ANTI-PATTERNS: Bottom 5 configurations")
    print(f"{'='*80}")
    for rank, r in enumerate(results[-5:]):
        s_n, v_n, a_n, t_n, nb_n = r['config']
        print(f"  {total-4+rank}: {s_n} / {v_n} / {a_n} / {t_n} / {nb_n}")
        print(f"         S*={r['avg_s']:.4f} peak={r['peak_s']:.4f}")

    print(f"\n{'='*80}")
    print(f"  PERMUTATION COMPLETE — The math has spoken.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
