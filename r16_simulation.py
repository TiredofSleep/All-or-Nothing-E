#!/usr/bin/env python3
"""
TIG COHERENT COMPUTER — R16 DEPLOYMENT SIMULATION
Council of Lattices • Self-Composition • Adversarial • Scaling
© 2024-2026 Brayden Sanders / 7Site LLC — Arkansas
"""

import numpy as np
import math
import time
from collections import Counter

SIGMA = 0.991
T_STAR = 0.714
D_STAR = 0.543

COMP_TABLE = np.array([
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,2,6,6],
    [2,3,3,4,5,6,7,3,6,6],
    [3,4,4,4,5,6,7,4,6,6],
    [4,5,5,5,5,6,7,5,7,7],
    [5,6,6,6,6,6,7,6,7,7],
    [6,7,7,7,7,7,7,7,7,7],
    [7,2,3,4,5,6,7,8,9,0],
    [8,6,6,6,7,7,7,9,7,8],
    [9,6,6,6,7,7,7,0,8,0],
], dtype=np.int32)

OP_NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE",
            "BALANCE","CHAOS","HARMONY","BREATH","FRUIT"]

# ═══════════════════════════════════════════════════════════════
# FAST LATTICE
# ═══════════════════════════════════════════════════════════════

class FastLattice:
    def __init__(self, rows, cols, lid=0):
        self.rows, self.cols, self.id = rows, cols, lid
        self.cells = np.zeros((rows,cols), dtype=np.int32)
        self.tick_count = 0
        self._history = []

    def init_canonical(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.cells[i,j] = (i*self.cols+j) % 10
        self.tick_count = 0; self._history = []

    def init_random(self):
        self.cells = np.random.randint(0,10,size=(self.rows,self.cols),dtype=np.int32)
        self.tick_count = 0; self._history = []

    @property
    def n_cells(self): return self.rows * self.cols

    def tick(self):
        padded = np.pad(self.cells, 1, mode='wrap')
        R, C = self.rows, self.cols
        shifts = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if di==0 and dj==0: continue
                shifts.append(padded[1+di:R+1+di, 1+dj:C+1+dj])
        new = np.zeros((R,C), dtype=np.int32)
        for i in range(R):
            for j in range(C):
                s = self.cells[i,j]
                votes = [int(COMP_TABLE[s, int(sh[i,j])]) for sh in shifts]
                votes.append(int(COMP_TABLE[s,s]))
                counts = np.bincount(votes, minlength=10)
                new[i,j] = np.argmax(counts)
        self.cells = new
        self.tick_count += 1
        coh = self.coherence()
        self._history.append(coh)
        return coh

    def coherence(self):
        n = self.n_cells
        padded = np.pad(self.cells, 1, mode='wrap')
        valid = 0
        for i in range(self.rows):
            for j in range(self.cols):
                s = self.cells[i,j]
                trivial = True
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        if di==0 and dj==0: continue
                        if COMP_TABLE[s, padded[i+1+di,j+1+dj]] != s:
                            trivial = False; break
                    if not trivial: break
                if not trivial or s==7: valid+=1
        v = valid/n
        a = np.sum((self.cells>=4)&(self.cells<=8))/n
        if v<1e-10 or a<1e-10: return 0.0
        return 3.0/(1.0/SIGMA + 1.0/v + 1.0/a)

    def census(self): return np.bincount(self.cells.flatten(), minlength=10)
    def output_row(self, r=-1): return self.cells[r].copy()
    def inject_row(self, r, states):
        L = min(len(states), self.cols)
        self.cells[r,:L] = states[:L]

# ═══════════════════════════════════════════════════════════════
# COUNCIL
# ═══════════════════════════════════════════════════════════════

class Council:
    def __init__(self, n, rows, cols):
        self.members = [FastLattice(rows, cols, lid=i) for i in range(n)]
        self.n = n
        self.tick_count = 0
        self._history = []

    def init_diverse(self):
        seeds = [[0,1,2],[0,7,1],[1,2,3],[7,8,9],[3,4,5],[5,6,7],
                 [2,3,4],[6,7,8],[1,3,5],[2,4,6],[0,5,9],[1,6,8],
                 [3,7,0],[4,8,1],[5,9,2],[0,0,7]]
        for i, m in enumerate(self.members):
            m.init_canonical()
            s = seeds[i % len(seeds)]
            m.inject_row(0, np.array((s * (m.cols//len(s)+1))[:m.cols], dtype=np.int32))

    def init_canonical(self):
        for m in self.members: m.init_canonical()

    def init_random(self):
        for m in self.members: m.init_random()

    def tick(self):
        member_cohs = [m.tick() for m in self.members]
        # Ring exchange
        if self.n > 1:
            boundaries = [m.output_row(-1) for m in self.members]
            for i in range(self.n):
                recv = (i+1)%self.n
                top = self.members[recv].cells[0].copy()
                exchanged = np.array([
                    COMP_TABLE[boundaries[i][j], top[j]]
                    for j in range(len(top))
                ], dtype=np.int32)
                self.members[recv].inject_row(0, exchanged)
        self.tick_count += 1
        cc = self.council_coherence(member_cohs)
        self._history.append(cc)
        return cc, member_cohs

    def council_coherence(self, mc=None):
        if mc is None: mc = [m.coherence() for m in self.members]
        valid = [c for c in mc if c > 1e-10]
        if not valid: return 0.0
        return len(valid) / sum(1.0/c for c in valid)

    def consensus(self):
        censuses = [m.census()/m.n_cells for m in self.members]
        if len(censuses) < 2: return 1.0
        total = 0.0; pairs = 0
        for i in range(len(censuses)):
            for j in range(i+1,len(censuses)):
                d = np.dot(censuses[i],censuses[j])
                ni = np.linalg.norm(censuses[i]); nj = np.linalg.norm(censuses[j])
                if ni>0 and nj>0: total += d/(ni*nj)
                pairs += 1
        return total/max(pairs,1)

    def inject_noise(self, frac, targets=None):
        tgts = targets if targets else range(self.n)
        for idx in tgts:
            m = self.members[idx]
            nc = int(m.n_cells * frac)
            for _ in range(nc):
                m.cells[np.random.randint(m.rows), np.random.randint(m.cols)] = np.random.randint(0,10)

# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def test_self_composition():
    print(f"\n{'='*72}")
    print(f"  TEST 1: SELF-COMPOSITION (18×14, 50 ticks)")
    print(f"  Output row feeds back as input each tick.")
    print(f"{'='*72}\n")
    lat = FastLattice(18,14)
    lat.init_canonical()
    cohs = []
    for t in range(50):
        coh = lat.tick()
        cohs.append(coh)
        lat.inject_row(0, lat.output_row(-1))
        if t%5==0 or t==49:
            c = lat.census()
            h78 = (c[7]+c[8])/lat.n_cells*100
            bar = "█"*int(coh*40)
            print(f"    t={t:3d}: S*={coh:.4f} H78={h78:5.1f}% {'▲' if coh>=T_STAR else ' '} {bar}")
    above = sum(1 for c in cohs if c>=T_STAR)
    print(f"\n  Above T*: {above}/50 ({above*2}%)")
    stable = np.std(cohs[-10:])
    print(f"  Tail stability (σ last 10): {stable:.6f}")
    return cohs

def test_council_16():
    print(f"\n{'='*72}")
    print(f"  TEST 2: R16 COUNCIL (16 lattices, diverse seeds, 50 ticks)")
    print(f"  Ring topology boundary exchange.")
    print(f"{'='*72}\n")
    council = Council(16, 14, 12)
    council.init_diverse()
    for t in range(50):
        cc, mc = council.tick()
        if t%5==0 or t==49:
            con = council.consensus()
            print(f"    t={t:3d}: Council S*={cc:.4f}  Consensus={con:.4f}  "
                  f"Range=[{min(mc):.3f},{max(mc):.3f}] {'▲' if cc>=T_STAR else '▽'}")
    print(f"\n  Per-core final:")
    for i,m in enumerate(council.members):
        c = m.coherence(); cs = m.census()
        h78 = (cs[7]+cs[8])/m.n_cells*100
        print(f"    Core {i:2d}: S*={c:.4f} H78={h78:5.1f}% {'▲' if c>=T_STAR else '▽'}")
    return council

def test_self_feed_council():
    print(f"\n{'='*72}")
    print(f"  TEST 3: COUNCIL + SELF-FEED (16 lattices, double loop)")
    print(f"  Each lattice feeds output to self AND receives from neighbor.")
    print(f"{'='*72}\n")
    council = Council(16, 14, 12)
    council.init_diverse()
    for t in range(50):
        cc, mc = council.tick()
        for m in council.members:
            m.inject_row(0, m.output_row(-1))
        if t%10==0 or t==49:
            con = council.consensus()
            print(f"    t={t:3d}: Council S*={cc:.4f}  Consensus={con:.4f}  {'▲' if cc>=T_STAR else '▽'}")
    return council

def test_adversarial():
    print(f"\n{'='*72}")
    print(f"  TEST 4: ADVERSARIAL COUNCIL (16 lattices, 80 ticks)")
    print(f"  t=20: 50% noise → 8 members  |  t=40: 80% noise → core 0")
    print(f"  t=60+: Recovery — does the council heal?")
    print(f"{'='*72}\n")
    council = Council(16, 14, 12)
    council.init_canonical()
    for t in range(80):
        if t==20:
            print(f"  ──── t=20: INJECT 50% NOISE → CORES 0-7 ────")
            council.inject_noise(0.5, list(range(8)))
        if t==40:
            print(f"  ──── t=40: INJECT 80% NOISE → CORE 0 (ROGUE) ────")
            council.inject_noise(0.8, [0])
        cc, mc = council.tick()
        if t%5==0 or t==79 or t in(20,21,40,41,60,61):
            phase = "BASE" if t<20 else "DMGD" if t<40 else "ROGUE" if t<60 else "HEAL"
            bar = "█"*int(cc*40)
            print(f"    t={t:3d} [{phase:5s}]: S*={cc:.4f}  "
                  f"Con={council.consensus():.4f}  {'▲' if cc>=T_STAR else '▽'}  {bar}")
    print(f"\n  Core 0 (rogue target) final S*: {council.members[0].coherence():.4f}")
    print(f"  Council final S*: {council.council_coherence():.4f}")
    return council

def test_council_vs_council():
    print(f"\n{'='*72}")
    print(f"  TEST 5: COUNCIL vs COUNCIL (8v8, 50 ticks)")
    print(f"  Canonical seed vs random seed. Cross-boundary exchange.")
    print(f"{'='*72}\n")
    ca = Council(8, 14, 12); ca.init_canonical()
    cb = Council(8, 14, 12); cb.init_random()
    for t in range(50):
        sa, _ = ca.tick(); sb, _ = cb.tick()
        # Cross exchange
        a_out = ca.members[0].output_row(-1)
        b_top = cb.members[0].cells[0].copy()
        cb.members[0].inject_row(0, np.array([COMP_TABLE[a_out[j],b_top[j]] for j in range(len(b_top))], dtype=np.int32))
        b_out = cb.members[-1].output_row(-1)
        a_top = ca.members[-1].cells[0].copy()
        ca.members[-1].inject_row(0, np.array([COMP_TABLE[b_out[j],a_top[j]] for j in range(len(a_top))], dtype=np.int32))
        if t%5==0 or t==49:
            w = "A" if sa>sb else "B" if sb>sa else "="
            print(f"    t={t:3d}: A={sa:.4f} vs B={sb:.4f}  [{w}]")
    fa = ca.council_coherence(); fb = cb.council_coherence()
    print(f"\n  Final: A(canonical)={fa:.4f} vs B(random)={fb:.4f}")
    print(f"  {'BOTH CONVERGE' if fa>=T_STAR and fb>=T_STAR else 'Canonical WINS' if fa>fb else 'Random WINS'}")

def test_scaling():
    print(f"\n{'='*72}")
    print(f"  TEST 6: SCALING LADDER (1→2→4→8→16 lattices, 30 ticks)")
    print(f"{'='*72}\n")
    for n in [1,2,4,8,16]:
        council = Council(n, 14, 12)
        council.init_diverse()
        t0 = time.time()
        for _ in range(30): council.tick()
        elapsed = time.time()-t0
        coh = council.council_coherence()
        con = council.consensus()
        cells = n*14*12
        ops = cells*30/max(elapsed,.001)
        print(f"    n={n:2d} ({cells:5d} cells): S*={coh:.4f}  Con={con:.4f}  "
              f"{elapsed:.2f}s  {ops:,.0f} ops/s {'▲' if coh>=T_STAR else '▽'}")

def test_large_lattice():
    print(f"\n{'='*72}")
    print(f"  TEST 7: LARGE LATTICE (64×48 = 3,072 cells, 15 ticks)")
    print(f"  R16 target payload size.")
    print(f"{'='*72}\n")
    lat = FastLattice(64, 48)
    lat.init_canonical()
    t0 = time.time()
    for t in range(15):
        coh = lat.tick()
        if t%3==0 or t==14:
            c = lat.census()
            h78 = (c[7]+c[8])/lat.n_cells*100
            el = time.time()-t0
            print(f"    t={t:3d}: S*={coh:.4f} H78={h78:5.1f}% {el:.1f}s {'▲' if coh>=T_STAR else '▽'}")
    total = time.time()-t0
    print(f"\n  {lat.n_cells:,} cells × 15 ticks in {total:.1f}s = {lat.n_cells*15/total:,.0f} cell-ops/s")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║  TIG COHERENT COMPUTER — R16 DEPLOYMENT SIMULATION                  ║")
    print("║  Self-Composition • 16-Core Council • Adversarial • Scaling          ║")
    print("║  © 2024-2026 Brayden Sanders / 7Site LLC — Hot Springs, Arkansas     ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    t_total = time.time()

    test_self_composition()
    test_council_16()
    test_self_feed_council()
    test_adversarial()
    test_council_vs_council()
    test_scaling()
    test_large_lattice()

    elapsed = time.time() - t_total
    print(f"\n{'='*72}")
    print(f"  R16 DEPLOYMENT SIMULATION COMPLETE — {elapsed:.1f}s total")
    print(f"{'='*72}")

if __name__ == '__main__':
    main()
