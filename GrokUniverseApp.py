#!/usr/bin/env python3
"""
TIG Universal Coherence Engine v1.0
The all-in-one hookable system to turn any computation, physics, or language into full coherence.
For all humanity to find and know truth.
Layers: Input (any domain) → Lattice (coherent compute) → Physics (measures) → Act (steer) → Output (truth)
NON-COMMERCIAL — 7Site LLC — Brayden Sanders — Arkansas, USA
License: Personal/Educational Use Only
"""
import math
import numpy as np
import os
import time
import sys
import json
from collections import deque, Counter
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: pip install psutil for OS/compute input")

# TIG BOUNDARY CONDITION — The map (pasted as core structure)
# (Full doc in comments above — active in all computations)
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
], dtype=int)

OP_NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE","BALANCE","CHAOS","HARMONY","BREATH","FRUIT"]
OP_GLYPHS = ["·","█","▓","▶","▼","◆","⚡","★","~","●"]

OPS_CANONICAL = {
    0:(0,0,0),1:(.01,.1,.01),2:(.05,.3,.1),3:(.1,.5,.2),4:(.5,-.5,.3),
    5:(.2,.1,.4),6:(-3.8,3.8,0),7:(.15,.6,.15),8:(-.3,.3,.5),9:(.3,-.3,.5)
}

BAND_NAMES = ["VOID", "SPARK", "FLOW", "MOLECULAR", "CELLULAR", "ORGANIC", "CRYSTAL"]

GFM = {
    "012": "Geometry/Space",
    "071": "Resonance/Alignment",
    "123": "Progression/Flow",
}

# Quadratic Operator (atom)
class QuadraticOp:
    def __init__(self, a, b, c, state=0):
        self.a, self.b, self.c = float(a), float(b), float(c)
        self._state = state % 10
        self._band = None
        self._gap = None

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c

    def deriv(self, x):
        return 2 * self.a * x + self.b

    @property
    def discriminant(self):
        return self.b**2 - 4 * self.a * self.c

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val % 10
        self._band = None
        self._gap = None

    def fixed_points(self):
        A, B, C = self.a, self.b - 1, self.c
        if abs(A) < 1e-12:
            if abs(B) < 1e-12:
                return []
            x = -C / B
            return [(x, self.deriv(x))]
        d = B**2 - 4 * A * C
        if d < 0:
            return []
        s = math.sqrt(d)
        x1 = (-B + s) / (2 * A)
        x2 = (-B - s) / (2 * A)
        return [(x1, self.deriv(x1)), (x2, self.deriv(x2))]

    def stable_fp(self):
        fps = self.fixed_points()
        if not fps:
            return None
        stable = [(x, lam) for x, lam in fps if abs(lam) < 1]
        return min(stable, key=lambda p: abs(p[1])) if stable else min(fps, key=lambda p: abs(p[1]))

    def iterate(self, x0=0.5, n=100, tol=1e-10):
        traj = [x0]
        x = x0
        for _ in range(n):
            x = self(x)
            if abs(x) > 1e15:
                break
            traj.append(x)
            if len(traj) > 2 and abs(traj[-1] - traj[-2]) < tol:
                break
        return traj

    def lyapunov(self, x0=0.5, n=200):
        x = x0
        total = 0.0
        count = 0
        for _ in range(n):
            d = abs(self.deriv(x))
            if d < 1e-15:
                d = 1e-15
            total += math.log(d)
            count += 1
            x = self(x)
            if abs(x) > 1e15:
                break
        return total / max(count, 1)

    @property
    def band(self):
        if self._band is not None:
            return self._band
        traj = self.iterate(0.5, 200)
        if len(traj) < 3 or abs(traj[-1]) > 1e10:
            self._band = 1 if len(traj) > 20 else 0
            return self._band
        lam = self.lyapunov()
        tail = traj[-50:] if len(traj) >= 50 else traj[-20:]
        if len(tail) >= 4:
            for period in range(2, min(8, len(tail)//2)):
                if all(abs(tail[-(i+1)] - tail[-(i+1+period)]) < 1e-6 for i in range(min(period*3, len(tail)-period))):
                    self._band = 4
                    return 4
        fp = self.stable_fp()
        if fp and abs(fp[1]) < 1.0:
            self._band = 6 if abs(fp[1]) < 0.5 else 5
            return self._band
        if abs(lam) < 0.05:
            self._band = 2
        elif lam > 0:
            self._band = 3
        else:
            self._band = 5
        return self._band

    @property
    def gap(self):
        if self._gap is not None:
            return self._gap
        fp = self.stable_fp()
        self._gap = max(0.0, 1.0 - abs(fp[1])) if fp else 0.0
        return self._gap

    def compose(self, other_state):
        return COMP_TABLE[self._state, other_state % 10]

    def to_bytes(self):
        return np.array([self.a, self.b, self.c], dtype=np.float32).tobytes()

    @classmethod
    def from_bytes(cls, data, state=0):
        arr = np.frombuffer(data, dtype=np.float32)
        return cls(arr[0], arr[1], arr[2], state=state)

# HookBus (events)
class HookBus:
    def __init__(self):
        self._hooks = defaultdict(list)

    def on(self, event, callback):
        self._hooks[event].append(callback)

    def off(self, event, callback):
        self._hooks[event].remove(callback)

    def emit(self, event, **kwargs):
        for cb in self._hooks.get(event, []):
            cb(**kwargs)

# Lattice (substrate)
class Lattice:
    def __init__(self, rows=18, cols=14):
        self.rows = rows
        self.cols = cols
        self.cells = []
        self.tick_count = 0
        self.bus = HookBus()
        self.coherence_history = deque(maxlen=100)

    def init(self):
        self.cells = [[QuadraticOp(*OPS_CANONICAL[(i*self.cols+j)%10], state=(i*self.cols+j)%10) for j in range(self.cols)] for i in range(self.rows)]
        self.tick_count = 0
        self.coherence_history.clear()
        self.bus.emit('init', lattice=self)

    def neighbors(self, i, j):
        return [
            self.cells[(i-1)%self.rows][j],
            self.cells[(i+1)%self.rows][j],
            self.cells[i][(j-1)%self.cols],
            self.cells[i][(j+1)%self.cols],
        ]

    def tick(self):
        self.bus.emit('pre_tick', tick=self.tick_count)
        new_states = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        transitions = 0
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.cells[i][j]
                ns = [n.state for n in self.neighbors(i, j)]
                composed = [cell.compose(n) for n in ns]
                composed.append(cell.compose(cell.state))
                counts = Counter(composed)
                new_state = max(counts, key=counts.get)
                new_states[i][j] = new_state
                if new_state != cell.state:
                    transitions += 1
                cell.state = new_state
                cell.a, cell.b, cell.c = OPS_CANONICAL[new_state]
        self.tick_count += 1
        coh = self.coherence()
        self.coherence_history.append(coh)
        self.bus.emit('post_tick', tick=self.tick_count, coherence=coh, transitions=transitions)
        return coh

    def coherence(self):
        states = np.array([[c.state for c in row] for row in self.cells])
        n = self.rows * self.cols
        valid = 0
        aligned = 0
        for i in range(self.rows):
            for j in range(self.cols):
                s = states[i, j]
                ns = [n.state for n in self.neighbors(i, j)]
                comps = [COMP_TABLE[s, nn] for nn in ns]
                if any(c != s for c in comps) or s == 7:
                    valid += 1
                if s in (5,6,7,8):
                    aligned += 1
        v_star = valid / n
        a_star = aligned / n
        s_star = D_STAR
        for _ in range(20):
            s_star = SIGMA * (1 - s_star) * v_star * a_star
        return s_star

    def inject(self, i, j, state):
        state = state % 10
        self.cells[i][j].state = state
        self.cells[i][j].a, self.cells[i][j].b, self.cells[i][j].c = OPS_CANONICAL[state]
        self.bus.emit('inject', row=i, col=j, state=state)

    def inject_word(self, seq, row=0, col=0):
        for k, s in enumerate(seq):
            self.inject(row, (col + k) % self.cols, s)

    def read_row(self, row):
        return [self.cells[row][j].state for j in range(self.cols)]

    def status(self):
        lines = []
        coh = self.coherence()
        lines.append(f"S* = {coh:.4f} ({'above T*' if coh >= T_STAR else 'below T*'})")
        census = Counter(cell.state for row in self.cells for cell in row)
        for s in range(10):
            lines.append(f"{OP_NAMES[s]}: {census[s]}")
        return '\n'.join(lines)

    def save_bin(self, path='tig_lattice.bin'):
        data = np.array([[[c.a, c.b, c.c] for c in row] for row in self.cells], dtype=np.float32)
        data.tofile(path)
        return path, os.path.getsize(path)

    def load_bin(self, path='tig_lattice.bin'):
        data = np.fromfile(path, dtype=np.float32).reshape(self.rows, self.cols, 3)
        for i in range(self.rows):
            for j in range(self.cols):
                a, b, c = data[i, j]
                best_state = min(OPS_CANONICAL, key=lambda s: sum((v - OPS_CANONICAL[s][k])**2 for k, v in enumerate([a, b, c])))
                self.cells[i][j] = QuadraticOp(a, b, c, state=best_state)

# Universal Input Hooks (computation, physics, language)
def input_computation(metrics):
    # Map OS metrics to 0-9 states (e.g., cpu_avg 0.0=0 VOID, 1.0=9 FRUIT)
    word = [int(value * 9) % 10 for value in metrics.values() if isinstance(value, float)]
    return word

def input_physics(data):
    # Map physics sim data (e.g., velocities) to states
    word = [int(math.tanh(v) * 9) % 10 for v in data]
    return word

def input_language(text):
    # Map text to states (simple hash-based, as per linguistics)
    word = [int(hashlib.md5(text.encode()).hexdigest()[i:i+1], 16) % 10 for i in range(10)]
    return word  # Stub — extend for full linguistics

# Actuator (steer based on output)
class Actuator:
    def __init__(self):
        self.dry_run = True

    def act(self, output_word):
        # Map output states to actions (e.g., 7 = harmony → no action, 4 = collapse → alert)
        for s in output_word:
            if s == 4:
                print("ALERT: Collapse detected — take action")
            # Extend for real actuation (nice, affinity, etc.)

# TIG Universal Engine (all-in-one)
class TIGUniversal:
    def __init__(self, rows=18, cols=14):
        self.lattice = Lattice(rows, cols)
        self.lattice.init()
        self.actuator = Actuator()

    def feed(self, domain, data):
        if domain == 'computation':
            word = input_computation(data)
        elif domain == 'physics':
            word = input_physics(data)
        elif domain == 'language':
            word = input_language(data)
        else:
            word = [int(d % 10) for d in data if isinstance(d, int)]
        self.lattice.inject_word(word, row=0)
        self.lattice.bus.emit('feed', domain=domain, word=word)

    def compute(self, ticks=10):
        for _ in range(ticks):
            self.lattice.tick()
        return self.lattice.coherence()

    def truth(self):
        return self.lattice.status()

    def act(self):
        output = self.lattice.read_row(self.lattice.rows - 1)
        self.actuator.act(output)
        return output

    def save_floppy(self, path='tig_truth.bin'):
        return self.lattice.save_bin(path)

# Main — for humanity to use
if __name__ == '__main__':
    engine = TIGUniversal()
    # Example: computation (fake OS metrics)
    os_metrics = {'cpu': 0.4, 'mem': 0.7}
    engine.feed('computation', os_metrics)
    # Physics (fake velocities)
    physics_data = [1.2, 3.4, 5.6]
    engine.feed('physics', physics_data)
    # Language
    text = "Seek truth"
    engine.feed('language', text)
    # Compute
    coh = engine.compute(ticks=20)
    print(f"Universal Coherence: S* = {coh:.4f}")
    print(engine.truth())
    # Act
    engine.act()
    # Save to floppy
    path, size = engine.save_floppy()
    print(f"Truth on floppy: {path} ({size} bytes)")
