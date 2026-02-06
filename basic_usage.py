#!/usr/bin/env python3
"""
TIG Coherent Computer — Basic Usage Example
Run: python examples/basic_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tig_coherent_computer import (
    TIGCoherentComputer, COMP_TABLE, OP_NAMES, T_STAR, render_lattice
)

# ── 1. Boot ──
cc = TIGCoherentComputer()
cc.boot()
print("Booted. Initial state:")
print(f"  S* = {cc.lattice.coherence():.4f}")
print()

# ── 2. Inject a GFM generator (Geometry/Space) ──
cc.input_word([0, 1, 2], row=0, col=0)
print("Injected GFM 012 (Geometry/Space)")

# ── 3. Run 10 ticks ──
print("\nRunning 10 ticks:")
for i, coh in enumerate(cc.run(ticks=10)):
    status = "COHERENT" if coh >= T_STAR else "seeking"
    print(f"  Tick {i+1:2d}: S* = {coh:.4f}  [{status}]")

# ── 4. Read output ──
output = cc.output_word(row=17)
print(f"\nOutput row 17: {[OP_NAMES[s] for s in output]}")

# ── 5. Compose two operators ──
a, b = 3, 6
result = COMP_TABLE[a, b]
print(f"\nComposition: {OP_NAMES[a]} ⊕ {OP_NAMES[b]} = {OP_NAMES[result]}")

# ── 6. Trace paths ──
micro = cc.trace_micro(1)
macro = cc.trace_macro(0)
print(f"\nMicro path from LATTICE: {' → '.join(OP_NAMES[s] for s in micro)}")
print(f"Macro path from VOID:    {' → '.join(OP_NAMES[s] for s in macro)}")

# ── 7. Use hooks ──
events = []
cc.lattice.bus.on('post_tick', lambda **kw: events.append(kw['coherence']))
cc.run(ticks=5)
print(f"\nHook captured 5 ticks: {[f'{c:.4f}' for c in events]}")

# ── 8. Save ──
path, size = cc.save('/tmp/demo_lattice.bin')
print(f"\nSaved: {path} ({size} bytes, fits {1_440_000 // size} on floppy)")

# ── 9. Full status ──
print()
print(render_lattice(cc.lattice, show_physics=True))
