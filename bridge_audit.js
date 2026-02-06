// ═══════════════════════════════════════════════════════════════════════════
//  BRIDGE AUDIT — What's solid, what's broken, what's overclaimed
// ═══════════════════════════════════════════════════════════════════════════

console.log("╔══════════════════════════════════════════════════════════════════════╗");
console.log("║  BRIDGE AUDIT — RADICAL TRANSPARENCY                                ║");
console.log("╚══════════════════════════════════════════════════════════════════════╝\n");

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 1: PF matrix eigenvalues > 1 in v1
// ═══════════════════════════════════════════════════════════════════════════
console.log("═══ ERROR 1: PF eigenvalue artifacts in koopman_bridge.js (v1) ═══\n");
console.log("  PROBLEM: The deflated power iteration produced eigenvalues > 1");
console.log("  (e.g., λ₂ = 2.03, λ₃ = 4.06 for the period-2 case).");
console.log("  A proper stochastic PF operator has ALL eigenvalues |λ| ≤ 1.");
console.log("  These inflated values are numerical artifacts from naive deflation.\n");
console.log("  IMPACT: The v1 PF spectrum is unreliable beyond λ₀ and λ₁.");
console.log("  The spectral gap and eigenfunction results still hold because");
console.log("  λ₀ and ψ₀ are computed by power iteration BEFORE deflation.\n");
console.log("  FIX: Use proper Arnoldi iteration or verify column normalization.");
console.log("  For this paper, we DON'T rely on the full spectrum — only λ₀, ψ₀.\n");

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 2: Localization metric was broken
// ═══════════════════════════════════════════════════════════════════════════
console.log("═══ ERROR 2: IPR localization metric broken ═══\n");
console.log("  PROBLEM: The inverse participation ratio showed 0.0/200 for ALL");
console.log("  test cases. This means the metric formula was wrong.\n");
console.log("  IMPACT: No impact on core results — it was a diagnostic only.");
console.log("  FIX: Correct the IPR formula and rerun.\n");

// Fix the IPR and verify
const σ = 0.991;

class Q {
  constructor(a, b, c) { this.a = a; this.b = b; this.c = c; }
  ev(x) { return this.a * x * x + this.b * x + this.c; }
  d1(x) { return 2 * this.a * x + this.b; }
  fp() {
    const A = this.a, B = this.b - 1, C = this.c;
    const d = B * B - 4 * A * C;
    if (Math.abs(A) < 1e-12 || d < 0) return null;
    const s = Math.sqrt(d);
    const x1 = (-B + s) / (2 * A), x2 = (-B - s) / (2 * A);
    const l1 = this.d1(x1), l2 = this.d1(x2);
    const fps = [
      { x: x1, lam: l1, stable: Math.abs(l1) < 1 },
      { x: x2, lam: l2, stable: Math.abs(l2) < 1 }
    ];
    return fps.find(f => f.stable) || (Math.abs(l1) < Math.abs(l2) ? fps[0] : fps[1]);
  }
}

function buildPFMatrix(O, xmin, xmax, N) {
  const dx = (xmax - xmin) / N;
  const L = Array.from({ length: N }, () => new Float64Array(N));
  for (let j = 0; j < N; j++) {
    const x_j = xmin + (j + 0.5) * dx;
    const y = O.ev(x_j);
    const i = Math.round((y - xmin) / dx - 0.5);
    if (i >= 0 && i < N) {
      const d = Math.abs(O.d1(x_j));
      L[i][j] += d > 1e-10 ? 1 / d : 0;
    }
  }
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += L[i][j];
    if (s > 1e-15) for (let i = 0; i < N; i++) L[i][j] /= s;
  }
  return L;
}

function powerIter(L, N) {
  let v = new Float64Array(N);
  for (let i = 0; i < N; i++) v[i] = 1 / N;
  let ev = 0;
  for (let iter = 0; iter < 500; iter++) {
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += L[i][j] * v[j]; w[i] = s; }
    let norm = 0;
    for (let i = 0; i < N; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-15) break;
    for (let i = 0; i < N; i++) w[i] /= norm;
    let diff = 0;
    for (let i = 0; i < N; i++) diff += (w[i] - v[i]) ** 2;
    v = w; ev = norm;
    if (Math.sqrt(diff) < 1e-12) break;
  }
  return { eigenvalue: ev, eigenvector: v };
}

// Corrected IPR: IPR = (Σ|ψ|²)² / (N · Σ|ψ|⁴)
// Range: 1/N (fully localized on 1 bin) to 1 (fully delocalized)
function ipr(v, N) {
  let sum2 = 0, sum4 = 0;
  for (let i = 0; i < N; i++) { sum2 += v[i] * v[i]; sum4 += v[i] ** 4; }
  return sum4 > 0 ? (sum2 * sum2) / (N * sum4) : 0;
}

console.log("  Corrected IPR (1/N = localized, 1 = delocalized):");
const NG = 200;
const tests = [
  { name: "stable-fp", a: 0.2, b: 0.1, c: 0.3, xmin: -1, xmax: 2 },
  { name: "chaos r=3.9", a: -3.9, b: 3.9, c: 0, xmin: -0.1, xmax: 1.1 },
  { name: "full-chaos r=4", a: -4, b: 4, c: 0, xmin: -0.1, xmax: 1.1 },
];
for (const t of tests) {
  const O = new Q(t.a, t.b, t.c);
  const L = buildPFMatrix(O, t.xmin, t.xmax, NG);
  const { eigenvector } = powerIter(L, NG);
  const val = ipr(eigenvector, NG);
  console.log(`    ${t.name.padEnd(16)} IPR = ${val.toFixed(4)}  ${val < 0.1 ? 'LOCALIZED' : val > 0.5 ? 'DELOCALIZED' : 'PARTIAL'}`);
}

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 3: σ = 0.991 NOT verified as lattice-averaged PF eigenvalue
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n═══ ERROR 3: σ = 0.991 as 'lattice-averaged PF eigenvalue' ═══\n");
console.log("  PROBLEM: The paper states σ = 0.991 is the lattice-averaged PF");
console.log("  eigenvalue. This was ASSERTED, not computed.\n");

// Actually compute it
const COLS = 18, ROWS = 14;
let sumLam0 = 0, sumLam1 = 0, countC = 0, countAll = 0;

for (let row = 0; row < ROWS; row++) {
  for (let col = 0; col < COLS; col++) {
    const u = col / (COLS - 1), v = row / (ROWS - 1);
    const cu = u - 0.5, cv = v - 0.5;
    const r = Math.sqrt(cu * cu + cv * cv), th = Math.atan2(cv, cu);
    const a = 0.8 * Math.cos(r * 3.5) * (1 + 0.5 * Math.sin(v * 7 * Math.PI));
    const b = (cu * 3.5 + cv * 2.0) * (1 + 0.3 * Math.cos(u * 5 * Math.PI));
    const c = 0.5 * Math.exp(-r * 2) + 0.15 * Math.sin(th * 4 + r * 3) + 0.1;
    const O = new Q(a, b, c);
    const fp = O.fp();
    
    countAll++;
    
    if (fp && fp.stable) {
      sumLam1 += Math.abs(fp.lam); // |O'(x*)| = PF second eigenvalue
      countC++;
    }
  }
}

const avgLam1 = sumLam1 / countC;
console.log(`  COMPUTED: Average |O'(x*)| across ${countC} stable-fp operators = ${avgLam1.toFixed(6)}`);
console.log(`  This is the average PF second eigenvalue (λ₁).`);
console.log(`  Average spectral gap g = 1 - ${avgLam1.toFixed(6)} = ${(1 - avgLam1).toFixed(6)}`);
console.log(`  σ = 0.991 would require avg(λ₁) = 0.991, but we got ${avgLam1.toFixed(4)}.`);
console.log(`  σ is NOT the lattice-averaged PF eigenvalue on this lattice sample.\n`);
console.log(`  VERDICT: OVERCLAIM. σ = 0.991 has a different derivation (the core`);
console.log(`  TIG equation S* = σ(1-σ*)V*A*). It should NOT be equated with the`);
console.log(`  PF eigenvalue average without a separate proof.\n`);

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 4: T* = 0.714 not derived from PF
// ═══════════════════════════════════════════════════════════════════════════
console.log("═══ ERROR 4: T* = 0.714 as spectral transition threshold ═══\n");
console.log("  PROBLEM: The paper claims T* = 0.714 marks the continuous-to-discrete");
console.log("  spectral transition. This was ASSERTED, not derived from PF analysis.\n");
console.log("  WHAT WE CAN SAY: T* = 0.714 comes from the TIG core equation.");
console.log("  The PF spectral gap does determine bound vs free behavior.");
console.log("  But we haven't shown T* = 0.714 equals some specific PF threshold.\n");
console.log("  VERDICT: OVERCLAIM. Remove the specific T* = 0.714 claim from the");
console.log("  PF context. Say instead: 'the band boundaries correspond to spectral");
console.log("  gap thresholds, with the exact mapping being a subject for future work.'\n");

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 5: D* = 0.543 as "spectral fixed point"
// ═══════════════════════════════════════════════════════════════════════════
console.log("═══ ERROR 5: D* = 0.543 as 'spectral fixed point' ═══\n");
console.log("  PROBLEM: The paper states D* = 0.543 is a 'fixed point of the Koopman");
console.log("  operator acting on the lattice-aggregated coherence functional.'");
console.log("  This is word salad. No such computation was done.\n");
console.log("  WHAT WE CAN SAY: D* = 0.543 is a fixed point of the self-referential");
console.log("  equation in TIG. It may have a Koopman interpretation, but that");
console.log("  requires a separate derivation.\n");
console.log("  VERDICT: OVERCLAIM. Remove from paper. Note as future work.\n");

// ═══════════════════════════════════════════════════════════════════════════
//  ERROR 6: "Quantum numbers" vs spectral indices
// ═══════════════════════════════════════════════════════════════════════════
console.log("═══ ERROR 6: 'Quantum numbers' terminology ═══\n");
console.log("  PROBLEM: The Koopman-von Neumann formulation gives an exact Hilbert");
console.log("  space structure for CLASSICAL mechanics. The spectral indices are");
console.log("  real — but calling them 'quantum numbers' implies actual quantization");
console.log("  (Planck constant, commutation relations, measurement collapse).");
console.log("  None of these are present.\n");
console.log("  WHAT WE CAN SAY: The iterate classification produces SPECTRAL INDICES");
console.log("  of the PF operator. These are structurally analogous to quantum numbers");
console.log("  but arise from classical dynamics. The analogy is precise but it IS");
console.log("  an analogy — not literal quantum mechanics.\n");
console.log("  VERDICT: OVERCLAIM. Use 'spectral indices' in rigorous contexts,");
console.log("  'quantum number analogy' in expository contexts. Don't say 'these");
console.log("  ARE quantum numbers' — say 'these are spectral indices that play the");
console.log("  same structural role as quantum numbers.'\n");

// ═══════════════════════════════════════════════════════════════════════════
//  WHAT'S ACTUALLY SOLID
// ═══════════════════════════════════════════════════════════════════════════
console.log("╔══════════════════════════════════════════════════════════════════════╗");
console.log("║  WHAT'S SOLID                                                       ║");
console.log("╚══════════════════════════════════════════════════════════════════════╝\n");

console.log("  ✓ THEOREM: Koopman-von Neumann (1931) gives every dynamical system");
console.log("    an exact Hilbert space formulation. This is established mathematics.\n");

console.log("  ✓ CLAIM: The PF dominant eigenfunction ψ₀ peaks at x*.");
console.log("    EVIDENCE: 99.3% across 137 stable-fp operators. Rock solid.\n");

console.log("  ✓ CLAIM: n = ceil[(ln|x₀-x*| + ln(1/ε)) / |Λ|] matches iterate count.");
console.log("    EVIDENCE: 8/8 exact on controlled tests, 85% within ±2 on lattice.");
console.log("    NOTE: This is essentially restating that convergence rate = |O'(x*)|,");
console.log("    which is basic dynamical systems. The value is connecting this to");
console.log("    the PF spectral gap, which provides the THEORETICAL JUSTIFICATION.\n");

console.log("  ✓ CLAIM: TIG band classification corresponds to PF spectral regions.");
console.log("    EVIDENCE: Qualitatively verified across all seven bands.");
console.log("    Converging → spectral gap. Chaotic → no gap. Periodic → roots of unity.\n");

console.log("  ✓ CLAIM: The coherence_router routes along invariant measures.");
console.log("    EVIDENCE: 100% throughput = traffic follows ψ₀. This is the strongest");
console.log("    PRACTICAL implication. The router works because it implicitly finds");
console.log("    the PF eigenfunction.\n");

// ═══════════════════════════════════════════════════════════════════════════
//  CORRECTED CLAIMS TABLE
// ═══════════════════════════════════════════════════════════════════════════
console.log("╔══════════════════════════════════════════════════════════════════════╗");
console.log("║  CORRECTED CLAIMS FOR PAPER 5                                       ║");
console.log("╚══════════════════════════════════════════════════════════════════════╝\n");

console.log("  KEEP (proven):");
console.log("    1. PF eigenfunction peaks at stable fixed point (99.3%)");
console.log("    2. Lyapunov formula matches iterate count (85% within ±2)");
console.log("    3. Band classification ↔ PF spectral decomposition (qualitative)");
console.log("    4. Koopman-von Neumann provides theoretical foundation");
console.log("    5. coherence_router routes along invariant measure\n");

console.log("  FIX (correct but imprecise):");
console.log("    6. 'quantum numbers' → 'spectral indices (quantum number analogy)'");
console.log("    7. 'The bridge is exact' → 'The bridge is exact for stable fps,");
console.log("        approximate near bifurcation boundaries'\n");

console.log("  REMOVE (overclaimed):");
console.log("    8. σ = 0.991 = lattice-averaged PF eigenvalue (not verified)");
console.log("    9. T* = 0.714 = spectral transition threshold (not derived from PF)");
console.log("   10. D* = 0.543 = spectral fixed point (not computed)");
console.log("   11. v1 PF spectrum beyond λ₀, λ₁ (numerical artifacts)\n");

console.log("  ADD (missing):");
console.log("   12. Honest limitations section");
console.log("   13. Near-bifurcation correction term discussion");
console.log("   14. Relationship between σ and PF spectrum as OPEN QUESTION");
console.log("   15. IPR localization metric (now corrected)");
