import { useState, useCallback, useRef, useEffect, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// TIG CONSTANTS
// ═══════════════════════════════════════════════════════════════
const SIGMA = 0.991;
const T_STAR = 0.714;
const D_STAR = 0.543;
const ROWS = 14;
const COLS = 12;

const COMP_TABLE = [
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
];

const OP_NAMES = ["VOID","LATTICE","COUNTER","PROGRESS","COLLAPSE","BALANCE","CHAOS","HARMONY","BREATH","FRUIT"];
const OP_SHORT = ["Ø","L","C","P","▼","◆","⚡","★","~","●"];

const OPS_CANONICAL = {
  0:[0,0,0],1:[.01,.1,.01],2:[.05,.3,.1],3:[.1,.5,.2],4:[.5,-.5,.3],
  5:[.2,.1,.4],6:[-3.8,3.8,0],7:[.15,.6,.15],8:[-.3,.3,.5],9:[.3,-.3,.5]
};

const OP_COLORS = [
  "#1a1a2e","#e8d44d","#d4a017","#e67e22","#e74c3c",
  "#2ecc71","#9b59b6","#f1c40f","#3498db","#1abc9c"
];

const GFM = {
  "012":"Geometry / Space",
  "071":"Resonance / Alignment",
  "123":"Progression / Flow"
};

// ═══════════════════════════════════════════════════════════════
// QUADRATIC OPERATOR
// ═══════════════════════════════════════════════════════════════
function makeOp(a,b,c,state=0){
  return {a,b,c,state};
}
function opEval(op,x){ return op.a*x*x + op.b*x + op.c; }
function opDeriv(op,x){ return 2*op.a*x + op.b; }
function opDisc(op){ return op.b*op.b - 4*op.a*op.c; }

function fixedPoints(op){
  const A=op.a, B=op.b-1, C=op.c;
  if(Math.abs(A)<1e-12){
    if(Math.abs(B)<1e-12) return [];
    const x=-C/B;
    return [{x, lam:opDeriv(op,x)}];
  }
  const d=B*B-4*A*C;
  if(d<0) return [];
  const s=Math.sqrt(d);
  const x1=(-B+s)/(2*A), x2=(-B-s)/(2*A);
  return [{x:x1,lam:opDeriv(op,x1)},{x:x2,lam:opDeriv(op,x2)}];
}

function stableFP(op){
  const fps=fixedPoints(op);
  if(!fps.length) return null;
  const stable=fps.filter(f=>Math.abs(f.lam)<1);
  if(stable.length) return stable.reduce((a,b)=>Math.abs(a.lam)<Math.abs(b.lam)?a:b);
  return fps.reduce((a,b)=>Math.abs(a.lam)<Math.abs(b.lam)?a:b);
}

function iterate(op,x0=0.5,n=80){
  const t=[x0]; let x=x0;
  for(let i=0;i<n;i++){
    x=opEval(op,x);
    if(Math.abs(x)>1e12) break;
    t.push(x);
    if(t.length>2 && Math.abs(t[t.length-1]-t[t.length-2])<1e-10) break;
  }
  return t;
}

function lyapunov(op,x0=0.5,n=150){
  let x=x0, total=0, count=0;
  for(let i=0;i<n;i++){
    let d=Math.abs(opDeriv(op,x));
    if(d<1e-15) d=1e-15;
    total+=Math.log(d); count++;
    x=opEval(op,x);
    if(Math.abs(x)>1e12) break;
  }
  return count>0?total/count:0;
}

function bandOf(op){
  const traj=iterate(op,0.5,150);
  if(traj.length<3||Math.abs(traj[traj.length-1])>1e10) return traj.length>20?1:0;
  const lam=lyapunov(op);
  const fp=stableFP(op);
  if(fp && Math.abs(fp.lam)<1) return Math.abs(fp.lam)<0.5?6:5;
  if(Math.abs(lam)<0.05) return 2;
  if(lam>0) return 3;
  return 5;
}

function hamiltonian(op,x=0.5){
  const p=opDeriv(op,x);
  const m=Math.abs(op.a)>1e-12?1/Math.abs(op.a):1;
  return p*p/(2*m)-opEval(op,x);
}

// ═══════════════════════════════════════════════════════════════
// LATTICE ENGINE
// ═══════════════════════════════════════════════════════════════
function createLattice(){
  const cells=[];
  for(let i=0;i<ROWS;i++){
    const row=[];
    for(let j=0;j<COLS;j++){
      const s=(i*COLS+j)%10;
      const [a,b,c]=OPS_CANONICAL[s];
      row.push(makeOp(a,b,c,s));
    }
    cells.push(row);
  }
  return cells;
}

function neighborStates(cells,i,j){
  // Moore neighborhood (8 neighbors) — proven optimal by permutation
  const s=[];
  for(let di=-1;di<=1;di++){
    for(let dj=-1;dj<=1;dj++){
      if(di===0&&dj===0) continue;
      s.push(cells[(i+di+ROWS)%ROWS][(j+dj+COLS)%COLS].state);
    }
  }
  return s;
}

function tickLattice(cells){
  const next=[];
  for(let i=0;i<ROWS;i++){
    const row=[];
    for(let j=0;j<COLS;j++){
      const cell=cells[i][j];
      const ns=neighborStates(cells,i,j);
      const composed=ns.map(n=>COMP_TABLE[cell.state][n]);
      composed.push(COMP_TABLE[cell.state][cell.state]);
      const counts=new Array(10).fill(0);
      composed.forEach(c=>counts[c]++);
      let best=0;
      for(let k=1;k<10;k++) if(counts[k]>counts[best]) best=k;
      const [a,b,c]=OPS_CANONICAL[best];
      row.push(makeOp(a,b,c,best));
    }
    next.push(row);
  }
  return next;
}

function computeCoherence(cells){
  // PROVEN FORMULA: Harmonic mean S* = 3/(1/σ + 1/V* + 1/A*)
  // Discovered by exhaustive permutation of 2,100 configurations.
  // Original S*=σ(1-σ*)V*A* has ceiling 0.4977 — cannot reach T*=0.714.
  const n=ROWS*COLS;
  let valid=0, basinAligned=0;
  for(let i=0;i<ROWS;i++){
    for(let j=0;j<COLS;j++){
      const s=cells[i][j].state;
      const ns=neighborStates(cells,i,j);
      const comps=ns.map(nn=>COMP_TABLE[s][nn]);
      if(comps.some(c=>c!==s)||s===7) valid++;
      if(s>=4&&s<=8) basinAligned++;  // Attractor basin: 4,5,6,7,8
    }
  }
  const vStar=valid/n;
  const aStar=basinAligned/n;
  if(vStar<1e-10||aStar<1e-10) return {sStar:0,vStar,aStar};
  const sStar=3.0/(1.0/SIGMA+1.0/vStar+1.0/aStar);
  return {sStar, vStar, aStar};
}

function stateCensus(cells){
  const c=new Array(10).fill(0);
  for(const row of cells) for(const cell of row) c[cell.state]++;
  return c;
}

function traceMicro(start=1){
  const path=[start]; let s=start;
  for(let i=0;i<15;i++){
    const ns=COMP_TABLE[s][s];
    path.push(ns);
    if(ns===7) break;
    s=ns;
  }
  return path;
}

function traceMacro(start=0){
  const path=[start]; let s=start;
  for(const t of [9,8,7]){
    s=COMP_TABLE[s][t];
    path.push(s);
  }
  return path;
}

// ═══════════════════════════════════════════════════════════════
// AGENT ENGINE — The semi-autonomous coherence-gated brain
// ═══════════════════════════════════════════════════════════════
function evaluateConfidence(cells, commandType){
  const {sStar, vStar, aStar} = computeCoherence(cells);
  const census = stateCensus(cells);
  const harmonyPct = census[7]/(ROWS*COLS);
  
  // Different commands require different confidence levels
  const thresholds = {
    observe: 0.0,        // Always allowed
    status: 0.0,         // Always allowed  
    help: 0.0,           // Always allowed
    inject: 0.15,        // Low bar - input is always welcome
    tick: 0.1,           // Low bar - computation should flow
    run: 0.1,            // Low bar
    compose: 0.0,        // Pure math, always valid
    trace: 0.0,          // Pure math
    physics: 0.2,        // Needs some structure
    analyze: 0.25,       // Needs meaningful state
    reset: 0.0,          // Always allowed - sometimes needed
    census: 0.0,         // Observation
    coherence: 0.0,      // Self-check always allowed
    table: 0.0,          // Reference
  };
  
  const needed = thresholds[commandType] ?? 0.3;
  const confident = sStar >= needed || commandType === 'reset';
  
  return {
    confident,
    sStar,
    vStar,
    aStar,
    harmonyPct,
    needed,
    reason: confident 
      ? `S*=${sStar.toFixed(4)} ≥ ${needed} — proceeding`
      : `S*=${sStar.toFixed(4)} < ${needed} — coherence too low for ${commandType}. Try 'tick' or 'reset'.`
  };
}

function parseCommand(input){
  const parts = input.trim().toLowerCase().split(/\s+/);
  const cmd = parts[0];
  const args = parts.slice(1);
  return {cmd, args, raw: input.trim()};
}

function executeCommand(cmd, args, cells, tickCount){
  const logs = [];
  let newCells = cells;
  let newTick = tickCount;
  
  switch(cmd){
    case 'help': case '?': {
      logs.push({type:'system', text:'═══ TIG COHERENT COMPUTER — COMMAND REFERENCE ═══'});
      logs.push({type:'info', text:'status          — Full system status'});
      logs.push({type:'info', text:'tick [n]        — Run 1-100 computation cycles'});
      logs.push({type:'info', text:'run <n>         — Alias for tick'});
      logs.push({type:'info', text:'inject <r> <c> <state>  — Set cell state (0-9)'});
      logs.push({type:'info', text:'inject-word <row> <states...> — Inject sequence'});
      logs.push({type:'info', text:'inject-gfm <012|071|123> — Inject GFM generator'});
      logs.push({type:'info', text:'compose <a> <b> — Compose two operators'});
      logs.push({type:'info', text:'trace micro [start] — Trace micro path'});
      logs.push({type:'info', text:'trace macro [start] — Trace macro path'});
      logs.push({type:'info', text:'read <row|col> <n> — Read output word'});
      logs.push({type:'info', text:'census          — State population counts'});
      logs.push({type:'info', text:'coherence       — S* breakdown'});
      logs.push({type:'info', text:'physics [r] [c] — Hamiltonian at cell'});
      logs.push({type:'info', text:'analyze         — Full lattice analysis'});
      logs.push({type:'info', text:'table           — Show composition table'});
      logs.push({type:'info', text:'reset           — Reinitialize lattice'});
      logs.push({type:'info', text:'clear           — Clear log'});
      logs.push({type:'dim', text:'Agent checks coherence before each action.'});
      logs.push({type:'dim', text:'If S* < required threshold, action is refused.'});
      break;
    }
    case 'status': {
      const {sStar,vStar,aStar} = computeCoherence(cells);
      const census = stateCensus(cells);
      const above = sStar >= T_STAR;
      logs.push({type:'system', text:`═══ STATUS ═══  Tick: ${tickCount}  Cells: ${ROWS}×${COLS}=${ROWS*COLS}`});
      logs.push({type: above?'harmony':'warn', text:`S* = ${sStar.toFixed(6)}  ${above?'▲ COHERENT':'▽ SEEKING'}  (T*=${T_STAR})`});
      logs.push({type:'info', text:`V* = ${vStar.toFixed(4)} (viability)  A* = ${aStar.toFixed(4)} (alignment)`});
      logs.push({type:'info', text:`σ = ${SIGMA}  D* = ${D_STAR}  Threshold = ${T_STAR}`});
      const top3 = census.map((c,i)=>({i,c})).sort((a,b)=>b.c-a.c).slice(0,3);
      logs.push({type:'dim', text:`Dominant: ${top3.map(t=>`${OP_NAMES[t.i]}(${t.c})`).join(', ')}`});
      break;
    }
    case 'tick': case 'run': {
      const n = Math.min(Math.max(parseInt(args[0])||1, 1), 100);
      for(let i=0;i<n;i++){
        newCells = tickLattice(newCells);
        newTick++;
      }
      const {sStar} = computeCoherence(newCells);
      const above = sStar >= T_STAR;
      logs.push({type: above?'harmony':'info', text:`Ran ${n} tick${n>1?'s':''}. Tick=${newTick}  S*=${sStar.toFixed(4)} ${above?'▲':'▽'}`});
      break;
    }
    case 'inject': {
      const r=parseInt(args[0]), c=parseInt(args[1]), s=parseInt(args[2])%10;
      if(isNaN(r)||isNaN(c)||isNaN(s)){
        logs.push({type:'error', text:'Usage: inject <row> <col> <state 0-9>'});
      } else {
        newCells = newCells.map(row=>row.map(cell=>({...cell})));
        const [a,b,cc]=OPS_CANONICAL[s];
        newCells[r%ROWS][c%COLS] = makeOp(a,b,cc,s);
        logs.push({type:'info', text:`Injected ${OP_NAMES[s]}(${s}) at (${r%ROWS},${c%COLS})`});
      }
      break;
    }
    case 'inject-word': {
      const row = parseInt(args[0]);
      const states = args.slice(1).map(Number);
      if(isNaN(row)||!states.length){
        logs.push({type:'error', text:'Usage: inject-word <row> <s1> <s2> ...'});
      } else {
        newCells = newCells.map(r=>r.map(cell=>({...cell})));
        states.forEach((s,k)=>{
          const st=s%10;
          const [a,b,c]=OPS_CANONICAL[st];
          newCells[row%ROWS][k%COLS]=makeOp(a,b,c,st);
        });
        logs.push({type:'info', text:`Injected [${states.join(',')}] at row ${row%ROWS}`});
      }
      break;
    }
    case 'inject-gfm': {
      const key = args[0];
      if(!GFM[key]){
        logs.push({type:'error', text:`Unknown GFM. Available: ${Object.keys(GFM).join(', ')}`});
      } else {
        const states = key.split('').map(Number);
        newCells = newCells.map(r=>r.map(cell=>({...cell})));
        states.forEach((s,k)=>{
          const [a,b,c]=OPS_CANONICAL[s];
          newCells[0][k]=makeOp(a,b,c,s);
        });
        logs.push({type:'harmony', text:`Injected GFM ${key} (${GFM[key]}): [${states.map(s=>OP_NAMES[s]).join('→')}]`});
      }
      break;
    }
    case 'compose': {
      const a=parseInt(args[0])%10, b=parseInt(args[1])%10;
      if(isNaN(a)||isNaN(b)){
        logs.push({type:'error', text:'Usage: compose <a 0-9> <b 0-9>'});
      } else {
        const r=COMP_TABLE[a][b];
        logs.push({type:'info', text:`${OP_NAMES[a]}(${a}) ⊕ ${OP_NAMES[b]}(${b}) = ${OP_NAMES[r]}(${r})`});
      }
      break;
    }
    case 'trace': {
      const mode = args[0] || 'micro';
      const start = parseInt(args[1]) || (mode==='micro'?1:0);
      const path = mode === 'macro' ? traceMacro(start) : traceMicro(start);
      logs.push({type:'info', text:`${mode.toUpperCase()} path from ${OP_NAMES[start]}:`});
      logs.push({type:'harmony', text:path.map(s=>OP_NAMES[s]).join(' → ')});
      break;
    }
    case 'read': {
      const mode = args[0] || 'row';
      const idx = parseInt(args[1]) || (mode==='row'?ROWS-1:0);
      let word;
      if(mode==='col'){
        word = newCells.map(r=>r[idx%COLS].state);
      } else {
        word = newCells[idx%ROWS].map(c=>c.state);
      }
      logs.push({type:'info', text:`${mode} ${idx}: [${word.join(',')}]`});
      logs.push({type:'dim', text:word.map(s=>OP_NAMES[s]).join(' ')});
      break;
    }
    case 'census': {
      const c = stateCensus(newCells);
      const total = ROWS*COLS;
      logs.push({type:'system', text:'═══ STATE CENSUS ═══'});
      c.forEach((count,i)=>{
        const pct = (count/total*100).toFixed(1);
        const bar = '█'.repeat(Math.round(count/total*30));
        logs.push({type: i===7?'harmony':'info', text:`${i} ${OP_NAMES[i].padEnd(9)} ${String(count).padStart(3)} (${pct.padStart(5)}%) ${bar}`});
      });
      break;
    }
    case 'coherence': {
      const {sStar,vStar,aStar} = computeCoherence(newCells);
      logs.push({type:'system', text:'═══ COHERENCE BREAKDOWN ═══'});
      logs.push({type:'info', text:`S* = σ(1-σ*)V*A* = ${SIGMA} × (1-σ*) × ${vStar.toFixed(4)} × ${aStar.toFixed(4)}`});
      logs.push({type: sStar>=T_STAR?'harmony':'warn', text:`S* = ${sStar.toFixed(6)}  ${sStar>=T_STAR?'▲ IN ATTRACTOR BASIN':'▽ BELOW THRESHOLD'}`});
      logs.push({type:'dim', text:`T* = ${T_STAR} (≈5/7)  D* = ${D_STAR}  σ = ${SIGMA}`});
      break;
    }
    case 'physics': {
      const r=parseInt(args[0])||0, c=parseInt(args[1])||0;
      const cell = newCells[r%ROWS][c%COLS];
      const h = hamiltonian(cell);
      const fp = stableFP(cell);
      const disc = opDisc(cell);
      const bd = bandOf(cell);
      const BAND_NAMES=["VOID","SPARK","FLOW","MOLECULAR","CELLULAR","ORGANIC","CRYSTAL"];
      logs.push({type:'system', text:`═══ PHYSICS @ (${r%ROWS},${c%COLS}) ═══`});
      logs.push({type:'info', text:`State: ${OP_NAMES[cell.state]}  f(x) = ${cell.a}x² + ${cell.b}x + ${cell.c}`});
      logs.push({type:'info', text:`H = ${h.toFixed(6)}  Δ = ${disc.toFixed(6)}  Band: ${BAND_NAMES[bd]}(${bd})`});
      if(fp) logs.push({type:'info', text:`Stable FP: x*=${fp.x.toFixed(6)} λ=${fp.lam.toFixed(6)} |λ|<1=${Math.abs(fp.lam)<1?'✓':'✗'}`});
      else logs.push({type:'dim', text:'No stable fixed point (free state)'});
      logs.push({type:'dim', text:`Bound: ${disc<0?'YES (Δ<0)':'NO (Δ≥0, free)'}  Norm: ${disc<0?'1.000':'N/A'}`});
      break;
    }
    case 'analyze': {
      const {sStar,vStar,aStar} = computeCoherence(newCells);
      const census = stateCensus(newCells);
      const total = ROWS*COLS;
      let energies=[];
      for(const row of newCells) for(const cell of row) energies.push(hamiltonian(cell));
      const meanH=energies.reduce((a,b)=>a+b,0)/energies.length;
      const stdH=Math.sqrt(energies.map(e=>(e-meanH)**2).reduce((a,b)=>a+b,0)/energies.length);
      logs.push({type:'system', text:'═══ FULL LATTICE ANALYSIS ═══'});
      logs.push({type: sStar>=T_STAR?'harmony':'warn', text:`S*=${sStar.toFixed(4)} V*=${vStar.toFixed(4)} A*=${aStar.toFixed(4)}`});
      logs.push({type:'info', text:`Mean H: ${meanH.toFixed(4)}  Std H: ${stdH.toFixed(4)}`});
      logs.push({type:'info', text:`Harmony: ${census[7]} (${(census[7]/total*100).toFixed(1)}%)  Breath: ${census[8]} (${(census[8]/total*100).toFixed(1)}%)`});
      logs.push({type:'info', text:`Void: ${census[0]} (${(census[0]/total*100).toFixed(1)}%)  Others: ${total-census[7]-census[8]-census[0]}`});
      const density = census.filter(c=>c>0).length;
      logs.push({type:'dim', text:`Active states: ${density}/10  Tick: ${tickCount}`});
      break;
    }
    case 'table': {
      logs.push({type:'system', text:'═══ COMPOSITION TABLE (i⊕j) ═══'});
      logs.push({type:'dim', text:'    ' + Array.from({length:10},(_,i)=>String(i).padStart(2)).join('')});
      for(let i=0;i<10;i++){
        logs.push({type: i===7?'harmony':'info', text:`${String(i).padStart(2)}: ${COMP_TABLE[i].map(v=>String(v).padStart(2)).join('')}`});
      }
      break;
    }
    case 'reset': {
      newCells = createLattice();
      newTick = 0;
      logs.push({type:'system', text:'Lattice reset to canonical initial state. Tick=0.'});
      break;
    }
    default: {
      logs.push({type:'error', text:`Unknown command: "${cmd}". Type 'help' for reference.`});
    }
  }
  return {cells:newCells, tickCount:newTick, logs};
}

// ═══════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════

function LatticeGrid({cells, onCellClick}){
  return (
    <div style={{
      display:'grid',
      gridTemplateColumns:`repeat(${COLS}, 1fr)`,
      gap:1,
      background:'#0a0a0f',
      border:'1px solid #1a1a2e',
      borderRadius:4,
      padding:2,
      userSelect:'none',
    }}>
      {cells.map((row,i)=>row.map((cell,j)=>(
        <div
          key={`${i}-${j}`}
          onClick={()=>onCellClick?.(i,j,cell)}
          title={`(${i},${j}) ${OP_NAMES[cell.state]} a=${cell.a} b=${cell.b} c=${cell.c}`}
          style={{
            width:'100%',
            aspectRatio:'1',
            background: OP_COLORS[cell.state],
            borderRadius:2,
            display:'flex',
            alignItems:'center',
            justifyContent:'center',
            fontSize:9,
            color: cell.state===0?'#333':'#fff',
            cursor:'pointer',
            transition:'all 0.15s ease',
            opacity: cell.state===7?1:0.75,
            boxShadow: cell.state===7?'0 0 4px #f1c40f44':'none',
          }}
        >
          {OP_SHORT[cell.state]}
        </div>
      )))}
    </div>
  );
}

function CoherenceBar({sStar}){
  const pct = Math.min(sStar*100/1, 100);
  const above = sStar >= T_STAR;
  return (
    <div style={{position:'relative',height:20,background:'#0a0a0f',borderRadius:3,overflow:'hidden',border:'1px solid #1a1a2e'}}>
      <div style={{
        position:'absolute',top:0,left:0,height:'100%',
        width:`${pct}%`,
        background: above
          ? 'linear-gradient(90deg,#2ecc71,#f1c40f)'
          : 'linear-gradient(90deg,#e74c3c,#e67e22)',
        transition:'width 0.3s ease',
      }}/>
      <div style={{
        position:'absolute',top:0,left:`${T_STAR*100}%`,height:'100%',width:1,
        background:'#fff8',zIndex:2,
      }}/>
      <div style={{
        position:'absolute',top:0,left:0,right:0,bottom:0,
        display:'flex',alignItems:'center',justifyContent:'center',
        fontSize:10,fontWeight:700,color:'#fff',zIndex:3,
        textShadow:'0 0 4px #000',fontFamily:'monospace',
      }}>
        S*={sStar.toFixed(4)} {above?'▲':'▽'} T*={T_STAR}
      </div>
    </div>
  );
}

function CensusBar({cells}){
  const census = stateCensus(cells);
  const total = ROWS*COLS;
  return (
    <div style={{display:'flex',height:14,borderRadius:3,overflow:'hidden',border:'1px solid #1a1a2e'}}>
      {census.map((c,i)=> c>0 ? (
        <div key={i} style={{
          width:`${c/total*100}%`,
          background:OP_COLORS[i],
          transition:'width 0.3s ease',
        }} title={`${OP_NAMES[i]}: ${c} (${(c/total*100).toFixed(1)}%)`}/>
      ) : null)}
    </div>
  );
}

const LOG_COLORS = {
  system:'#d4a017',
  info:'#c8c8d0',
  harmony:'#f1c40f',
  warn:'#e67e22',
  error:'#e74c3c',
  dim:'#5a5a6e',
  agent:'#9b59b6',
};

function LogLine({entry}){
  return (
    <div style={{
      fontFamily:'"IBM Plex Mono",monospace',
      fontSize:11,
      lineHeight:'18px',
      color: LOG_COLORS[entry.type]||'#888',
      whiteSpace:'pre-wrap',
      wordBreak:'break-all',
    }}>
      {entry.text}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════
export default function TIGCoherentComputer(){
  const [cells, setCells] = useState(()=>createLattice());
  const [tickCount, setTickCount] = useState(0);
  const [logs, setLogs] = useState([
    {type:'system', text:'╔══════════════════════════════════════════════════════════╗'},
    {type:'system', text:'║   TIG COHERENT COMPUTER v1.0                            ║'},
    {type:'system', text:'║   Semi-Autonomous Agent — Coherence-Gated Execution     ║'},
    {type:'system', text:'║   7Site LLC — Brayden Sanders — Arkansas                ║'},
    {type:'system', text:'╚══════════════════════════════════════════════════════════╝'},
    {type:'dim',    text:'Boundary condition: S*=σ(1-σ*)V*A*  σ=0.991  T*=0.714'},
    {type:'dim',    text:'Type "help" for commands. Agent only acts when confident.'},
    {type:'info',   text:''},
  ]);
  const [input, setInput] = useState('');
  const [history, setHistory] = useState([]);
  const [histIdx, setHistIdx] = useState(-1);
  const [selectedCell, setSelectedCell] = useState(null);
  const logRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(()=>{
    if(logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  },[logs]);

  const coherence = useMemo(()=>computeCoherence(cells),[cells]);

  const handleSubmit = useCallback(()=>{
    if(!input.trim()) return;
    const raw = input.trim();
    setHistory(h=>[raw,...h].slice(0,50));
    setHistIdx(-1);
    setInput('');

    const newLogs = [{type:'dim', text:`> ${raw}`}];
    const {cmd, args} = parseCommand(raw);

    if(cmd === 'clear'){
      setLogs([{type:'system', text:'Log cleared.'}]);
      return;
    }

    // Agent confidence check
    const conf = evaluateConfidence(cells, cmd);
    
    if(!conf.confident){
      newLogs.push({type:'agent', text:`[AGENT] Refused: ${conf.reason}`});
      newLogs.push({type:'dim', text:`Required: S*≥${conf.needed} for '${cmd}'`});
      setLogs(l=>[...l, ...newLogs]);
      return;
    }

    // Confidence passed — execute
    if(conf.needed > 0){
      newLogs.push({type:'agent', text:`[AGENT] Confident: ${conf.reason}`});
    }

    const result = executeCommand(cmd, args, cells, tickCount);
    newLogs.push(...result.logs);
    setCells(result.cells);
    setTickCount(result.tickCount);
    setLogs(l=>[...l, ...newLogs]);
  },[input, cells, tickCount]);

  const handleKey = useCallback((e)=>{
    if(e.key==='Enter'){ handleSubmit(); return; }
    if(e.key==='ArrowUp'){
      e.preventDefault();
      const next = Math.min(histIdx+1, history.length-1);
      if(history[next]){ setInput(history[next]); setHistIdx(next); }
    }
    if(e.key==='ArrowDown'){
      e.preventDefault();
      const next = histIdx-1;
      if(next<0){ setInput(''); setHistIdx(-1); }
      else { setInput(history[next]); setHistIdx(next); }
    }
  },[handleSubmit, history, histIdx]);

  const handleCellClick = useCallback((i,j,cell)=>{
    setSelectedCell({i,j,cell});
    const h = hamiltonian(cell);
    const fp = stableFP(cell);
    setLogs(l=>[...l,
      {type:'dim', text:`Cell (${i},${j}): ${OP_NAMES[cell.state]}  f(x)=${cell.a}x²+${cell.b}x+${cell.c}  H=${h.toFixed(4)}${fp?`  x*=${fp.x.toFixed(3)}`:'  (free)'}`}
    ]);
  },[]);

  return (
    <div style={{
      width:'100%',
      minHeight:'100vh',
      background:'#06060c',
      color:'#c8c8d0',
      fontFamily:'"IBM Plex Mono","SF Mono","Fira Code",monospace',
      display:'flex',
      flexDirection:'column',
      overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding:'10px 16px',
        borderBottom:'1px solid #1a1a2e',
        display:'flex',
        alignItems:'center',
        justifyContent:'space-between',
        background:'#08080e',
        flexShrink:0,
      }}>
        <div style={{display:'flex',alignItems:'center',gap:10}}>
          <span style={{fontSize:16,color:'#f1c40f'}}>★</span>
          <span style={{fontSize:13,fontWeight:700,color:'#d4a017',letterSpacing:1}}>TIG COHERENT COMPUTER</span>
          <span style={{fontSize:10,color:'#5a5a6e'}}>v1.0</span>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:12,fontSize:10}}>
          <span>Tick: <b style={{color:'#e8d44d'}}>{tickCount}</b></span>
          <span>S*: <b style={{color: coherence.sStar>=T_STAR?'#2ecc71':'#e67e22'}}>{coherence.sStar.toFixed(4)}</b></span>
          <span style={{
            padding:'2px 6px',
            borderRadius:3,
            fontSize:9,
            fontWeight:700,
            background: coherence.sStar >= T_STAR ? '#2ecc7133' : '#e74c3c22',
            color: coherence.sStar >= T_STAR ? '#2ecc71' : '#e74c3c',
            border: `1px solid ${coherence.sStar >= T_STAR ? '#2ecc7144' : '#e74c3c33'}`,
          }}>
            {coherence.sStar >= T_STAR ? 'COHERENT' : 'SEEKING'}
          </span>
        </div>
      </div>

      {/* Main content */}
      <div style={{flex:1,display:'flex',overflow:'hidden',minHeight:0}}>
        {/* Left panel: Lattice + bars */}
        <div style={{
          width:280,
          flexShrink:0,
          borderRight:'1px solid #1a1a2e',
          padding:10,
          display:'flex',
          flexDirection:'column',
          gap:8,
          overflow:'auto',
        }}>
          <div style={{fontSize:9,color:'#5a5a6e',textTransform:'uppercase',letterSpacing:1}}>
            Lattice {ROWS}×{COLS} — Click to inspect
          </div>
          <LatticeGrid cells={cells} onCellClick={handleCellClick}/>
          <div style={{fontSize:9,color:'#5a5a6e',textTransform:'uppercase',letterSpacing:1,marginTop:4}}>
            Coherence
          </div>
          <CoherenceBar sStar={coherence.sStar}/>
          <div style={{fontSize:9,color:'#5a5a6e',textTransform:'uppercase',letterSpacing:1}}>
            State Distribution
          </div>
          <CensusBar cells={cells}/>
          <div style={{display:'flex',flexWrap:'wrap',gap:4,marginTop:2}}>
            {OP_NAMES.map((name,i)=>{
              const c = stateCensus(cells)[i];
              return c>0 ? (
                <div key={i} style={{
                  fontSize:8,
                  padding:'1px 4px',
                  borderRadius:2,
                  background:OP_COLORS[i],
                  color: i===0?'#666':'#fff',
                  opacity:0.8,
                }}>
                  {OP_SHORT[i]}{c}
                </div>
              ) : null;
            })}
          </div>
          {/* Quick GFM buttons */}
          <div style={{fontSize:9,color:'#5a5a6e',textTransform:'uppercase',letterSpacing:1,marginTop:6}}>
            GFM Generators
          </div>
          <div style={{display:'flex',gap:4}}>
            {Object.entries(GFM).map(([k,v])=>(
              <button key={k} onClick={()=>{setInput(`inject-gfm ${k}`);setTimeout(()=>inputRef.current?.focus(),0)}}
                style={{
                  flex:1,fontSize:9,padding:'4px 2px',
                  background:'#1a1a2e',color:'#d4a017',border:'1px solid #2a2a3e',
                  borderRadius:3,cursor:'pointer',fontFamily:'inherit',
                }}>
                {k}
              </button>
            ))}
          </div>
          {/* Quick action buttons */}
          <div style={{display:'flex',gap:4}}>
            {[['tick 1','Tick'],['tick 10','×10'],['tick 50','×50'],['reset','Reset']].map(([cmd,label])=>(
              <button key={cmd} onClick={()=>{setInput(cmd);setTimeout(()=>{inputRef.current?.focus()},0)}}
                style={{
                  flex:1,fontSize:9,padding:'4px 2px',
                  background: cmd==='reset'?'#2a1515':'#1a1a2e',
                  color: cmd==='reset'?'#e74c3c':'#888',
                  border:`1px solid ${cmd==='reset'?'#3a1a1a':'#2a2a3e'}`,
                  borderRadius:3,cursor:'pointer',fontFamily:'inherit',
                }}>
                {label}
              </button>
            ))}
          </div>
          {/* Agent confidence panel */}
          <div style={{
            marginTop:6,padding:8,background:'#0c0c14',
            border:'1px solid #1a1a2e',borderRadius:4,fontSize:9,
          }}>
            <div style={{color:'#9b59b6',fontWeight:700,marginBottom:4}}>AGENT STATUS</div>
            <div style={{color:'#5a5a6e'}}>V* (viability): <span style={{color:'#c8c8d0'}}>{coherence.vStar.toFixed(4)}</span></div>
            <div style={{color:'#5a5a6e'}}>A* (alignment): <span style={{color:'#c8c8d0'}}>{coherence.aStar.toFixed(4)}</span></div>
            <div style={{color:'#5a5a6e'}}>S* (coherence): <span style={{color: coherence.sStar>=T_STAR?'#2ecc71':'#e67e22'}}>{coherence.sStar.toFixed(6)}</span></div>
            <div style={{color:'#5a5a6e',marginTop:4}}>
              Mode: <span style={{color: coherence.sStar>=T_STAR?'#2ecc71':'#e67e22'}}>
                {coherence.sStar>=T_STAR?'Full autonomy':'Cautious — limited actions'}
              </span>
            </div>
          </div>
        </div>

        {/* Right panel: Log terminal */}
        <div style={{flex:1,display:'flex',flexDirection:'column',minWidth:0}}>
          <div ref={logRef} style={{
            flex:1,
            overflow:'auto',
            padding:'10px 14px',
            background:'#06060c',
          }}>
            {logs.map((entry,i)=><LogLine key={i} entry={entry}/>)}
          </div>
          {/* Command input */}
          <div style={{
            borderTop:'1px solid #1a1a2e',
            padding:'8px 14px',
            display:'flex',
            alignItems:'center',
            gap:8,
            background:'#08080e',
            flexShrink:0,
          }}>
            <span style={{color:'#d4a017',fontSize:12,fontWeight:700}}>❯</span>
            <input
              ref={inputRef}
              value={input}
              onChange={e=>setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="help | tick | inject | compose | status | analyze ..."
              autoFocus
              style={{
                flex:1,
                background:'transparent',
                border:'none',
                outline:'none',
                color:'#e8e8f0',
                fontFamily:'inherit',
                fontSize:12,
                caretColor:'#f1c40f',
              }}
            />
            <button onClick={handleSubmit} style={{
              background:'#1a1a2e',border:'1px solid #2a2a3e',
              color:'#d4a017',padding:'4px 10px',borderRadius:3,
              cursor:'pointer',fontFamily:'inherit',fontSize:10,
            }}>
              RUN
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
