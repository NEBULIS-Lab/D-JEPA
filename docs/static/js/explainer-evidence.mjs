import { descriptor } from './explainer-model.mjs';

// A deterministic, actual subtraction + LayerNorm for the teaching example.
// These vectors are illustrative, not cached model activations.
export function evidenceVector(candidate, source, dimensions=192) {
  const goal=Array.from({length:dimensions},(_,j)=>.5*Math.cos(j*.43+source));
  const future=goal.map((x,j)=>x+2*descriptor(candidate,source,j)-1);
  const delta=future.map((x,j)=>x-goal[j]);
  const mean=delta.reduce((a,b)=>a+b,0)/dimensions;
  const variance=delta.reduce((sum,x)=>sum+(x-mean)**2,0)/dimensions;
  return {goal,future,delta,normalized:delta.map(x=>(x-mean)/Math.sqrt(variance+1e-5))};
}

export function evidencePhase(phase) {
  const ease=x=>{const t=Math.max(0,Math.min(1,x));return t*t*(3-2*t);};
  return {difference:ease((phase-.04)/.23),sort:ease((phase-.28)/.25),pack:ease((phase-.57)/.22),encode:ease((phase-.81)/.14)};
}
