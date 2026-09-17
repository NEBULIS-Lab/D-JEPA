import { attention, descriptor } from './explainer-model.mjs';
const ease=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
export function relationPhase(progress) {
  return {compare:ease(progress/.3),messages:Array.from({length:6},(_,i)=>ease((progress-.3-i*.038)/.17)),head:ease((progress-.72)/.22),step:progress<.3?0:progress<.72?1:2};
}
export function relationMix(candidate,head,sources,progress=1) {
  const weights=attention(head,sources)[candidate],phase=relationPhase(progress);
  const values=weights.map((_,i)=>Array.from({length:8},(_,j)=>2*descriptor(i,head%2,j)-1));
  const mixed=Array.from({length:8},(_,j)=>weights.reduce((sum,w,i)=>sum+w*values[i][j]*phase.messages[i],0));
  return {weights,values,mixed,phase};
}
