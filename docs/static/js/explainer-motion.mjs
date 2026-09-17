// Presentation timing and geometry; no model parameters or experiment changes.
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const t=clamp(x); return t*t*(3-2*t); };

export function liftingProgress(phase) {
  return ease((phase-.34)/.44);
}

export function liftingStep(phase) {
  return phase < .16 ? 0 : phase < .34 ? 1 : phase < .80 ? 2 : 3;
}

export function decisionProgress(phase, data) {
  const base=data.baseWinner, next=data.winner;
  const denominator=data.delta[base]-data.delta[next];
  const crossing=base!==next && denominator>0
    ? clamp((data.base[next]-data.base[base])/denominator+.025) : .4;
  if(phase<.12)return 0;
  if(phase<.30)return crossing*ease((phase-.12)/.18);
  if(phase<.48)return crossing;
  return crossing+(1-crossing)*ease((phase-.48)/.30);
}

export function liftingGeometry(data, progress) {
  return data.ordinal.map((rank,id)=>{
    const original=(1+data.base[id]*5)/7, target=rank/7;
    const radius=original+(target-original)*clamp(progress);
    return {id,rank,original,target,radius,cost:radius*radius};
  });
}
