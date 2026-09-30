// Presentation timing and geometry; no model parameters or experiment changes.
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const t=clamp(x); return t*t*(3-2*t); };

// A short, clock-driven handoff between adjacent views of the same teaching candidate.
export function canCarryCandidate(from, to, running) {
  return running && from>=1 && from<=3 && to===from+1;
}
export function candidateCarryFrame(from, to, seconds) {
  const t=clamp(seconds/.68),p=t*t*t*(t*(t*6-15)+10);
  return {x:from.x+(to.x-from.x)*p,y:from.y+(to.y-from.y)*p,
    halo: t>0&&t<1 ? Math.sin(Math.PI*t)**2 : 0,done:t===1};
}

export function liftingCues(phase) {
  const arrival=clamp((phase-.78)/.10);
  return {target:ease((phase-.16)/.18),move:liftingProgress(phase),
    arrival:arrival>0&&arrival<1?Math.sin(Math.PI*arrival)**2:0,
    read:ease((phase-.80)/.20)};
}

export function liftingProgress(phase) {
  return ease((phase-.34)/.44);
}

export function liftingStep(phase) {
  return phase <= .16 ? 0 : phase <= .34 ? 1 : phase <= .80 ? 2 : 3;
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
