// Presentation timing and geometry; no model parameters or experiment changes.
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const t=clamp(x); return t*t*(3-2*t); };

// Narrow, spread into separate lanes, then join: labels never cross each other.
export function tokenAssemblyFrame(index, progress) {
  const x=[543,607,671][index],width=[62,62,54][index],y=99+index*40;
  return {x:550+(x-550)*ease((progress-.18)/.38),
    y:y+(145-y)*ease((progress-.58)/.42),width:164+(width-164)*ease(progress/.28)};
}

// Presentation emphasis only; computational timing and values remain unchanged.
export function computationFocus(stage, phase) {
  const boundaries=stage===1?[.28,.57]:stage===2?[.30,.72]:[.26,.80];
  const first=ease((phase-boundaries[0]+.035)/.07);
  const second=ease((phase-boundaries[1]+.035)/.07);
  return [1-.24*first,.76+.24*first-.24*second,.76+.24*second];
}

// A fixed orthographic tilt of the illustrative plane, not an embedding or metric.
export function latentPlanePoint([x,y]) {
  const c=Math.cos(-.22),s=Math.sin(-.22);
  return [358+96*(c*x-s*y),191+96*.68*(s*x+c*y)];
}

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
