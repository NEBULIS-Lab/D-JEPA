// Orthographic, rotatable radial view. Numeric radii are independent of the camera.
export const PAIR_INTRO_SECONDS=4;
const ORIGINAL_INTRO_SECONDS=12*.22;
export const PAIR_STAGE_SECONDS=12+PAIR_INTRO_SECONDS-ORIGINAL_INTRO_SECONDS;
// Stretch only comparison; the recorded execution and diagnosis retain their
// original wall-clock durations. The inverse also drives direct-operation jumps.
export function pairTimelinePhase(seconds) {
  return seconds<=PAIR_INTRO_SECONDS
    ? Math.max(0,seconds)/PAIR_INTRO_SECONDS*.22
    : Math.min(1,(seconds-PAIR_INTRO_SECONDS+ORIGINAL_INTRO_SECONDS)/12);
}
export function pairTimelineSeconds(phase) {
  return phase<=.22?Math.max(0,phase)/.22*PAIR_INTRO_SECONDS
    :Math.min(1,phase)*12+PAIR_INTRO_SECONDS-ORIGINAL_INTRO_SECONDS;
}
export function pairPhase(progress) {
  const smooth = x => { x=Math.max(0,Math.min(1,x));return x*x*(3-2*x); };
  return {
    motion: smooth((progress-.22)/.46),
    outcome: smooth((progress-.68)/.06),
    diagnostic: smooth((progress-.82)/.12),
    step: progress<=.22?0:progress<.82?1:2,
  };
}
export function projectPoint(point, yaw, pitch=.35) {
  const [x,y,z]=point, a=x*Math.cos(yaw)+z*Math.sin(yaw), b=-x*Math.sin(yaw)+z*Math.cos(yaw);
  return [a,y*Math.cos(pitch)-b*Math.sin(pitch),y*Math.sin(pitch)+b*Math.cos(pitch)];
}
export function pairPoints(pair) {
  return pair.candidates.map((c,i)=>{
    const theta=(i===0?-1:1)*pair.angle_rad/2;
    return [Math.sin(theta)*c.rms_distance,Math.cos(theta)*c.rms_distance,0];
  });
}
export function latentIntro(progress) {
  const u=Math.max(0,Math.min(1,progress/(.22*.9)));
  const t=u*u*(3-2*u); // Ease in/out, then hold the completed view for 0.4 s.
  return {t, yawOffset:.65*Math.sin(2*Math.PI*t)};
}
// Original illustrative backdrop, NOT exported embeddings, model neighborhoods,
// candidate counts, or success clusters. The measured A/B points are separate.
export function latentBackdrop() {
  const centers=[[-.21,.035,-.06],[.17,.075,.045],[.015,-.025,.17]];
  return centers.flatMap((center,cluster)=>Array.from({length:76},(_,i)=>{
    const u=(i+.5)/76, angle=i*2.399963229728653+cluster*.7;
    const spread=Math.sqrt(u), layer=Math.sin(i*1.71+cluster);
    return {
      point:[center[0]+.108*spread*Math.cos(angle),
        center[1]+.065*spread*Math.sin(angle)+.027*layer,
        center[2]+.085*spread*Math.sin(angle*.73)+.028*layer],
      cluster,
    };
  }));
}
const backdrop=latentBackdrop();
// Sparse local links convey depth, not a measured affinity graph.
const backdropEdges=backdrop.flatMap((a,i)=>{
  if(i%4)return [];
  return backdrop.map((b,j)=>({j,d:Math.hypot(...a.point.map((v,k)=>v-b.point[k]))}))
    .filter(b=>b.j>i&&backdrop[b.j].cluster===a.cluster&&b.d<.065)
    .sort((a,b)=>a.d-b.d).slice(0,2).map(b=>[i,b.j]);
});
export function latentMarkup(pair,yaw,intro=1,animate=true) {
  const t=Math.max(0,Math.min(1,intro)),cx=183,cy=191;
  const drift=animate?Math.sin(Math.PI*t):0;
  const pitch=.43+.09*drift,scale=400*(1+.025*drift);
  const screen=point=>{const p=projectPoint(point,yaw,pitch);return [cx+p[0]*scale,cy-p[1]*scale,p[2]];};
  const n=x=>x.toFixed(2);
  const curve=points=>points.map((p,i)=>{const [x,y]=screen(p);return (i?'L':'M')+n(x)+' '+n(y);}).join(' ');
  const colors=['var(--gold)','var(--blue)'];
  let s='<defs><clipPath id="latent-field-clip"><rect x="18" y="91" width="333" height="176" rx="12"/></clipPath>';
  s+='<radialGradient id="latent-field-glow"><stop stop-color="var(--purple)" stop-opacity=".12"/><stop offset="1" stop-color="var(--purple)" stop-opacity="0"/></radialGradient></defs>';
  s+='<g clip-path="url(#latent-field-clip)" aria-label="Illustrative latent-space context; A and B show measured goal distances">';
  s+='<ellipse cx="183" cy="187" rx="162" ry="84" fill="url(#latent-field-glow)"/>';
  // A receding, open reference plane replaces the enclosing planetary shells.
  const floor=-.105;
  s+='<path d="'+curve([[-.34,floor,-.24],[.34,floor,-.24],[.34,floor,.24],[-.34,floor,.24]])+'Z" fill="var(--purple)" opacity=".025"/>';
  for(let k=-4;k<=4;k++){
    for(const points of [ [[k*.08,floor,-.24],[k*.08,floor,.24]], [[-.34,floor,k*.06],[.34,floor,k*.06]] ]){
      s+='<path data-latent-grid d="'+curve(points)+'" stroke="var(--purple)" stroke-width=".6" opacity="'+(.15-Math.abs(k)*.018)+'" fill="none"/>';
    }
  }
  const projected=backdrop.map(a=>screen(a.point));
  for(const [i,j] of backdropEdges){
    const a=projected[i],b=projected[j];
    s+='<path d="M'+n(a[0])+' '+n(a[1])+'L'+n(b[0])+' '+n(b[1])+'" stroke="var(--purple)" stroke-width=".55" opacity=".17"/>';
  }
  // Painter ordering, depth-dependent size and opacity make the orbit legible.
  projected.map((p,i)=>({p,i})).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,i})=>{
    const depth=Math.max(0,Math.min(1,(p[2]+.28)/.56));
    const sweep=Math.max(0,1-Math.abs(i/228-t)*9)*drift;
    const opacity=.18+.43*depth+.2*sweep,r=.65+1.05*depth+.65*sweep;
    if(i%9===0)s+='<circle cx="'+n(p[0])+'" cy="'+n(p[1])+'" r="'+n(r*3.1)+'" fill="var(--purple)" opacity=".07"/>';
    s+='<circle data-latent-context="'+i+'" cx="'+n(p[0])+'" cy="'+n(p[1])+'" r="'+n(r)+'" fill="var(--accent)" opacity="'+n(opacity)+'"/>';
  });
  // Two thin sections retain an intuitive radius comparison; they are not
  // a learned manifold. Unlike the old globe, their shared plane stays open.
  pair.candidates.forEach((c,i)=>{
    const arc=Array.from({length:49},(_,k)=>{
      const a=-.88+k/48*1.76;
      return [c.rms_distance*Math.sin(a),c.rms_distance*Math.cos(a),0];
    });
    s+='<path d="'+curve(arc)+'" stroke="'+colors[i]+'" stroke-width="1" opacity="'+(.15+.25*t)+'" stroke-dasharray="2 4" fill="none"/>';
  });
  pairPoints(pair).forEach((p,i)=>{
    const [x,y]=screen(p),[fx,fy]=screen([p[0],floor,p[2]]),label=pair.candidates[i].label;
    const reveal=Math.max(0,Math.min(1,(t-i*.22)/.45));
    s+='<path d="M'+n(fx)+' '+n(fy)+'L'+n(x)+' '+n(y)+'" stroke="'+colors[i]+'" opacity=".22" stroke-dasharray="2 4" fill="none"/>';
    s+='<ellipse cx="'+n(fx)+'" cy="'+n(fy)+'" rx="6" ry="2.5" fill="'+colors[i]+'" opacity=".2"/>';
    s+='<path d="M183 191L'+n(x)+' '+n(y)+'" stroke="'+colors[i]+'" stroke-width="7" opacity="'+(.065*reveal)+'" fill="none"/>';
    s+='<path data-radius-ray="'+label+'" d="M183 191L'+n(x)+' '+n(y)+'" stroke="'+colors[i]+'" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="'+(1-reveal)+'"/>';
    if(reveal>0&&reveal<1){
      s+='<circle cx="'+n(cx+(x-cx)*reveal)+'" cy="'+n(cy+(y-cy)*reveal)+'" r="3" fill="'+colors[i]+'"/>';
    }
    s+='<g opacity="'+(.35+.65*reveal)+'">';
    s+='<circle cx="'+n(x)+'" cy="'+n(y)+'" r="'+n(10+5*Math.sin(Math.PI*reveal))+'" fill="'+colors[i]+'" opacity=".13"/>';
    s+='<circle data-pair-point="'+label+'" cx="'+n(x)+'" cy="'+n(y)+'" r="4.5" fill="'+colors[i]+'" stroke="var(--panel)" stroke-width="1.5"/>';
    s+='<text x="'+n(x+(i===0?-12:12))+'" y="'+n(y-8)+'" text-anchor="'+(i===0?'end':'start')+'" class="svg-label" style="fill:'+colors[i]+'" paint-order="stroke" stroke="var(--panel)" stroke-width="3" stroke-linejoin="round">'+label+'</text></g>';
  });
  const [gx,gy]=screen([0,floor,0]);
  s+='<ellipse cx="'+n(gx)+'" cy="'+n(gy)+'" rx="7" ry="2.5" fill="var(--purple)" opacity=".18"/>';
  s+='<path d="M183 191L'+n(gx)+' '+n(gy)+'" stroke="var(--purple)" opacity=".35" stroke-dasharray="2 4"/>';
  s+='<circle cx="183" cy="191" r="11" fill="var(--purple)" opacity=".16"/><circle cx="183" cy="191" r="4" fill="var(--accent)" stroke="var(--panel)" stroke-width="1.5"/>';
  s+='<circle cx="183" cy="191" r="7" fill="none" stroke="var(--accent)" stroke-width=".65" opacity=".4"/>';
  s+='<text x="183" y="210" class="svg-small svg-accent" text-anchor="middle" paint-order="stroke" stroke="var(--panel)" stroke-width="3">Goal</text>';
  return s+'</g>';
}
