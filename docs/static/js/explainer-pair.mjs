// Orthographic, rotatable radial view. Numeric radii are independent of the camera.
export function pairPhase(progress) {
  const smooth = x => { x=Math.max(0,Math.min(1,x));return x*x*(3-2*x); };
  return {
    motion: smooth((progress-.22)/.46),
    outcome: smooth((progress-.68)/.06),
    diagnostic: smooth((progress-.82)/.12),
    step: progress<.22?0:progress<.82?1:2,
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
  const t=Math.max(0,Math.min(1,progress/.22));
  return {t, yawOffset:.65*Math.sin(2*Math.PI*t), shell:.25+.75*Math.min(1,t*3)};
}
export function latentMarkup(pair,yaw,intro=1) {
  const cx=183,cy=181,scale=390;
  const screen=point=>{const p=projectPoint(point,yaw);return [cx+p[0]*scale,cy-p[1]*scale,p[2]];};
  const curve=points=>points.map((p,i)=>{const [x,y]=screen(p);return (i?'L':'M')+x.toFixed(2)+' '+y.toFixed(2);}).join(' ');
  const colors=['var(--gold)','var(--blue)'];
  let s='<defs><radialGradient id="latent-light" cx="32%" cy="27%"><stop offset="0" stop-color="var(--purple)" stop-opacity=".17"/><stop offset="1" stop-color="var(--purple)" stop-opacity=".025"/></radialGradient></defs>';
  s+='<ellipse cx="183" cy="260" rx="87" ry="12" fill="var(--purple)" opacity=".05"/>';
  s+='<circle cx="183" cy="181" r="85" fill="url(#latent-light)" stroke="var(--line)"/>';
  pair.candidates.forEach((c,i)=>{
    const r=c.rms_distance;
    const ring=(kind,angle)=>Array.from({length:73},(_,k)=>{
      const t=k*Math.PI/36;
      return kind==='latitude'?[r*Math.cos(angle)*Math.cos(t),r*Math.sin(angle),r*Math.cos(angle)*Math.sin(t)]
        :[r*Math.cos(t)*Math.cos(angle),r*Math.sin(t),r*Math.cos(t)*Math.sin(angle)];
    });
    const shell=.09+.21*Math.min(1,intro*3);
    for(const angle of [-.65,0,.65])s+='<path d="'+curve(ring('latitude',angle))+'" fill="none" stroke="'+colors[i]+'" opacity="'+shell+'" stroke-width=".8"/>';
    for(const angle of [0,Math.PI/3,2*Math.PI/3])s+='<path d="'+curve(ring('meridian',angle))+'" fill="none" stroke="'+colors[i]+'" opacity="'+shell+'" stroke-width=".8"/>';
  });
  // Axis guide and the two goal-relative directions share the same camera.
  const axes=[[.27,0,0],[0,.27,0],[0,0,.27]];
  for(const p of axes){const [x,y]=screen(p);s+='<path d="M183 181L'+x+' '+y+'" stroke="var(--line)" stroke-dasharray="2 4" fill="none"/>';}
  pairPoints(pair).forEach((p,i)=>{
    const [x,y]=screen(p),label=pair.candidates[i].label;
    const reveal=Math.max(0,Math.min(1,(intro-i*.22)/.45));
    s+='<path data-radius-ray="'+label+'" d="M183 181L'+x+' '+y+'" stroke="'+colors[i]+'" stroke-width="2.3" pathLength="1" stroke-dasharray="1" stroke-dashoffset="'+(1-reveal)+'"/>';
    s+='<g opacity="'+reveal+'">';
    s+='<circle cx="'+x+'" cy="'+y+'" r="'+(11+5*Math.sin(Math.PI*reveal))+'" fill="'+colors[i]+'" opacity=".2"/>';
    s+='<circle data-pair-point="'+label+'" cx="'+x+'" cy="'+y+'" r="5" fill="'+colors[i]+'" stroke="var(--panel)" stroke-width="1.5"/>';
    s+='<text x="'+(x+(i===0?-13:13))+'" y="'+(y-8)+'" text-anchor="'+(i===0?'end':'start')+'" class="svg-label" fill="'+colors[i]+'">'+label+'</text>';
    s+='</g>';
  });
  s+='<circle cx="183" cy="181" r="10" fill="var(--purple)" opacity=".15"/><circle cx="183" cy="181" r="4" fill="var(--accent)"/>';
  s+='<text x="183" y="200" class="svg-small svg-accent" text-anchor="middle">Goal</text>';
  return s;
}
