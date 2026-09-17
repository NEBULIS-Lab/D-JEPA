// Architecture locators explain computational roles, not recorded activations.
export const MODEL_MAP_COPY={
  overview:{title:'From observations to candidate futures.',note:'Predictive models supply the futures; decision alignment learns how to compare them.'},
  predictor:{title:'Adapt where future representations are produced.',note:'The last TD-JEPA predictor block and projection supply a complementary proposal.'},
  lifting:{title:'Write the learned order into the future.',note:'Representation lifting receives the predicted future, the goal and the aligned rank.'},
  transport:{title:'Refine corresponding steps in the future.',note:'The Reacher representation study learns bounded updates across five matched steps.'},
};

export function modelMapCue(segment,time){
  const cue=segment.architecture?.find(c=>time>=c.start&&time<c.end);
  if(!cue)return null;
  return {mode:cue.mode,progress:Math.max(0,Math.min(1,(time-cue.start)/Math.max(.1,cue.end-cue.start-.45)))};
}

const label=(x,y,s,cls='map-label',anchor='middle')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${s}</text>`;
const box=(id,x,y,w,h,title,sub='',active=false)=>`<g data-map-node="${id}" class="map-node${active?' map-active':''}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>${label(x+w/2,y+(sub?21:h/2+4),title)}${sub?label(x+w/2,y+38,sub,'map-small'):''}</g>`;
const wire=(id,d,delay=0,active=false,dashed=false)=>`<g><path class="map-wire${dashed?' map-complementary':''}" d="${d}" marker-end="url(#map-arrow)"/>${active?`<path data-map-flow="${id}" data-delay="${delay}" class="map-flow" d="${d}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/><circle data-map-packet="${id}" r="3" class="map-packet" opacity="0"/>`:''}</g>`;

export function modelMapMarkup(mode='overview'){
  const p=mode==='predictor',transport=mode==='transport',realize=mode==='lifting'||transport;
  let s='<title>D-JEPA computational locations</title><desc>Predictor adaptation changes the final predictor block and projection. Relational alignment compares candidate futures. Representation lifting acts on future representations after the learned decision order.</desc>';
  s+='<defs><marker id="map-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1L7 4L1 7" fill="none" stroke="var(--muted)" stroke-width="1.2"/></marker></defs>';
  // The same prediction and goal anchors are retained in every view.
  s+=wire('observation','M116 78H144',0,!realize);
  s+=wire('encoding','M252 78H280',.1,!realize);
  s+=wire('prediction','M566 78H621',.36,true);
  s+=wire('action','M423 26V41',.13,!realize);
  s+=wire('goal-input','M116 174H144',0,realize);
  s+=wire('goal-encoding','M252 174H286',.05,realize);
  s+=box('observation',22,54,94,48,'Observation');
  s+=box('encoder',144,54,108,48,'Encoder','Context features');
  s+=label(423,17,'Candidate actions','map-small');
  s+='<g class="map-predictor"><rect x="280" y="42" width="286" height="84" rx="12"/>';
  s+=label(423,62,p?'TD-JEPA predictor':'Action-conditioned predictor','map-label');
  s+=box('earlier-blocks',291,75,111,37,'Earlier blocks');
  s+=box('final-block',412,75,80,37,'Final block','',p);
  s+=box('projection',502,75,53,37,'Projection','',p)+'</g>';
  s+=box('future',621,54,173,48,p?'Adapted futures':'Predicted futures',transport?'LeWM + TD-JEPA':'Candidate representations',true);
  s+=box('goal-input',22,153,94,42,'Goal');
  s+=box('goal-encoder',144,153,108,42,'Encoder');
  s+=box('goal',286,153,140,42,'Goal representation','',realize);
  if(p){
    // Adaptation is a complementary branch, never a mandatory P -> R chain.
    s+=wire('adapted-cost','M707 102V153',.49,true);
    s+=wire('proposal-goal','M426 174H622',.2,false);
    s+=box('proposal',622,153,172,49,'Native-distance','proposal',true);
    s+=wire('base-to-relation','M226 274H278',.14,false);
    s+=wire('relational-default','M478 274H580',.35,false);
    s+=wire('complementary-proposal','M707 202V248',.68,true,true);
    s+=box('base-futures',22,249,204,50,'Pretrained futures','Relational branch');
    s+=box('relational',278,249,200,50,'Relational alignment','Default decision');
    s+=box('composition',580,248,214,52,'Calibrated composition','Selected action',true);
    s+=label(410,348,'Update the predictor locally. Combine its proposal with the relational default.','map-takeaway');
  }else{
    s+=wire('future-relations','M665 102V134H565V153',.42,!realize);
    s+=wire('goal-relations','M426 174H490',.25,!realize);
    s+=wire('relational-order','M640 179H681',.61,!realize);
    s+=box('relational',490,153,150,58,'Relational alignment','Candidate-set learning',!realize);
    s+=box('rank',681,153,113,58,'Aligned order','Decision structure',realize);
    s+=wire('order-lifting','M737 211V239H535V280',.18,realize);
    s+=wire('future-lifting','M794 78H810V259H604V280',.03,realize);
    if(!transport)s+=wire('goal-lifting','M356 195V238H464V308H490',.08,realize);
    s+=box('lifting',490,280,150,58,transport?'Temporal transport':'Representation lifting',transport?'Bounded step updates':'Rank → future latent',realize);
    s+=wire('native-readout','M640 309H681',.65,realize);
    s+=box('readout',681,280,113,58,transport?'Future latents':'Native distance',transport?'Five refined steps':'Aligned choice',realize);
    if(!transport)s+=wire('goal-distance','M356 195V359H737V338',.55,false);
    s+=label(24,276,transport?'Action identity stays fixed.':realize?'Predictor output is the starting point.':'Predict first. Learn the relations.','map-takeaway','start');
    s+=label(24,300,transport?'The update follows each matching future step.':realize?'The aligned order determines its realization.':'Realize the decision structure in future latents.','map-small','start');
    if(transport)s+=label(24,330,'Reacher · representation study','map-small','start');
  }
  return s;
}

export function createModelMap({pause,getStage,getLifting}){
  const root=document.querySelector('#model-map'),svg=root.querySelector('svg');
  const toggle=document.querySelector('#model-map-toggle');
  let activeMode='',lastStage=-1;
  function draw(mode,progress=1){
    if(mode!==activeMode){
      activeMode=mode;svg.innerHTML=modelMapMarkup(mode);
      root.querySelector('#model-map-title').textContent=MODEL_MAP_COPY[mode].title;
      root.querySelector('#model-map-note').textContent=MODEL_MAP_COPY[mode].note;
      root.querySelectorAll('[data-map-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mapMode===mode)));
    }
    root.hidden=false;root.dataset.mode=mode;
    document.querySelector('#diagram-viewport').classList.add('model-map-open');
    document.querySelector('#detail-visual').setAttribute('inert','');
    document.querySelector('#detail-visual').setAttribute('aria-hidden','true');
    toggle.setAttribute('aria-expanded','true');
    const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    svg.querySelectorAll('[data-map-flow]').forEach(path=>{
      const t=Math.max(0,Math.min(1,(progress-Number(path.dataset.delay))/.28));
      path.setAttribute('stroke-dashoffset',1-t);
      const packet=svg.querySelector(`[data-map-packet="${path.dataset.mapFlow}"]`);
      packet.setAttribute('opacity',!reduced&&t>0&&t<1?'1':'0');
      if(t>0&&t<1){const p=path.getPointAtLength(path.getTotalLength()*t);packet.setAttribute('cx',p.x);packet.setAttribute('cy',p.y);}
    });
    svg.querySelectorAll('.map-active rect').forEach(r=>r.style.setProperty('--map-light',.5+.5*Math.min(1,progress*1.5)));
  }
  function close(){
    root.hidden=true;
    document.querySelector('#diagram-viewport').classList.remove('model-map-open');
    document.querySelector('#detail-visual').removeAttribute('inert');
    document.querySelector('#detail-visual').removeAttribute('aria-hidden');
    toggle.setAttribute('aria-expanded','false');
  }
  function open(mode){pause();draw(mode);root.querySelector(`[data-map-mode="${mode}"]`).focus();}
  toggle.onclick=()=>{if(!root.hidden){pause();close();return;}open(getStage()===3?'predictor':getStage()===4?(getLifting()==='transport'?'transport':'lifting'):'overview');};
  root.querySelector('#model-map-close').onclick=()=>{pause();close();toggle.focus();};
  root.querySelectorAll('[data-map-mode]').forEach(b=>b.onclick=()=>open(b.dataset.mapMode));
  root.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();pause();close();toggle.focus();}});
  return {close,update(stage){
    toggle.hidden=stage===0||stage===5;
    if(lastStage!==stage){close();lastStage=stage;}
  },frame(cue){if(cue)draw(cue.mode,cue.progress);else close();}};
}
