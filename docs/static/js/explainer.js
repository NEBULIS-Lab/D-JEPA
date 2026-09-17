import { evidenceVector, evidencePhase } from './explainer-evidence.mjs';
import { liftingProgress, liftingStep, liftingGeometry, decisionProgress } from './explainer-motion.mjs';
import { sampleState, sceneMarkup, updateScenes } from './explainer-scenes.mjs';
import { CANDIDATES, COSTS, example, attention, descriptor, realizedPoint, transportPoint } from './explainer-model.mjs';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const state = { stage: 0, candidate: 0, sources: 2, bound: 0.2, head: 0, lifting: 'ordinal', progress: 1, playing: false, speed: 1, elapsed: 0, supervision: false, recordCandidate:2, replay:true, comparing:false, hoverCandidate:null };
const stageSeconds = [8, 10, 9, 10, 12, 8];
const totalDuration = stageSeconds.reduce((a,b)=>a+b,0);
let record = null;
let phaseProgress = 0;
let resumeAfterCompare = false;
let currentContext = '';
let previousTime = 0;
let frame = 0;
let slideFrame = 0;

const fmt = (n, digits = 3) => n.toFixed(digits);
const escapeText = text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const text = (x, y, content, cls = 'svg-small', extra = '') => `<text x="${x}" y="${y}" class="${cls}" ${extra}>${escapeText(content)}</text>`;
const rect = (x, y, w, h, fill = 'var(--panel)', stroke = 'var(--line)', radius = 7, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
const line = (x1, y1, x2, y2, stroke = 'var(--line)', extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" ${extra}/>`;
const circle = (cx, cy, r, fill = 'var(--accent)', extra = '') => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" ${extra}/>`;
const path = (d, cls = 'detail-line', extra = '') => `<path d="${d}" class="${cls}" ${extra}/>`;
const tokenMark = (i, x, y, selected = i === state.candidate, size = 11) => '<g data-token="'+i+'">' + circle(x, y, size, selected ? 'var(--purple-soft)' : 'var(--panel)', `stroke="${selected ? 'var(--accent)' : 'var(--line)'}"`) + text(x, y + 3.5, CANDIDATES[i], selected ? 'svg-small svg-accent' : 'svg-small', 'text-anchor="middle"') + '</g>';
const candidateHit = (i, x, y, w, h, contents) => `<g class="candidate-mark" data-candidate="${i}" role="button" tabindex="0" aria-label="Trace candidate ${CANDIDATES[i]}"><rect class="candidate-hit" x="${x}" y="${y}" width="${w}" height="${h}" fill="transparent" rx="5"/>${contents}</g>`;

function tShape(x, y, angle = 0, scale = 1, fill = '#92A4B7', opacity = 1) {
  return `<g transform="translate(${x} ${y}) rotate(${angle}) scale(${scale})" opacity="${opacity}"><path d="M-23 -16H23V-5H6V25H-6V-5H-23Z" fill="${fill}" stroke="${fill === '#92A4B7' ? '#8091A2' : fill}" stroke-width="1.2"/></g>`;
}
function miniatureScene(x, y, w, h, aligned = false) {
  return rect(x, y, w, h, 'var(--scene-bg)', 'var(--line)', 8) +
    tShape(x + w * .65, y + h * .43, 0, .6, '#95EE95', .65) +
    tShape(x + w * (aligned ? .65 : .48), y + h * (aligned ? .43 : .61), aligned ? 0 : -29, .6) +
    circle(x + w * .3, y + h * .78, 4, '#4A75E8');
}

const chapters = [
  { kicker: '01 / THE CANDIDATE SET', title: 'One scene. Several possible futures.', description: 'Each candidate is an action sequence. Pretrained predictive models map the same context and candidate actions to future representations, which a planner compares with the goal.', formula: 'ẑᵢ,₁:ᴴ = F(context, actionᵢ)', fact: 'The relational operator processes the complete candidate set. Six are drawn here to make the computation visible.', visual: 'One context, six candidate futures', caption: 'The T-shaped scenes illustrate alternative futures. They are schematic, not decoded model predictions.' },
  { kicker: '02 / THE PREDICTIVE EVIDENCE', title: 'Different geometries. Shared ordinal evidence.', description: 'Future–goal descriptors retain directional information inside each model. Candidate ranks express preferences on a common scale, forming a token for each possible action.', formula: 'vᵢ = [dᵢᴸ ; dᵢᵀ ; rᵢᴸ ; rᵢᵀ]', fact: 'Two 192-dimensional descriptors + two ranks → 386 dimensions → a shared 64-dimensional encoder.', visual: 'From future–goal descriptors to candidate tokens', caption: 'Small coloured cells stand for descriptor dimensions. Ranks are computed from the illustrative candidate costs.' },
  { kicker: '03 / THE CORE OPERATOR', title: 'Learn the relations between futures.', description: 'Two Transformer layers compare candidates as a set. Shared attention lets each future use evidence from the alternatives; a rank-eight head produces a bounded score correction.', formula: 'δᵢ = ε tanh(Wup tanh(Wdown hᵢ))', fact: '2 layers · 4 attention heads · 64 → 8 → 1 correction head. Candidate permutations preserve the corresponding outputs.', visual: 'Candidate-to-candidate relations', caption: 'Attention and correction values are illustrative. Select a head and trace a candidate to inspect its relations.' },
  { kicker: '04 / THE DECISION RULE', title: 'Refine the choices closest to execution.', description: 'The relational correction adjusts the base ordering. A calibrated margin gate selects the relational winner when its advantage supports a switch; otherwise it keeps the base action.', formula: 'sᵢ = bᵢ + δᵢ   ·   |δᵢ| ≤ ε', fact: 'The bound limits the correction. Move it toward zero to recover the base decision in this example.', visual: 'Base scores → bounded corrections → refined scores', caption: 'Lower scores are preferred. Gold marks the base winner; purple marks the gated aligned choice.' },
  { kicker: '05 / REPRESENTATION LIFTING', title: 'Write the decision into the future.', description: 'Exact ordinal realization assigns each terminal future a goal-relative radius determined by its final rank. Native goal distance then recovers the learned order.', formula: 'z̃ᵢ,ᴴ = zgoal + [πᵢ / (K + 1)] uᵢ', fact: 'PushT exact realization retains the preceding future steps and embeds predictive and relational computation in one self-contained checkpoint.', visual: 'Aligned order → terminal future geometry', caption: 'Radii are shown in RMS units. The nearest realized future has rank 1; directions are retained.' },
  { kicker: '06 / ACTION SELECTION', title: 'The native interface selects the aligned action.', description: 'The planner compares the realized futures with the goal, selects the nearest one and passes its corresponding action sequence to the environment.', formula: 'a* = arg minᵢ ‖z̃ᵢ,ᴴ − zgoal‖² / D', fact: 'The candidate identity connects the predicted future, aligned order and executed action sequence.', visual: 'A single decision, carried through the interface', caption: 'This is the final choice in the illustrative example. Recorded comparisons are available on the project page.' },
];


function evidenceDetail() {
  const d=example(state), selected=state.candidate, names=['LeWM','TD-JEPA','JEPA-WM','DINO-WM'];
  let s='';
  ['Form goal-relative descriptors','Convert costs to ordinal evidence','Assemble and encode each token'].forEach((label,i)=>{
    const x=16+i*247;
    s+='<g data-evidence-step="'+i+'" role="button" tabindex="0" aria-label="'+label+'">';
    s+=rect(x,5,235,36,'var(--surface)','var(--line)',8,'data-evidence-operation="'+i+'"');
    s+=text(x+12,28,String(i+1),'svg-small svg-accent')+text(x+29,28,label,'svg-small');
    s+='</g>';
  });
  s+=text(26,65,'Within each predictive geometry','svg-label');
  for(let source=0;source<2;source++){
    const y=81+source*100, color=source===0?'var(--blue)':'var(--purple)', values=evidenceVector(selected,source);
    s+=rect(18,y,227,91,'var(--surface)','var(--line)',9);
    s+=text(28,y+17,names[source],'svg-small')+text(231,y+17,'Candidate '+CANDIDATES[selected],'svg-tiny svg-accent','text-anchor="end"');
    s+=text(29,y+37,'Future','svg-tiny')+text(29,y+57,'− Goal','svg-tiny')+text(29,y+78,'LN(Δ)','svg-tiny');
    for(let j=0;j<8;j++){
      const x=92+j*17;
      s+=rect(x,y+26,12,13,color,'none',2,'opacity="'+(.22+.65*(values.future[j]+1.5)/3)+'"');
      s+=rect(x,y+46,12,13,'var(--gold)','none',2,'opacity="'+(.3+.55*(values.goal[j]+.5))+'"');
      s+='<rect data-evidence-difference="'+source+'" data-dimension="'+j+'" x="'+x+'" y="'+(y+26)+'" width="12" height="13" rx="2" fill="'+color+'"/>';
    }
  }
  s+=text(27,291,'dᵐᵢ = LN(ẑᵐᵢ,H − zᵐgoal)','svg-mono');
  s+=text(27,308,'8 shown / 192 dimensions per model','svg-tiny');
  s+=line(255,60,255,309);
  s+=text(274,65,'Order within each model','svg-label');
  const step=state.sources===2?117:59, x0=state.sources===2?279:272;
  Object.entries(COSTS).slice(0,state.sources).forEach(([key,costs],source)=>{
    const x=x0+source*step;
    s+=text(x,88,names[source],'svg-small');
    s+='<text x="'+x+'" y="105" class="svg-tiny" data-evidence-heading="'+source+'">cost</text>';
    costs.forEach((cost,id)=>{
      s+='<g data-evidence-rank="'+id+'" data-source-index="'+source+'" data-linked-candidate="'+id+'">';
      s+=rect(x-4,-12,step-8,23,id===selected?'var(--purple-soft)':'var(--surface)','none',4);
      s+=tokenMark(id,x+6,0,id===selected,7);
      s+='<text x="'+(x+step-17)+'" y="4" class="svg-mono" text-anchor="end" data-evidence-value="'+source+'-'+id+'"></text></g>';
    });
  });
  s+=text(274,282,'rᵐᵢ = (rank − 1) / (K − 1)','svg-mono');
  s+=text(274,300,'Lower cost → earlier rank → smaller r','svg-tiny');
  s+=line(513,60,513,309);
  s+=text(533,65,'Token for candidate '+CANDIDATES[selected],'svg-label svg-accent');
  s+=rect(528,80,213,193,'var(--surface)','var(--line)',10);
  const parts=[['dᴸ · 192','var(--blue)'],['dᵀ · 192','var(--purple)'],[state.sources+' ranks','var(--gold)']];
  parts.forEach(([label,color],i)=>{
    s+='<g data-evidence-piece="'+i+'">';
    s+='<rect data-piece-box="'+i+'" width="164" height="26" rx="5" fill="'+color+'" fill-opacity=".2" stroke="'+color+'"/>';
    s+='<text data-piece-label="'+i+'" y="17" class="svg-small" text-anchor="middle">'+label+'</text></g>';
  });
  s+='<g id="evidence-encoder"><path d="M634 175V215" class="trace-line"/>';
  s+=rect(548,220,176,32,'var(--purple-soft)','var(--purple)',7);
  s+=text(636,240,'Shared encoder · '+(state.sources===2?'386':'388')+' → 64','svg-small svg-accent','text-anchor="middle"');
  s+='</g>';
  s+=text(535,292,'192 + 192 + '+state.sources+' = '+(state.sources===2?'386':'388'),'svg-mono');
  s+=text(535,308,'Token segments shown schematically','svg-tiny');
  s+='<g id="evidence-bank">';
  for(let i=0;i<6;i++){
    const x=31+i*122;
    s+='<g data-linked-candidate="'+i+'" data-evidence-output="'+i+'">';
    s+=rect(x,323,109,27,i===selected?'var(--purple-soft)':'var(--surface)',i===selected?'var(--purple)':'var(--line)',7);
    s+=tokenMark(i,x+14,336,i===selected,8);
    s+=text(x+59,340,'64D token','svg-small','text-anchor="middle"');
    s+='</g>';
  }
  s+='</g>';
  return s;
}
function updateEvidence(svg) {
  const phase=evidencePhase(phaseProgress), d=example(state);
  const active=phaseProgress<.28?0:phaseProgress<.57?1:2;
  svg.querySelectorAll('[data-evidence-operation]').forEach(n=>{
    n.setAttribute('fill',Number(n.dataset.evidenceOperation)===active?'var(--purple-soft)':'var(--surface)');
    n.setAttribute('stroke',Number(n.dataset.evidenceOperation)===active?'var(--purple)':'var(--line)');
  });
  for(let source=0;source<2;source++){
    const values=evidenceVector(state.candidate,source), y=81+source*100;
    svg.querySelectorAll('[data-evidence-difference="'+source+'"]').forEach(n=>{
      const j=Number(n.dataset.dimension), normalized=values.normalized[j];
      n.setAttribute('y',y+26+40*phase.difference);
      n.setAttribute('opacity',String(.2+phase.difference*(.25+.45*Math.min(1,Math.abs(normalized)/1.7))));
      n.setAttribute('fill',normalized>=0?(source===0?'var(--blue)':'var(--purple)'):'var(--peach)');
    });
  }
  const entries=Object.entries(COSTS).slice(0,state.sources);
  entries.forEach(([key,costs],source)=>{
    const order=costs.map((value,id)=>({value,id})).sort((a,b)=>a.value-b.value).map(x=>x.id);
    svg.querySelector('[data-evidence-heading="'+source+'"]').textContent=phase.sort<1?'cost ↓':'ordinal r ↓';
    costs.forEach((value,id)=>{
      const row=svg.querySelector('[data-evidence-rank="'+id+'"][data-source-index="'+source+'"]');
      row.setAttribute('transform','translate(0 '+(125+(id+(order.indexOf(id)-id)*phase.sort)*24)+')');
      svg.querySelector('[data-evidence-value="'+source+'-'+id+'"]').textContent=phase.sort<1?value.toFixed(2):d.sourceRanks[key][id].toFixed(1);
    });
  });
  const finalX=[543,607,671], finalW=[62,62,54];
  svg.querySelectorAll('[data-evidence-piece]').forEach(n=>{
    const i=Number(n.dataset.evidencePiece), p=phase.pack;
    n.setAttribute('transform','translate('+(550+(finalX[i]-550)*p)+' '+(99+i*40+(145-(99+i*40))*p)+')');
    const w=164+(finalW[i]-164)*p;
    svg.querySelector('[data-piece-box="'+i+'"]').setAttribute('width',w);
    svg.querySelector('[data-piece-label="'+i+'"]').setAttribute('x',w/2);
    n.style.opacity=String(.12+.88*(i<2?phase.difference:phase.sort));
  });
  svg.querySelector('#evidence-encoder').style.opacity=String(.1+.9*phase.encode);
  svg.querySelectorAll('[data-evidence-output]').forEach(n=>{
    const i=Number(n.dataset.evidenceOutput),p=Math.max(0,Math.min(1,(phase.encode-i*.045)/.775));
    n.style.opacity=String(.08+.92*p);
    n.setAttribute('transform','translate(0 '+(-14*(1-p))+')');
  });
}

function relationsDetail() {
  const weights = attention(state.head, state.sources);
  let s = text(32, 28, 'Candidate tokens', 'svg-label') + text(375, 28, `Attention head ${state.head + 1}`, 'svg-label', 'text-anchor="middle"') + text(646, 28, 'Bounded update', 'svg-label', 'text-anchor="middle"');
  const x0 = 270, y0 = 63, cell = 31;
  for (let i = 0; i < 6; i++) {
    const y = 69 + i * 35;
    s += tokenMark(i, 41, y, i === state.candidate, 12);
    for (let j = 0; j < 8; j++) s += rect(64 + j * 9, y - 8, 6, 16, 'var(--purple)', 'none', 2, `opacity="${.2 + descriptor(i, 0, j) * .7}"`);
    s += path(`M143 ${y}C199 ${y} 202 ${y0 + state.candidate * cell + 14} ${x0 - 19} ${y0 + state.candidate * cell + 14}`, 'attention-edge', `stroke-width="${weights[state.candidate][i] * 10}" opacity="${i === state.candidate ? .9 : .25}"`);
    s += text(x0 - 14, y0 + i * cell + 18, CANDIDATES[i], 'svg-small', 'text-anchor="middle"');
    s += text(x0 + i * cell + 14, y0 - 12, CANDIDATES[i], 'svg-small', 'text-anchor="middle"');
    for (let j = 0; j < 6; j++) {
      const value = weights[i][j];
      s += `<g class="matrix-cell" tabindex="0" data-row="${i}" data-col="${j}" data-weight="${value}"><title>Illustrative attention ${CANDIDATES[i]} to ${CANDIDATES[j]}: ${fmt(value)}</title>`;
      s += rect(x0 + j * cell, y0 + i * cell, 27, 27, 'var(--purple)', i === state.candidate ? 'var(--accent)' : 'none', 4, `opacity="${.2 + value * 1.5}"`);
      if (i === state.candidate) s += text(x0 + j * cell + 13.5, y0 + i * cell + 17, value.toFixed(2), 'svg-tiny', 'text-anchor="middle"');
      s += '</g>';
    }
  }
  s += '<g id="head-output">'+path('M467 156H522', 'trace-line');
  s += rect(535, 83, 199, 167, 'var(--surface)', 'var(--line)', 12);
  s += text(635, 116, 'Shared low-rank head', 'svg-small', 'text-anchor="middle"');
  s += text(635, 150, '64 → 8 → 1', 'svg-title svg-accent', 'text-anchor="middle"');
  s += text(635, 178, 'tanh · bound ε', 'svg-mono', 'text-anchor="middle"');
  s += text(635, 219, `δ${CANDIDATES[state.candidate]} = ${fmt(example(state).delta[state.candidate])}`, 'svg-title svg-accent', 'text-anchor="middle"');
  s += '</g>';
  s += text(365, 285, 'Every candidate can exchange evidence with every other candidate.', 'svg-small', 'text-anchor="middle"');
  return s;
}
function decisionDetail() {
  const d = example(state);
  let s = text(65, 27, 'Candidate', 'svg-small') + text(183, 27, 'Base score b', 'svg-small') + text(373, 27, 'Correction δ', 'svg-small') + text(574, 27, 'Refined score s', 'svg-small');
  for (let i = 0; i < 6; i++) {
    const y = 61 + i * 33;
    if (i === d.winner) s += rect(37, y - 15, 687, 30, 'var(--purple-soft)', 'none', 6);
    s += tokenMark(i, 79, y, i === state.candidate, 10);
    if (i === d.baseWinner) s += circle(106, y, 3, 'var(--gold)');
    s += rect(166, y - 7, Math.max(2, d.base[i] * 140), 14, i === d.baseWinner ? 'var(--gold)' : 'var(--blue)', 'none', 3, 'opacity=".7"');
    s += text(321, y + 4, fmt(d.base[i]), 'svg-mono', 'text-anchor="end"');
    s += line(408, y - 13, 408, y + 13);
    const barW = Math.abs(d.delta[i]) * 240;
    s += rect(d.delta[i] < 0 ? 408 - barW : 408, y - 7, Math.max(.2, barW), 14, 'var(--purple)', 'none', 3);
    s += text(483, y + 4, `${d.delta[i] > 0 ? '+' : ''}${fmt(d.delta[i])}`, 'svg-mono', 'text-anchor="end"');
    s += text(512, y + 4, '=', 'svg-small');
    s += rect(553, y - 7, Math.max(2, d.scores[i] * 108), 14, i === d.winner ? 'var(--purple)' : 'var(--blue)', 'none', 3, 'opacity=".8"');
    s += text(705, y + 4, fmt(d.scores[i]), i === d.winner ? 'svg-mono svg-accent' : 'svg-mono', 'text-anchor="end"');
  }
  s += rect(37, 272, 687, 35, 'var(--surface)', 'var(--line)', 8);
  s += text(54, 294, `Gate: advantage ${fmt(d.advantage)} ${d.admitted ? '>' : '≤'} ${fmt(d.threshold)}  →  ${d.admitted ? 'admit relational winner' : 'keep base winner'}`, 'svg-mono');
  s += text(705, 294, `choose ${CANDIDATES[d.winner]}`, 'svg-label svg-accent', 'text-anchor="end"');
  return s;
}
function liftingDetail() {
  const d = example(state);
  let s = '';
  if (state.lifting === 'transport') {
    s += text(40, 28, 'Five future steps · same candidate', 'svg-label');
    s += text(711, 28, `Candidate ${CANDIDATES[state.candidate]}`, 'svg-small svg-accent', 'text-anchor="end"');
    const points = Array.from({length: 5}, (_, t) => transportPoint(state.candidate, t, state.progress));
    const map = p => [65 + p[0] * 790, 158 - p[1] * 275];
    const base = points.map(p => map(p.source)), refined = points.map(p => map(p.refined));
    s += path(base.map(([x,y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' '), 'detail-line', 'stroke-width="2"');
    s += path(refined.map(([x,y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' '), 'trace-line');
    points.forEach((p, t) => {
      const [x,y] = base[t], [xx,yy] = refined[t];
      s += line(x, y, xx, yy, 'var(--purple)', 'stroke-dasharray="3 3"') + circle(x,y,5,'var(--blue)') + circle(xx,yy,6,'var(--accent)');
      s += text(x, 258, `t${t + 1}`, 'svg-label', 'text-anchor="middle"');
      s += text(x, 281, `β ${fmt(p.beta)}`, 'svg-mono', 'text-anchor="middle"');
    });
    s += text(380, 310, 'Blue: original future   ·   Purple: transported future   ·   |β| ≤ 0.1', 'svg-small', 'text-anchor="middle"');
  } else { return ordinalLiftingDetail(); }
  return s;
}

function ordinalLiftingDetail() {
  const d=example(state), geometry=liftingGeometry(d,state.progress), selected=geometry[state.candidate];
  const phase=state.comparing?0:phaseProgress, active=liftingStep(phase);
  const nearest=[...geometry].sort((a,b)=>a.cost-b.cost)[0].id;
  const labels=['Read aligned rank','Set RMS radius','Rewrite terminal latent','Read native distance'];
  let s='<defs><marker id="lift-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="var(--accent)"/></marker></defs>';
  labels.forEach((label,i)=>{
    const x=16+i*185;
    s+='<g data-lift-step="'+i+'" role="button" tabindex="0" aria-label="'+(i+1)+'. '+label+'">';
    s+=rect(x,5,171,38,i===active?'var(--purple-soft)':'var(--surface)',i===active?'var(--purple)':'var(--line)',9);
    s+=text(x+11,28,String(i+1),i===active?'svg-small svg-accent':'svg-small')+text(x+29,28,label,'svg-small');
    s+='</g>';
    if(i<3)s+=text(x+177,28,'→','svg-tiny');
  });
  s+=text(27,78,'Aligned order π','svg-label');
  d.finalOrder.forEach((id,index)=>{
    const y=106+index*29,chosen=id===state.candidate;
    s+='<g data-linked-candidate="'+id+'">';
    s+=rect(19,y-16,161,29,chosen?'var(--purple-soft)':'var(--surface)','none',6);
    s+=tokenMark(id,37,y-1,chosen,9);
    s+=text(72,y+3,'rank '+(index+1),'svg-mono');
    s+=text(166,y+3,(index+1)+'/7','svg-mono', 'text-anchor="end"');
    s+='</g>';
  });
  s+=text(26,303,'ρᵢ = πᵢ / (K + 1)','svg-mono svg-accent');
  s+=text(26,324,'K = 6 candidates','svg-small');
  s+=text(26,346,'Same candidate identity','svg-tiny');
  const cx=368,cy=187,scale=76;
  s+=text(cx,78,'Same direction · new goal distance','svg-label','text-anchor="middle"');
  [1,2,3,4,5,6].forEach(rank=>{s+=circle(cx,cy,Math.SQRT2*rank/7*scale,'none','stroke="var(--line)" stroke-dasharray="3 5"');});
  s+=line(cx-6,cy,cx+6,cy,'var(--gold)','stroke-width="2"')+line(cx,cy-6,cx,cy+6,'var(--gold)','stroke-width="2"');
  s+=text(cx-28,cy+18,'goal','svg-tiny svg-gold');
  for(const g of geometry){
    const i=g.id,from=realizedPoint(i,1+d.base[i]*5),to=realizedPoint(i,d.ordinal[i]);
    const point=[from[0]+(to[0]-from[0])*state.progress,from[1]+(to[1]-from[1])*state.progress];
    const x=cx+point[0]*scale,y=cy+point[1]*scale;
    s+='<g data-linked-candidate="'+i+'" data-latent="'+i+'" data-candidate="'+i+'" role="button" tabindex="0" aria-label="Trace future '+CANDIDATES[i]+'">';
    s+=line(cx,cy,cx+to[0]*scale,cy+to[1]*scale,i===state.candidate?'var(--purple)':'var(--line)','stroke-dasharray="3 4"');
    s+=circle(cx+from[0]*scale,cy+from[1]*scale,4,'none','stroke="var(--blue)"');
    s+=circle(cx+to[0]*scale,cy+to[1]*scale,7,'none','stroke="var(--purple)" opacity=".4"');
    if(i===state.candidate){
      s+=line(cx+from[0]*scale,cy+from[1]*scale,cx+to[0]*scale,cy+to[1]*scale,'var(--accent)','stroke-width="2" marker-end="url(#lift-arrow)"');
    }
    s+=circle(x,y,i===state.candidate?6:4,i===state.candidate?'var(--accent)':'var(--blue)','data-lift-point="'+i+'"');
    s+=text(x+9,y+(i===state.candidate?4:-8),CANDIDATES[i],i===state.candidate?'svg-label svg-accent':'svg-small');
    s+='</g>';
  }
  s+=text(cx,295,'Open: original · ring: target · filled: current','svg-tiny','text-anchor="middle"');
  for(let step=0;step<5;step++){
    const x=245+step*49;
    s+=rect(x,314,39,21,step===4?'var(--purple-soft)':'var(--surface)',step===4?'var(--purple)':'var(--line)',5);
    s+=text(x+19.5,328,'t'+(step+1),step===4?'svg-small svg-accent':'svg-small','text-anchor="middle"');
    if(step<4)s+=text(x+43,328,'·','svg-tiny');
  }
  s+=text(cx,351,'t1–t4 retained · terminal t5 rewritten','svg-tiny','text-anchor="middle"');
  s+=rect(553,66,190,242,'var(--surface)','var(--line)',10);
  s+=text(567,93,'Candidate '+CANDIDATES[state.candidate],'svg-title svg-accent');
  s+=text(567,121,'Aligned rank  π = '+selected.rank,'svg-mono');
  s+=text(567,143,'Target  ρ = '+selected.rank+'/7 = '+fmt(selected.target),'svg-mono');
  s+=line(566,155,730,155);
  const rows=[['Original radius',selected.original],['Current radius',selected.radius],['Native cost',selected.cost]];
  rows.forEach(([label,value],index)=>{
    const y=181+index*25;
    s+=text(567,y,label,'svg-small');
    s+=text(730,y,fmt(value),'svg-mono svg-accent','text-anchor="end" data-lift-readout="'+index+'"');
  });
  s+=line(566,248,730,248);
  s+=text(567,270,'Nearest future now','svg-small');
  s+=text(724,289,CANDIDATES[nearest],'svg-title svg-accent','text-anchor="end" data-native-choice="true"');
  s+=text(567,291,'arg min native distance','svg-tiny');
  s+=text(647,329,'Native cost = RMS radius²','svg-small','text-anchor="middle"');
  s+=text(647,348,'Planning interface retained','svg-tiny','text-anchor="middle"');
  return s;
}

function recordedDetail() {
  if (!record) return text(380,170,'Loading recorded trajectories…','svg-label','text-anchor="middle"');
  let s='';
  record.candidates.forEach((candidate,i)=>{
    const x=20+i*248, size=224, selected=i===state.recordCandidate;
    s+='<g data-record-panel="'+i+'">';
    s+=text(x+112,24,candidate.method,selected?'svg-title svg-accent':'svg-title','text-anchor="middle"');
    s+=sceneMarkup(record,i,x,43,size,'stage'+state.stage);
    s+=rect(x,43,size,size,'none',selected?'var(--purple)':'var(--line)',10,'stroke-width="'+(selected?2:1)+'"');
    s+=text(x+10,288,'Candidate '+candidate.id,'svg-small');
    s+=text(x+size-10,288,'25 actions','svg-tiny','text-anchor="end"');
    s+='<g data-outcome="'+i+'" opacity="0">';
    s+=circle(x+12,315,3,candidate.success?'#8EAD7D':'#C96E66');
    s+=text(x+24,319,candidate.success?'Goal reached':'Goal not reached',candidate.success?'svg-small svg-accent':'svg-small');
    s+='</g>';
    s+='<rect class="record-hit" data-record-candidate="'+i+'" role="button" tabindex="0" aria-label="Inspect '+candidate.method+' candidate '+candidate.id+'" x="'+x+'" y="43" width="'+size+'" height="'+size+'" rx="10" fill="transparent"/>';
    s+='</g>';
  });
  s+=text(380,351,'Same start and goal · recorded physical trajectories · 0.00–2.50 s','svg-small','text-anchor="middle" data-record-caption="true"');
  return s;
}
function schematicScenes() {
  let s=text(28,23,'One observation and goal → six alternative action sequences','svg-label');
  for(let i=0;i<6;i++){
    const x=28+(i%3)*243,y=42+Math.floor(i/3)*152;
    const cell=rect(x,y,218,133,'var(--scene-bg)',i===state.candidate?'var(--purple)':'var(--line)',10)
      +tShape(x+143,y+60,0,.8,'#95EE95',.75)
      +'<g data-toy-object="'+i+'"></g><circle data-toy-pusher="'+i+'" r="5" fill="#4A75E8"/>'
      +tokenMark(i,x+17,y+17,i===state.candidate,10);
    s+=candidateHit(i,x,y,218,133,cell);
  }
  s+=text(380,354,'Schematic trajectories for the six-candidate computation below.','svg-small','text-anchor="middle"');
  return s;
}
function movingDecision() {
  const data=example(state);
  let s=text(35,25,'Original order','svg-label svg-gold')+text(263,25,'Bounded correction','svg-label')+text(553,25,'Aligned order','svg-label svg-accent');
  const initial=[...CANDIDATES.keys()].sort((a,b)=>data.base[a]-data.base[b]);
  initial.forEach((id,rank)=>{
    const y=61+rank*39;
    s+=tokenMark(id,49,y,id===state.candidate,10);
    s+=rect(70,y-6,Math.max(3,data.base[id]*90),12,id===data.baseWinner?'var(--gold)':'var(--blue)','none',3);
    s+=text(199,y+4,fmt(data.base[id]),'svg-mono','text-anchor="end"');
    s+='<path data-rank-path="'+id+'" data-linked-candidate="'+id+'" class="detail-line"/>';
    s+='<g data-score-row="'+id+'" data-linked-candidate="'+id+'">';
    s+=rect(544,-17,184,33,id===data.winner?'var(--purple-soft)':'var(--surface)','none',7);
    s+=tokenMark(id,561,0,id===state.candidate,10);
    s+='<rect data-aligned-bar="'+id+'" x="581" y="-6" width="4" height="12" rx="3" fill="'+(id===data.winner?'var(--purple)':'var(--blue)')+'"/>';
    s+='<text data-aligned-value="'+id+'" x="715" y="4" class="svg-mono" text-anchor="end"></text></g>';
    s+=rect(321,y-12,88,24,'var(--panel)','none',6);
    s+=text(365,y+4,(data.delta[id]>0?'+':'')+fmt(data.delta[id]),'svg-mono svg-accent','text-anchor="middle"');
  });
  s+=rect(26,314,707,35,'var(--surface)','var(--line)',8);
  s+='<text id="gate-message" x="380" y="336" class="svg-mono" text-anchor="middle"></text>';
  return s;
}
function contextStrip() {
  const labels=['Action sequences','Future–goal evidence','Candidate relations','Aligned order','Representation lifting','Native-distance selection'];
  const current=labels[state.stage],before=labels[state.stage-1]||'Observation + goal',after=state.stage===4?(state.lifting==='ordinal'?'Native-distance planning':'Representation diagnostic'):(labels[state.stage+1]||'Environment');
  $('#architecture').innerHTML=text(12,26,before,'svg-small')+path('M174 22H240','detail-line')
    +rect(250,5,264,33,'var(--purple-soft)','none',16)
    +text(382,26,current,'svg-label svg-accent','text-anchor="middle"')
    +path('M524 22H581','detail-line')+text(744,26,after,'svg-small','text-anchor="end"');
}
function isRecorded(){return (state.stage===0||state.stage===5)&&state.replay;}
function updateViewport(){
  const compact=matchMedia('(max-width:900px)').matches;
  $('#diagram-viewport').classList.toggle('recorded',isRecorded());
  $('#detail-visual').setAttribute('viewBox',compact&&isRecorded()?(14+state.recordCandidate*248)+' 8 236 350':'0 0 760 360');
  $$('[data-record-panel]').forEach(n=>n.style.display=compact&&Number(n.dataset.recordPanel)!==state.recordCandidate?'none':'');
  const caption=$('[data-record-caption]');if(caption)caption.style.display=compact?'none':'';
}
function chapterCopy() {
  const chapter={...chapters[state.stage]};
  if(isRecorded()){
    chapter.kicker=state.stage===0?'01 / POSSIBLE ACTIONS':'06 / RECORDED EXECUTION';
    chapter.title=state.stage===0?'One start. Three different futures.':'The choice changes the outcome.';
    chapter.description=state.stage===0
      ?'Watch three stored action sequences unfold from the same PushT state. The pusher moves, makes contact and rotates the object toward the green goal.'
      :'These are the original action selections from TD-JEPA, LeWM and D-JEPA. Play them together to see how the selected action changes the physical future.';
    chapter.formula='Same observation + goal → different actions';
    chapter.fact='Start 176. Three method-selected candidates from the original pool; full 25-control-action sequences over 2.50 seconds. Their recorded states drive the animation.';
    chapter.visual='PushT · synchronized physical replay';
    chapter.caption='Recorded simulator states. The internal computation stages use a separate, labelled six-candidate teaching example.';
    if(record){
      const candidate=record.candidates[state.recordCandidate];
      chapter.formula='Candidate '+candidate.id+' · '+candidate.method;
      chapter.fact+=' Full-pool ranks for this candidate: LeWM '+candidate.ranks[0]+', TD-JEPA '+candidate.ranks[1]+', D-JEPA '+candidate.ranks[2]+'.';
    }
  } else if(state.stage===0||state.stage===5){
    chapter.visual='Six schematic candidate trajectories';
    chapter.caption='Illustrative motion for the teaching example; switch to recorded PushT to view actual executions.';
  }
  if(state.stage===1){
    chapter.description='Subtract each model’s goal representation from its predicted future, then normalize the descriptor. Sort candidate costs within each source and convert rank to a shared ordinal scale. Concatenate these signals and encode each candidate token.';
    chapter.caption='Click the three operations or scrub their progress. Blue/purple: source descriptors; peach: negative components; gold: ordinal evidence.';
    chapter.fact='Descriptors are LayerNorm-normalized future–goal differences. Displayed vectors use the actual subtraction and normalization on synthetic values; costs and ranks are deterministic teaching data. Four sources add JEPA-WM and DINO-WM ranks while retaining the two 192D descriptors.';
  }
  if(state.stage===1&&state.sources===4){
    chapter.formula='vᵢ = [dᵢᴸ ; dᵢᵀ ; rᵢᴸ ; rᵢᵀ ; rᵢᴶ ; rᵢᴰ]';
    chapter.fact='388-dimensional tokens. The four-geometry configuration has separately learned parameters and a JEPA-WM base.';
  }
  if(state.stage===2){
    chapter.description='Trace one candidate through set-wise attention. Inspect a matrix cell to see which alternative contributes evidence, then follow the bounded correction head.';
    chapter.caption='Schematic attention and correction outputs illustrate distinct operations; they are not recorded checkpoint activations.';
  }
  if(state.stage===3){
    chapter.description='The bounded update changes the closest alternatives. Watch the ranking reorder, then inspect whether the margin gate admits the relational winner.';
    chapter.caption='Drag the step progress to see score correction and reordering. Gold: base choice. Purple: aligned choice.';
  }
  if(state.stage===4&&state.lifting==='ordinal'){
    chapter.title='Turn the learned rank into a future latent.';
    chapter.description='After relational selection, lifting places each terminal future at its rank-defined distance from the goal. Its direction and earlier future steps are retained; native distance reads out the aligned order.';
    chapter.visual='Representation lifting · between aligned ranks and native-distance planning';
    chapter.caption='Follow a candidate: rank → radius → terminal latent → native distance. Click a numbered operation to inspect it.';
  }
  if(state.stage===4&&state.lifting==='transport'){
    chapter.title='Refine the future, step by step.';
    chapter.description='A time-conditioned network learns a bounded displacement between corresponding predictive futures. Action identity and future-step identity stay fixed.';
    chapter.formula='z̃ᵀᵢ,ₜ = ẑᵀᵢ,ₜ + βᵢ,ₜ · uᵢ,ₜ';
    chapter.fact='Reacher: five future steps, a 5 → 32 → 1 coefficient network and a 0.1 bound. Physical decisions use relational selection; transported latents are representation diagnostics.';
    chapter.visual='Bounded temporal transport · five future steps';
    chapter.caption='Schematic representation diagnostic. Blue: original future. Purple: transported future.';
  }
  return chapter;
}
function options() {
  const panel=$('#stage-options');
  panel.innerHTML='';
  if(state.stage===0||state.stage===5){
    panel.innerHTML='<div class="segmented" role="group" aria-label="Scene source"><button data-replay="true" aria-pressed="'+state.replay+'">Recorded PushT</button><button data-replay="false" aria-pressed="'+!state.replay+'">Teaching example</button></div>';
  }
  if(state.stage===2){
    panel.innerHTML='<label>Attention head <select id="attention-head">'+[0,1,2,3].map(h=>'<option value="'+h+'" '+(h===state.head?'selected':'')+'>'+ (h+1)+' / 4</option>').join('')+'</select></label>';
    $('#attention-head').onchange=e=>{state.head=Number(e.target.value);renderDiagram();};
  }
  if(state.stage===3){
    panel.innerHTML='<details><summary>Complementary predictor adaptation +</summary><p>The final TD-JEPA predictor block and projection can supply a native-distance proposal. Calibrated composition combines it with the relational default.</p></details>';
  }
  if(state.stage===4){
    panel.innerHTML='<div class="segmented" role="group" aria-label="Representation mechanism"><button data-lifting="ordinal" aria-pressed="'+(state.lifting==='ordinal')+'">Ordinal realization</button><button data-lifting="transport" aria-pressed="'+(state.lifting==='transport')+'">Temporal transport</button></div>';
  }
  const localLabel=isRecorded()?'Execution time':state.stage===4?'Transformation':'Step progress';
  panel.insertAdjacentHTML('beforeend','<label class="lifting-control"><span>'+localLabel+' <output id="step-value" for="step-progress">0%</output></span><input id="step-progress" type="range" min="0" max="1" step=".005" value="0" aria-label="Current step progress"></label><button id="replay-step" class="quiet-button">Replay this step ↻</button>');
  $('#step-progress').oninput=e=>{stop();state.elapsed=Number(e.target.value)*stageSeconds[state.stage];updateAnimation();};
  $('#replay-step').onclick=()=>{stop();state.elapsed=0;play(true);};
  if(state.stage===3||state.stage===4){
    panel.insertAdjacentHTML('beforeend','<button id="compare-before" class="compare-button" aria-pressed="false">Hold to see before alignment</button>');
    const compare=$('#compare-before');
    compare.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();compare.setPointerCapture(e.pointerId);compareBefore(true);};
    compare.onpointerup=()=>compareBefore(false);
    compare.onpointercancel=()=>compareBefore(false);
    compare.onlostpointercapture=()=>compareBefore(false);
    compare.onkeydown=e=>{if((e.code==='Space'||e.key==='Enter')&&!e.repeat){e.preventDefault();compareBefore(true);}};
    compare.onkeyup=e=>{if(e.code==='Space'||e.key==='Enter'){e.preventDefault();compareBefore(false);}};
    compare.onblur=()=>compareBefore(false);
  }
}
function compareBefore(active){
  if(state.comparing===active)return;
  if(active){resumeAfterCompare=state.playing;stop();}
  state.comparing=active;
  const button=$('#compare-before');
  if(button){button.setAttribute('aria-pressed',String(active));button.textContent=active?'Before alignment · release to return':'Hold to see before alignment';}
  updateAnimation();
  if(!active&&resumeAfterCompare){resumeAfterCompare=false;play();}
}
function applyLinkedFocus(){
  const id=state.hoverCandidate;
  $$('#detail-visual [data-linked-candidate], #detail-visual [data-token]').forEach(n=>{
    const value=Number(n.dataset.linkedCandidate??n.dataset.token);
    n.classList.toggle('trace-muted',id!==null&&value!==id);
    n.classList.toggle('trace-focused',id!==null&&value===id);
  });
  $$('#detail-visual .matrix-cell').forEach(n=>{
    n.classList.toggle('trace-muted',id!==null&&Number(n.dataset.row)!==id&&Number(n.dataset.col)!==id);
  });
}
function renderCandidateControls() {
  const panel=$('#candidate-controls');
  const context=isRecorded()?'recorded':'schematic';
  if(currentContext!==context){
    panel.innerHTML='<span>Trace</span>'+(context==='recorded'&&record
      ?record.candidates.map((c,i)=>'<button data-record-candidate="'+i+'" aria-label="Inspect '+c.method+' candidate '+c.id+'">'+c.method+'</button>').join('')
      :CANDIDATES.map((id,i)=>'<button data-candidate="'+i+'" aria-label="Trace candidate '+id+'">'+id+'</button>').join(''));
    currentContext=context;
  }
  $$('[data-record-candidate]').filter(n=>n.tagName==='BUTTON').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.recordCandidate)===state.recordCandidate)));
  $$('#candidate-controls [data-candidate]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.candidate)===state.candidate)));
}
function renderDiagram(transition=false) {
  const oldTokens=new Map([...$('#detail-visual').querySelectorAll('[data-token]')].map(n=>[n.dataset.token,n.getBoundingClientRect()]));
  let renderer=state.stage===0||state.stage===5?(isRecorded()?recordedDetail:schematicScenes)
    :[null,evidenceDetail,relationsDetail,movingDecision,liftingDetail][state.stage];
  $('#detail-visual').innerHTML=renderer();
  updateViewport();
  if(transition&&!reducedMotion.matches){
    $('#detail-visual').getAnimations().forEach(a=>a.cancel());
    $('#detail-visual').animate([{opacity:.1,transform:'translateY(9px)'},{opacity:1,transform:'translateY(0)'}],{duration:420,easing:'ease-out'});
  }
  $('#matrix-tooltip').hidden=true;
  updateAnimation();
  if(transition&&!reducedMotion.matches){
    const scale=$('#detail-visual').getScreenCTM().a;
    const seen=new Set();
    $$('#detail-visual [data-token]').forEach(node=>{
      const key=node.dataset.token,from=oldTokens.get(key);if(!from||seen.has(key))return;seen.add(key);
      const to=node.getBoundingClientRect();
      node.animate([{transform:'translate('+((from.x-to.x)/scale)+'px,'+((from.y-to.y)/scale)+'px)'},{transform:'translate(0,0)'}],{duration:600,easing:'cubic-bezier(.2,.7,.25,1)'});
    });
  }
}
function render(transition=false) {
  const c=chapterCopy();
  document.documentElement.dataset.stage=state.stage;
  $('#detail-kicker').textContent=c.kicker;$('#detail-title').textContent=c.title;
  $('#detail-description').textContent=c.description;$('#detail-formula').textContent=c.formula;
  $('#detail-fact').textContent=c.fact;$('#focus-label').textContent=c.visual;
  $('#visual-caption').textContent=c.caption;
  $('#data-badge').textContent=isRecorded()?'RECORDED EXECUTION':'ILLUSTRATIVE COMPUTATION';
  $('#detail-visual').setAttribute('aria-label',c.visual);
  $('#experiment-controls').hidden=isRecorded()||state.stage===0||state.stage===5||state.stage===4;
  $('#supervision-toggle').hidden=isRecorded();
  $('#model-note').textContent=isRecorded()?'The reconstruction follows recorded positions and angles; interpolation only smooths playback.':state.sources===2?'Two 192-dimensional descriptors and two ranks form a 386-dimensional token.':'The four-source configuration uses 388 dimensions and its own learned parameters.';
  $$('#chapters button').forEach(b=>{
    const i=Number(b.dataset.stage);
    b.classList.toggle('past',i<state.stage);
    if(i===state.stage)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');
  });
  $$('[data-sources]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.sources)===state.sources)));
  $('#previous').disabled=state.stage===0;$('#next').disabled=state.stage===5;
  const d=example(state);
  $('#base-choice').textContent=CANDIDATES[d.baseWinner];$('#aligned-choice').textContent=CANDIDATES[d.winner];
  $('#strength-value').value=state.bound.toFixed(2);
  contextStrip();options();renderCandidateControls();renderDiagram(transition);
}
function smooth(t){t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);}
function physicalProgress(){return smooth((phaseProgress-.08)/.8);}
function updateAnimation() {
  phaseProgress=Math.min(1,state.elapsed/stageSeconds[state.stage]);
  state.progress=state.comparing?0:state.stage===4?liftingProgress(phaseProgress):smooth((phaseProgress-.12)/.74);
  const svg=$('#detail-visual');
  if(isRecorded()&&record){
    const p=physicalProgress();
    updateScenes(svg,record,p);
    $$('[data-outcome]').forEach(n=>n.setAttribute('opacity',p>=.999?1:0));
  } else if(state.stage===0||state.stage===5){
    const p=physicalProgress(), endings=[[143,60,0],[103,89,-25],[134,69,8],[84,102,35],[168,91,60],[121,109,-50]];
    endings.forEach(([ex,ey,a],i)=>{
      const x=28+(i%3)*243,y=42+Math.floor(i/3)*152,move=smooth((p-.2)/.8);
      const bx=x+85+(ex-85)*move,by=y+90+(ey-90)*move;
      const body=svg.querySelector('[data-toy-object="'+i+'"]');
      if(body)body.innerHTML=tShape(bx,by,-30+(a+30)*move,.8);
      const ball=svg.querySelector('[data-toy-pusher="'+i+'"]');
      if(ball){ball.setAttribute('cx',x+52+(bx-x-20-52)*smooth(p/.75));ball.setAttribute('cy',y+113+(by-y+20-113)*smooth(p/.75));}
    });
  } else if(state.stage===1){
    updateEvidence(svg);
  } else if(state.stage===2){
    const head=svg.querySelector('#head-output');
    if(head)head.style.opacity=String(.12+.88*smooth((phaseProgress-.62)/.24));
    const edges=[...svg.querySelectorAll('.attention-edge')];
    edges.forEach((n,i)=>{
      const p=smooth((phaseProgress-.12-i*.035)/.32);
      n.style.strokeDasharray=n.getTotalLength();
      n.style.strokeDashoffset=(1-p)*n.getTotalLength();
      n.style.animation='none';
    });
    [...svg.querySelectorAll('.matrix-cell')].forEach(n=>{
      const row=Number(n.dataset.row),p=smooth((phaseProgress-.06-row*.065)/.3);
      n.style.opacity=String(.2+.8*p);
    });
  } else if(state.stage===3){
    const d=example(state),initial=[...CANDIDATES.keys()].sort((a,b)=>d.base[a]-d.base[b]);
    const p=state.comparing?0:decisionProgress(phaseProgress,d);
    initial.forEach((id,i)=>{
      const from=61+i*39,to=61+d.finalOrder.indexOf(id)*39,y=from+(to-from)*p;
      const row=svg.querySelector('[data-score-row="'+id+'"]');
      if(!row)return;
      row.setAttribute('transform','translate(0 '+y+')');
      const path=svg.querySelector('[data-rank-path="'+id+'"]');
      path.setAttribute('d','M211 '+from+'C285 '+from+' 451 '+y+' 535 '+y);
      path.style.stroke=id===state.candidate?'var(--purple)':'var(--line)';
      path.style.strokeWidth=id===state.candidate?'2':'1';
      svg.querySelector('[data-aligned-bar="'+id+'"]').setAttribute('width',Math.max(2,(d.base[id]+d.delta[id]*p)*80));
      svg.querySelector('[data-aligned-value="'+id+'"]').textContent=fmt(d.base[id]+d.delta[id]*p);
    });
    const node=svg.querySelector('#gate-message');
    if(node){
      const scores=d.base.map((b,i)=>b+d.delta[i]*p);
      const switched=d.winner!==d.baseWinner&&scores[d.winner]<scores[d.baseWinner];
      node.textContent=state.comparing?'Before alignment · base winner '+CANDIDATES[d.baseWinner]
        :p>=.995?'Gate: '+fmt(d.advantage)+(d.admitted?' > ':' ≤ ')+fmt(d.threshold)+' → '+(d.admitted?'admit '+CANDIDATES[d.winner]:'keep '+CANDIDATES[d.baseWinner])
        :switched?'Boundary switch: '+CANDIDATES[d.winner]+' '+fmt(scores[d.winner])+' < '+CANDIDATES[d.baseWinner]+' '+fmt(scores[d.baseWinner])+' · inspect the two alternatives'
        :'Base choice '+CANDIDATES[d.baseWinner]+' · apply the bounded correction';
      node.classList.toggle('svg-accent',switched&&!state.comparing);
    }
    if(phaseProgress>=.30&&phaseProgress<.48&&!state.comparing){
      svg.querySelectorAll('[data-score-row]').forEach(n=>n.classList.toggle('boundary-muted',![d.winner,d.baseWinner].includes(Number(n.dataset.scoreRow))));
    }else svg.querySelectorAll('[data-score-row]').forEach(n=>n.classList.remove('boundary-muted'));
  } else if(state.stage===4){
    svg.innerHTML=liftingDetail();
  }
  applyLinkedFocus();
  $('#step-progress').value=phaseProgress;
  $$('input[type=range]').forEach(input=>input.style.setProperty('--fill',100*(Number(input.value)-Number(input.min))/(Number(input.max)-Number(input.min))+'%'));
  $('#step-value').value=isRecorded()?(physicalProgress()*2.5).toFixed(2)+' s':Math.round(phaseProgress*100)+'%';
  const seconds=stageSeconds.slice(0,state.stage).reduce((a,b)=>a+b,0)+state.elapsed;
  $('#tour-seek').value=seconds;
  $('#tour-seek').style.setProperty('--fill',100*seconds/totalDuration+'%');
  const stamp=t=>'00:'+String(Math.floor(t)).padStart(2,'0');
  $('#tour-time').value=stamp(seconds)+' / '+stamp(totalDuration);
  $('#timeline-label').textContent=state.playing?'Playing · '+$('#chapters [aria-current]').textContent.replace('→','').trim():'Drag to explore · '+(isRecorded()?'recorded motion':'computation');
}
function selectStage(stage,manual=true) {
  if(manual)stop();
  state.comparing=false;resumeAfterCompare=false;state.hoverCandidate=null;
  state.stage=Math.max(0,Math.min(5,stage));state.elapsed=0;state.progress=0;
  render(true);
  const tab=$('#chapters [data-stage="'+state.stage+'"]');
  tab.scrollIntoView({block:'nearest',inline:'nearest',behavior:'instant'});
}
function stop() {
  state.playing=false;cancelAnimationFrame(frame);document.body.classList.remove('running');
  $('#play-symbol').textContent='▶';$('#play-label').textContent='Play tour';
  $('#play').setAttribute('aria-label','Play guided tour');
}
function play(stepOnly=false) {
  if(state.playing){stop();return;}
  if(state.elapsed>=stageSeconds[state.stage]){
    if(stepOnly){state.elapsed=0;}else selectStage(state.stage===5?0:state.stage+1,false);
  }
  state.playing=true;previousTime=performance.now();document.body.classList.add('running');
  $('#play-symbol').textContent='Ⅱ';$('#play-label').textContent='Pause';$('#play').setAttribute('aria-label','Pause guided tour');
  function tick(now){
    if(!state.playing)return;
    state.elapsed+=Math.min(.5,(now-previousTime)/1000)*state.speed;previousTime=now;
    if(state.elapsed>=stageSeconds[state.stage]){
      if(stepOnly||state.stage===5){state.elapsed=stageSeconds[state.stage];updateAnimation();stop();return;}
      selectStage(state.stage+1,false);
    }
    updateAnimation();frame=requestAnimationFrame(tick);
  }
  frame=requestAnimationFrame(tick);
}
function seek(seconds) {
  stop();state.comparing=false;resumeAfterCompare=false;let index=0;
  while(index<5&&seconds>=stageSeconds[index]){seconds-=stageSeconds[index];index++;}
  if(index!==state.stage){state.stage=index;state.elapsed=seconds;render(true);}
  else{state.elapsed=seconds;updateAnimation();}
}
function setTheme(theme){
  document.documentElement.dataset.theme=theme;$('#theme-label').textContent=theme==='dark'?'Dark':'Light';
  $('#theme-toggle').setAttribute('aria-pressed',String(theme==='dark'));
  $('#theme-toggle').setAttribute('aria-label','Switch to '+(theme==='dark'?'light':'dark')+' theme');
  $('#site-icon').href='static/images/branding/jepa-icon-'+theme+'.svg';
  try{localStorage.setItem('djepa-theme',theme);}catch(_){}
}
document.addEventListener('click',event=>{
  const evidenceOperation=event.target.closest('[data-evidence-step]');
  if(evidenceOperation){stop();state.elapsed=[.20,.55,1][Number(evidenceOperation.dataset.evidenceStep)]*stageSeconds[1];updateAnimation();return;}
  const operation=event.target.closest('[data-lift-step]');
  if(operation){stop();state.elapsed=[.08,.24,.58,1][Number(operation.dataset.liftStep)]*stageSeconds[4];updateAnimation();return;}
  const milestone=event.target.closest('[data-jump-stage]');
  if(milestone){selectStage(Number(milestone.dataset.jumpStage));state.elapsed=Number(milestone.dataset.jumpPhase)*stageSeconds[state.stage];updateAnimation();return;}
  const stage=event.target.closest('#chapters [data-stage]');
  if(stage){selectStage(Number(stage.dataset.stage));return;}
  const candidate=event.target.closest('[data-candidate]');
  if(candidate){state.candidate=Number(candidate.dataset.candidate);render(false);return;}
  const recorded=event.target.closest('[data-record-candidate]');
  if(recorded){state.recordCandidate=Number(recorded.dataset.recordCandidate);render(false);return;}
  const replay=event.target.closest('[data-replay]');
  if(replay){stop();state.replay=replay.dataset.replay==='true';state.elapsed=0;currentContext='';render(true);return;}
  const lifting=event.target.closest('[data-lifting]');
  if(lifting){stop();state.lifting=lifting.dataset.lifting;state.elapsed=0;render(true);return;}
});
function showMatrix(event){
  const cell=event.target.closest('.matrix-cell'),tip=$('#matrix-tooltip');
  if(!cell){tip.hidden=true;state.hoverCandidate=null;applyLinkedFocus();return;}
  tip.hidden=false;
  state.hoverCandidate=Number(cell.dataset.row);applyLinkedFocus();
  tip.textContent='Candidate '+CANDIDATES[Number(cell.dataset.row)]+' ← '+CANDIDATES[Number(cell.dataset.col)]+' · illustrative attention '+Number(cell.dataset.weight).toFixed(3);
}
$('#detail-visual').addEventListener('pointerover',showMatrix);
$('#detail-visual').addEventListener('pointerdown',event=>{if(event.target.closest('[data-lift-step], [data-latent], [data-evidence-step]'))stop();});
$('#detail-visual').addEventListener('focusin',showMatrix);
$('#detail-visual').addEventListener('pointerleave',()=>{$('#matrix-tooltip').hidden=true;state.hoverCandidate=null;applyLinkedFocus();});
$('#candidate-controls').addEventListener('pointerover',event=>{
  const target=event.target.closest('[data-candidate]');if(!target)return;
  state.hoverCandidate=Number(target.dataset.candidate);applyLinkedFocus();
});
$('#candidate-controls').addEventListener('pointerleave',()=>{state.hoverCandidate=null;applyLinkedFocus();});
document.addEventListener('keydown',event=>{
  const target=event.target;
  if((event.key==='Enter'||event.key===' ')&&target.matches('[role=button]')){event.preventDefault();target.dispatchEvent(new MouseEvent('click',{bubbles:true}));return;}
  if(target.closest('input,select,textarea,button,a,summary,dialog'))return;
  if(event.key==='ArrowRight'){event.preventDefault();selectStage(state.stage+1);}
  if(event.key==='ArrowLeft'){event.preventDefault();selectStage(state.stage-1);}
  if(event.code==='Space'){event.preventDefault();play();}
});
$('#play').onclick=()=>play();
$('#previous').onclick=()=>selectStage(state.stage-1);
$('#next').onclick=()=>selectStage(state.stage+1);
$('#tour-seek').max=totalDuration;
const milestones=[
  [1,.55,'Inspect ordinal evidence'],
  [1,.95,'Inspect candidate-token assembly'],
  [2,.72,'Inspect candidate relations'],
  [3,.38,'Inspect the decision boundary'],
  [3,.94,'Read the gate decision'],
  [4,.58,'Inspect representation lifting'],
  [4,1,'Read native-distance selection'],
];
$('.timeline').insertAdjacentHTML('beforeend','<div class="timeline-markers" aria-label="Key moments">'+milestones.map(([stage,phase,label])=>{
  const time=stageSeconds.slice(0,stage).reduce((a,b)=>a+b,0)+stageSeconds[stage]*phase;
  return '<button type="button" data-jump-stage="'+stage+'" data-jump-phase="'+phase+'" aria-label="'+label+'" title="'+label+'" style="left:'+100*time/totalDuration+'%"><span class="sr-only">'+label+'</span></button>';
}).join('')+'</div>');
$('#tour-seek').oninput=e=>seek(Number(e.target.value));
$('#reset').onclick=()=>{stop();Object.assign(state,{stage:0,candidate:0,sources:2,bound:.2,head:0,lifting:'ordinal',progress:0,elapsed:0,recordCandidate:2,replay:true,speed:1,comparing:false,hoverCandidate:null});$('#strength').value=.2;$('#speed').value='1';render(true);};
$('#speed').onchange=e=>state.speed=Number(e.target.value);
$('#strength').oninput=e=>{stop();state.bound=Number(e.target.value);state.elapsed=stageSeconds[state.stage];render(false);};
$$('[data-sources]').forEach(b=>b.onclick=()=>{stop();state.sources=Number(b.dataset.sources);render(false);});
$('#theme-toggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
$('#supervision-toggle').onclick=()=>{stop();$('#supervision-panel').showModal();$('#supervision-toggle').setAttribute('aria-expanded','true');};
$('#close-supervision').onclick=()=>$('#supervision-panel').close();
$('#supervision-panel').onclose=()=>$('#supervision-toggle').setAttribute('aria-expanded','false');
$('#fullscreen').onclick=async()=>{
  try{if(document.fullscreenElement)await document.exitFullscreen();else await $('#explorer').requestFullscreen();}
  catch(_){$('#fullscreen').title='Fullscreen is not available in this browser.';}
};
document.addEventListener('fullscreenchange',()=>$('#fullscreen').setAttribute('aria-label',document.fullscreenElement?'Exit fullscreen':'Enter fullscreen'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('storage',e=>{if(e.key==='djepa-theme')setTheme(e.newValue==='light'?'light':'dark');});
window.addEventListener('resize',updateViewport);
reducedMotion.addEventListener('change',()=>stop());
setTheme(document.documentElement.dataset.theme||'dark');
try {
  const response=await fetch('static/data/explainer-pusht.json');
  if(!response.ok)throw Error('Replay data unavailable');
  record=await response.json();
} catch(error) {
  state.replay=false;console.warn(error.message);
}
const hash=location.hash.match(/^#stage-([1-6])$/);if(hash)state.stage=Number(hash[1])-1;
render();
