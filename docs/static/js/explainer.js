import { CANDIDATES, COSTS, example, attention, descriptor, realizedPoint, transportPoint } from './explainer-model.mjs';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const state = { stage: 0, candidate: 0, sources: 2, bound: 0.2, head: 0, lifting: 'ordinal', progress: 1, playing: false, speed: 1, elapsed: 0, supervision: false };
const stageSeconds = [5.5, 7, 8, 7, 8, 6];
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
const tokenMark = (i, x, y, selected = i === state.candidate, size = 11) => circle(x, y, size, selected ? 'var(--purple-soft)' : 'var(--panel)', `stroke="${selected ? 'var(--accent)' : 'var(--line)'}"`) + text(x, y + 3.5, CANDIDATES[i], selected ? 'svg-small svg-accent' : 'svg-small', 'text-anchor="middle"');
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
  { kicker: '06 / ACTION SELECTION', title: 'The native interface selects the aligned action.', description: 'The planner compares the realized futures with the goal, selects the nearest one and passes its corresponding action sequence to the environment.', formula: 'a* = arg minᵢ ‖z̃ᵢ,ᴴ − zgoal‖² / D', fact: 'The candidate identity connects the predicted future, aligned order and executed action sequence.', visual: 'A single decision, carried through the interface', caption: 'This is the final choice in the illustrative example. A recorded experimental comparison is available below.' },
];

function renderArchitecture() {
  const data = example(state);
  const nodes = [
    { x: 22, w: 160, name: 'Candidate futures', sub: 'context + actions' },
    { x: 222, w: 193, name: 'Predictive evidence', sub: state.sources === 2 ? 'LeWM + TD-JEPA' : 'four predictive geometries' },
    { x: 455, w: 220, name: 'Relational alignment', sub: 'set-wise Transformer' },
    { x: 715, w: 145, name: 'Decision rule', sub: 'bounded correction' },
    { x: 900, w: 192, name: 'Future representation', sub: state.lifting === 'ordinal' ? 'exact ordinal realization' : 'temporal transport' },
    { x: 1132, w: 165, name: 'Action selection', sub: state.lifting === 'ordinal' ? 'native goal distance' : 'relational selection' },
  ];
  let svg = '<title id="architecture-title">D-JEPA architecture</title><desc id="architecture-description">A six-stage pipeline from candidates to predictive evidence, relations, scores, representations and action selection.</desc>';
  svg += '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="var(--line)"/></marker></defs>';
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i], b = nodes[i + 1];
    const d = `M${a.x + a.w} 221H${b.x - 3}`;
    svg += path(d, 'connector', 'marker-end="url(#arrow)"');
    svg += path(d, `flow-line ${state.stage > i ? 'finished' : ''}`);
  }
  nodes.forEach((n, i) => {
    svg += `<g class="module ${state.stage === i ? 'active' : ''}" data-stage="${i}" role="button" tabindex="0" aria-label="Explore ${n.name}" aria-pressed="${state.stage === i}">`;
    svg += `<rect class="node-bg" x="${n.x}" y="84" width="${n.w}" height="263" rx="14"/>`;
    svg += text(n.x + 3, 38, `0${i + 1}`, 'module-kicker') + text(n.x + 3, 60, n.name, 'svg-label');
    svg += text(n.x + n.w / 2, 107, n.sub, 'svg-small', 'text-anchor="middle"');
    svg += circle(n.x + n.w - 12, 96, 3, 'var(--purple)', 'class="stage-indicator"');
    if (i === 0) {
      svg += miniatureScene(n.x + 16, 122, n.w - 32, 95);
      svg += text(n.x + 15, 242, 'Candidate action sequences', 'svg-tiny');
      for (let j = 0; j < 6; j++) {
        const cx = n.x + 27 + (j % 3) * 49, cy = 266 + Math.floor(j / 3) * 35;
        svg += tokenMark(j, cx, cy, j === state.candidate, 10);
        svg += line(cx + 12, cy, cx + 22, cy, 'var(--blue)');
      }
    } else if (i === 1) {
      for (let j = 0; j < 6; j++) {
        const y = 139 + j * 29;
        svg += tokenMark(j, n.x + 19, y, j === state.candidate, 8);
        for (let d = 0; d < 9; d++) svg += rect(n.x + 36 + d * 10, y - 7, 7, 14, d < 4 ? 'var(--blue)' : 'var(--purple)', 'none', 2, `opacity="${.2 + descriptor(j, d < 4 ? 0 : 1, d) * .7}"`);
        const count = state.sources;
        for (let s = 0; s < count; s++) svg += rect(n.x + 134 + s * 10, y - 7, 7, 14, 'var(--gold)', 'none', 2, `opacity="${.25 + Object.values(data.sourceRanks)[s][j] * .6}"`);
      }
      svg += text(n.x + n.w / 2, 328, `${state.sources === 2 ? '386' : '388'} → 64 per candidate`, 'svg-mono', 'text-anchor="middle"');
    } else if (i === 2) {
      const weights = attention(state.head, state.sources);
      for (let row = 0; row < 6; row++) for (let col = 0; col < 6; col++) {
        svg += rect(n.x + 37 + col * 24, 130 + row * 24, 20, 20, 'var(--purple)', row === state.candidate || col === state.candidate ? 'var(--accent)' : 'none', 3, `opacity="${.15 + weights[row][col] * 1.5}" class="matrix-signal" style="animation-delay:${(row + col) * .09}s"`);
      }
      svg += rect(n.x + 25, 286, 170, 29, 'var(--surface)', 'var(--line)', 6);
      svg += text(n.x + 110, 305, 'LayerNorm · attention · MLP', 'svg-tiny', 'text-anchor="middle"');
      svg += text(n.x + 110, 332, '2 layers · 4 heads', 'svg-mono', 'text-anchor="middle"');
    } else if (i === 3) {
      data.finalOrder.forEach((candidate, index) => {
        const y = 140 + index * 29;
        svg += tokenMark(candidate, n.x + 20, y, candidate === state.candidate, 8);
        svg += rect(n.x + 39, y - 6, Math.max(4, (data.scores[candidate] + .2) * 65), 12, candidate === data.winner ? 'var(--purple)' : 'var(--blue)', 'none', 3, 'opacity=".8"');
        if (candidate === data.winner) svg += circle(n.x + 129, y, 3, 'var(--accent)');
      });
      svg += text(n.x + n.w / 2, 332, 'calibrated gate', 'svg-mono', 'text-anchor="middle"');
    } else if (i === 4) {
      if (state.lifting === 'ordinal') {
        const cx = n.x + 96, cy = 215;
        [22, 43, 65].forEach(r => { svg += circle(cx, cy, r, 'none', 'stroke="var(--line)"'); });
        svg += line(cx - 7, cy, cx + 7, cy, 'var(--gold)') + line(cx, cy - 7, cx, cy + 7, 'var(--gold)');
        data.ordinal.forEach((rank, j) => {
          const [x, y] = realizedPoint(j, rank);
          svg += circle(cx + x * 52, cy + y * 52, j === data.winner ? 5 : 3, j === data.winner ? 'var(--accent)' : 'var(--blue)');
        });
        svg += text(cx, 311, 'distance² ∝ rank²', 'svg-mono', 'text-anchor="middle"');
      } else {
        for (let t = 0; t < 5; t++) {
          const x = n.x + 20 + t * 37, y = 197 + Math.sin(t) * 22;
          svg += circle(x, y, 5, 'var(--blue)') + circle(x, y - 18, 5, 'var(--accent)') + line(x, y - 5, x, y - 12, 'var(--purple)');
        }
        svg += text(n.x + 96, 291, 'same action · same step', 'svg-tiny', 'text-anchor="middle"');
      }
      svg += text(n.x + n.w / 2, 332, 'JEPA-compatible latents', 'svg-tiny', 'text-anchor="middle"');
    } else {
      svg += text(n.x + n.w / 2, 148, `Action ${CANDIDATES[data.winner]}`, 'svg-title svg-accent', 'text-anchor="middle"');
      svg += miniatureScene(n.x + 17, 167, n.w - 34, 110, data.winner === 0);
      svg += text(n.x + n.w / 2, 309, 'future → candidate ID', 'svg-tiny', 'text-anchor="middle"');
      svg += text(n.x + n.w / 2, 328, '→ executed action', 'svg-tiny', 'text-anchor="middle"');
    }
    svg += '</g>';
  });
  if (state.supervision) {
    svg += path('M787 361V378H563V355', 'supervision-arrow');
    svg += text(684, 395, 'executed training outcomes → success mass + local ordering + correction penalty', 'svg-tiny svg-accent', 'text-anchor="middle"');
  } else {
    svg += text(663, 383, 'PREDICTIVE FUTURES  →  LEARNED RELATIONS  →  ALIGNED FUTURE GEOMETRY', 'module-kicker', 'text-anchor="middle"');
  }
  $('#architecture').innerHTML = svg;
}

function candidatesDetail() {
  let s = text(34, 30, 'Observed context + goal', 'svg-label');
  s += miniatureScene(30, 48, 192, 210);
  s += text(126, 283, 'The same start for every candidate', 'svg-small', 'text-anchor="middle"');
  const endpoints = [[.67, .38, 0], [.41, .55, -30], [.60, .44, 8], [.32, .65, 30], [.80, .63, 60], [.47, .70, -60]];
  endpoints.forEach(([xx, yy, angle], i) => {
    const x = 326 + (i % 3) * 139, y = 43 + Math.floor(i / 3) * 129;
    const selected = i === state.candidate;
    s += path(`M223 153C265 153 280 ${y + 48} ${x - 10} ${y + 48}`, selected ? 'trace-line' : 'detail-line', `opacity="${selected ? 1 : .45}"`);
    let cell = rect(x, y, 112, 97, 'var(--scene-bg)', selected ? 'var(--accent)' : 'var(--line)', 10);
    cell += tShape(x + 72, y + 39, 0, .6, '#95EE95', .5) + tShape(x + xx * 103, y + yy * 80, angle, .58);
    cell += tokenMark(i, x + 13, y + 14, selected, 9);
    cell += circle(x + 25, y + 79, 3.5, '#4A75E8');
    if (state.supervision) cell += text(x + 56, y + 115, i === 0 || i === 2 ? 'success target' : 'failure target', 'svg-tiny', 'text-anchor="middle"');
    s += candidateHit(i, x, y, 112, 97, cell);
  });
  return s;
}
function evidenceDetail() {
  const data = example(state);
  let s = text(75, 28, 'LeWM descriptor', 'svg-small') + text(263, 28, 'TD-JEPA descriptor', 'svg-small') + text(465, 28, 'Ordinal evidence', 'svg-small') + text(628, 28, 'Token', 'svg-small');
  for (let i = 0; i < 6; i++) {
    const y = 58 + i * 37;
    if (i === state.candidate) s += rect(15, y - 17, 726, 33, 'var(--purple-soft)', 'none');
    s += tokenMark(i, 36, y);
    for (let d = 0; d < 12; d++) {
      s += rect(77 + d * 12, y - 10, 9, 21, 'var(--blue)', 'none', 2, `opacity="${.15 + descriptor(i, 0, d) * .8}"`);
      s += rect(267 + d * 12, y - 10, 9, 21, 'var(--purple)', 'none', 2, `opacity="${.15 + descriptor(i, 1, d) * .8}"`);
    }
    for (let j = 0; j < state.sources; j++) {
      const rank = Object.values(data.sourceRanks)[j][i];
      s += rect(459 + j * 31, y - 10, 25, 21, 'var(--gold-fill)', 'none', 3);
      s += text(471 + j * 31, y + 4, rank.toFixed(1), 'svg-tiny svg-gold', 'text-anchor="middle"');
    }
    s += line(590, y, 615, y, 'var(--line)', 'stroke-width="1.3"');
    s += rect(628, y - 11, 94, 23, 'var(--purple-soft)', i === state.candidate ? 'var(--accent)' : 'var(--line)', 5);
    s += text(675, y + 4, `${state.sources === 2 ? '386' : '388'} → 64`, 'svg-mono svg-accent', 'text-anchor="middle"');
  }
  s += text(143, 302, '192 dimensions', 'svg-mono', 'text-anchor="middle"') + text(339, 302, '192 dimensions', 'svg-mono', 'text-anchor="middle"') + text(512, 302, `${state.sources} ranks`, 'svg-mono', 'text-anchor="middle"');
  return s;
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
      s += `<g><title>Illustrative attention ${CANDIDATES[i]} to ${CANDIDATES[j]}: ${fmt(value)}</title>`;
      s += rect(x0 + j * cell, y0 + i * cell, 27, 27, 'var(--purple)', i === state.candidate ? 'var(--accent)' : 'none', 4, `opacity="${.2 + value * 1.5}"`);
      if (i === state.candidate) s += text(x0 + j * cell + 13.5, y0 + i * cell + 17, value.toFixed(2), 'svg-tiny', 'text-anchor="middle"');
      s += '</g>';
    }
  }
  s += path('M467 156H522', 'trace-line');
  s += rect(535, 83, 199, 167, 'var(--surface)', 'var(--line)', 12);
  s += text(635, 116, 'Shared low-rank head', 'svg-small', 'text-anchor="middle"');
  s += text(635, 150, '64 → 8 → 1', 'svg-title svg-accent', 'text-anchor="middle"');
  s += text(635, 178, 'tanh · bound ε', 'svg-mono', 'text-anchor="middle"');
  s += text(635, 219, `δ${CANDIDATES[state.candidate]} = ${fmt(example(state).delta[state.candidate])}`, 'svg-title svg-accent', 'text-anchor="middle"');
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
    const map = p => [65 + p[0] * 930, 158 - p[1] * 275];
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
  } else {
    const cx = 215, cy = 164, scale = 96;
    s += text(37, 27, 'Goal-relative future geometry', 'svg-label');
    [1,2,3,4,5,6].forEach(rank => {
      const r = Math.SQRT2 * rank / 7 * scale;
      s += circle(cx, cy, r, 'none', 'stroke="var(--line)" stroke-dasharray="3 5"');
    });
    s += line(cx-8,cy,cx+8,cy,'var(--gold)','stroke-width="2"')+line(cx,cy-8,cx,cy+8,'var(--gold)','stroke-width="2"');
    s += text(cx + 12, cy + 4, 'goal', 'svg-tiny svg-gold');
    for (let i = 0; i < 6; i++) {
      const from = realizedPoint(i, 1 + d.base[i] * 5), to = realizedPoint(i,d.ordinal[i]);
      const x = cx + (from[0] * (1-state.progress) + to[0] * state.progress)*scale;
      const y = cy + (from[1] * (1-state.progress) + to[1] * state.progress)*scale;
      s += line(cx,cy,cx+to[0]*scale,cy+to[1]*scale,'var(--line)');
      s += circle(cx+from[0]*scale,cy+from[1]*scale,4,'none','stroke="var(--blue)" opacity=".55"');
      s += circle(x,y,i===d.winner?7:5,i===d.winner?'var(--accent)':'var(--blue)');
      s += text(x+10,y-8,CANDIDATES[i],i===d.winner?'svg-label svg-accent':'svg-small');
      if (i===state.candidate) s += circle(x,y,11,'none','stroke="var(--accent)" stroke-width="1.3"');
    }
    s += line(414,49,414,291);
    s += text(462,27,'Rank','svg-small')+text(550,27,'RMS radius','svg-small')+text(667,27,'Native cost','svg-small');
    d.finalOrder.forEach((id,idx)=>{
      const y=65+idx*35, rank=idx+1;
      if(id===d.winner)s+=rect(446,y-16,293,29,'var(--purple-soft)','none');
      s+=tokenMark(id,461,y,id===state.candidate,9)+text(497,y+4,rank,'svg-mono');
      s+=text(577,y+4,fmt(rank/7),'svg-mono','text-anchor="middle"')+text(700,y+4,fmt((rank/7)**2),'svg-mono','text-anchor="middle"');
    });
    s+=text(599,294,'native mean-squared distance = radius²','svg-small','text-anchor="middle"');
    s+=text(215,313,'Open points: original · filled points: realized','svg-tiny','text-anchor="middle"');
  }
  return s;
}
function executionDetail() {
  const d=example(state);
  let s = text(121,28,'Pretrained ordering','svg-label svg-gold','text-anchor="middle"')+text(379,28,'Aligned ordering','svg-label svg-accent','text-anchor="middle"')+text(646,28,'Action interface','svg-label','text-anchor="middle"');
  s+=rect(30,48,183,211,'var(--surface)','var(--line)',12)+rect(282,48,195,211,'var(--purple-soft)','var(--purple)',12);
  s+=text(121,87,`Base choice ${CANDIDATES[d.baseWinner]}`,'svg-title svg-gold','text-anchor="middle"')+text(379,87,`Aligned choice ${CANDIDATES[d.winner]}`,'svg-title svg-accent','text-anchor="middle"');
  s+=miniatureScene(52,106,139,133,false)+miniatureScene(306,106,146,133,d.winner===0);
  s+=path('M221 150H273','trace-line')+path('M485 150H531','trace-line');
  s+=rect(541,68,193,60,'var(--surface)','var(--line)',10)+text(637,92,state.lifting==='ordinal'?'Native latent distance':'Relational score','svg-small','text-anchor="middle"')+text(637,114,`select candidate ${CANDIDATES[d.winner]}`,'svg-mono svg-accent','text-anchor="middle"');
  s+=path('M637 131V166','trace-line')+rect(541,171,193,68,'var(--surface)','var(--line)',10)+text(637,197,'Execute its action sequence','svg-small','text-anchor="middle"');
  for(let j=0;j<8;j++)s+=rect(568+j*18,214,12,8,'var(--purple)','none',2,`opacity="${.25+j*.08}"`);
  s+=text(379,290,'Candidate identity is retained through prediction, alignment and execution.','svg-small','text-anchor="middle"');
  return s;
}

function renderDetail(animate = false) {
  let chapter = {...chapters[state.stage]};
  if(state.stage===1 && state.sources===4){chapter.formula='vᵢ = [dᵢᴸ ; dᵢᵀ ; rᵢᴸ ; rᵢᵀ ; rᵢᴶ ; rᵢᴰ]';chapter.fact='The four-geometry configuration uses 388-dimensional tokens, its own learned parameters and a JEPA-WM base score.';}
  if(state.stage===4 && state.lifting==='transport'){
    chapter.title='Refine a trajectory, one future step at a time.';
    chapter.description='A time-conditioned network learns a signed, bounded displacement along the direction between corresponding LeWM and TD-JEPA futures. Each update retains the same action and time index.';
    chapter.formula='z̃ᵀᵢ,ₜ = ẑᵀᵢ,ₜ + βᵢ,ₜ · normalized(ẑᴸᵢ,ₜ − ẑᵀᵢ,ₜ)';
    chapter.fact='Reacher: a 5 → 32 → 1 coefficient network, five future steps and a 0.1 bound. Physical decisions use relational selection; transported latents are evaluated in representation diagnostics.';
    chapter.visual='Bounded temporal transport · Reacher mechanism';
    chapter.caption='A schematic representation diagnostic. Each arrow updates the same candidate at the same future step.';
  }
  if(state.stage===5 && state.lifting==='transport'){
    chapter.title='Use aligned relations to choose the action.';
    chapter.description='The Reacher execution path selects the candidate using its relational score. Bounded temporal transport provides the complementary learned future-representation diagnostic.';
    chapter.formula='a* = calibrated relational selection';
    chapter.fact='The execution path and representation diagnostic retain their respective readouts.';
  }
  $('#detail-kicker').textContent=chapter.kicker;
  $('#detail-title').textContent=chapter.title;
  $('#detail-description').textContent=chapter.description;
  $('#detail-formula').textContent=chapter.formula;
  $('#detail-fact').textContent=chapter.fact;
  $('#focus-label').textContent=chapter.visual;
  $('#visual-caption').textContent=chapter.caption;
  const renderers=[candidatesDetail,evidenceDetail,relationsDetail,decisionDetail,liftingDetail,executionDetail];
  $('#detail-visual').innerHTML=renderers[state.stage]();
  $('#detail-visual').setAttribute('aria-label',chapter.visual);
  if(animate&&!reducedMotion.matches){$('#detail-visual').classList.remove('detail-in');void $('#detail-visual').getBoundingClientRect();$('#detail-visual').classList.add('detail-in');}
  renderOptions();
}
function renderOptions(){
  const container=$('#stage-options');
  // Retain focused controls during pointer/keyboard range updates.
  if(container.dataset.stage===String(state.stage))return;
  container.dataset.stage=state.stage;
  container.innerHTML='';
  if(state.stage===2){
    container.innerHTML=`<label>Attention head <select id="attention-head">${[0,1,2,3].map(h=>`<option value="${h}" ${h===state.head?'selected':''}>${h+1} / 4</option>`).join('')}</select></label>`;
    $('#attention-head').addEventListener('change',e=>{state.head=Number(e.target.value);renderArchitecture();renderDetail();});
  }
  if(state.stage===4){
    container.innerHTML=`<div class="segmented" role="group" aria-label="Representation mechanism"><button type="button" data-lifting="ordinal" aria-pressed="${state.lifting==='ordinal'}">Ordinal realization</button><button type="button" data-lifting="transport" aria-pressed="${state.lifting==='transport'}">Temporal transport</button></div><button type="button" id="animate-lifting">Replay the transformation ↻</button>`;
    container.querySelectorAll('[data-lifting]').forEach(button=>button.addEventListener('click',()=>{stop();state.lifting=button.dataset.lifting;state.progress=1;container.querySelectorAll('[data-lifting]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderArchitecture();renderDetail(true);animateLifting();}));
    $('#animate-lifting').addEventListener('click',()=>{stop();animateLifting();});
  }
  if(state.stage===0||state.stage===2){
    const button=document.createElement('button');button.type='button';button.textContent='View training supervision';button.addEventListener('click',()=>setSupervision(!state.supervision));container.append(button);
  }
}
function render(){
  const d=example(state);
  document.documentElement.dataset.stage=state.stage;
  $('#base-choice').textContent=CANDIDATES[d.baseWinner];
  $('#aligned-choice').textContent=CANDIDATES[d.winner];
  $('#strength-value').value=state.bound.toFixed(2);
  $('#model-note').textContent=state.sources===2?'Dual-geometry model: two 192-dimensional descriptors and two ordinal coordinates form each 386-dimensional token.':'Four-geometry model: 388-dimensional tokens, separate learned parameters and a JEPA-WM base score. Changing sources here switches the illustrative configuration.';
  $$('#chapters button').forEach(b=>{if(Number(b.dataset.stage)===state.stage)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  $$('[data-sources]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.sources)===state.sources)));
  $$('#candidate-controls button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.candidate)===state.candidate)));
  $('#previous').disabled=state.stage===0;$('#next').disabled=state.stage===5;
  renderArchitecture();renderDetail(true);updateProgress();
}
function selectStage(stage, manual=true){
  if(manual)stop();cancelAnimationFrame(slideFrame);state.progress=1;state.stage=Math.max(0,Math.min(5,stage));state.elapsed=0;render();
  const tab=$(`#chapters [data-stage="${state.stage}"]`);
  const tabStrip=$('#chapters');
  if(tabStrip.scrollWidth>tabStrip.clientWidth)tabStrip.scrollTo({left:tab.offsetLeft-tabStrip.offsetLeft-tabStrip.clientWidth/2+tab.offsetWidth/2,behavior:reducedMotion.matches?'instant':'smooth'});
  const canvas=$('.architecture-scroll');
  if(canvas.scrollWidth>canvas.clientWidth){
    const centers=[102,318,565,787,996,1214];
    canvas.scrollTo({left:centers[state.stage]*$('#architecture').getBoundingClientRect().width/1320-canvas.clientWidth/2,behavior:reducedMotion.matches?'instant':'smooth'});
  }
  if(state.stage===4)animateLifting();
}
function animateLifting(){
  cancelAnimationFrame(slideFrame);
  if(reducedMotion.matches){state.progress=1;renderDetail();return;}
  const start=performance.now();
  function tick(now){
    const t=Math.min(1,(now-start)/1600);state.progress=t*t*(3-2*t);renderDetail();
    if(t<1)slideFrame=requestAnimationFrame(tick);
  }
  state.progress=0;slideFrame=requestAnimationFrame(tick);
}
function updateProgress(){
  const duration=stageSeconds.reduce((a,b)=>a+b,0);
  const past=stageSeconds.slice(0,state.stage).reduce((a,b)=>a+b,0);
  $('#tour-progress').style.width=`${100*(past+state.elapsed)/duration}%`;
}
function stop(){state.playing=false;cancelAnimationFrame(frame);document.body.classList.remove('running');$('#play-symbol').textContent='▶';$('#play-label').textContent='Play tour';$('#play').setAttribute('aria-label','Play guided tour');}
function play(){
  if(state.playing){stop();return;}
  if(state.stage===5){selectStage(0,false);}
  state.playing=true;previousTime=performance.now();document.body.classList.add('running');$('#play-symbol').textContent='Ⅱ';$('#play-label').textContent='Pause';$('#play').setAttribute('aria-label','Pause guided tour');
  function tick(now){
    if(!state.playing)return;
    state.elapsed+=Math.min(.2,(now-previousTime)/1000)*state.speed;previousTime=now;
    if(state.elapsed>=stageSeconds[state.stage]){
      if(state.stage===5){state.elapsed=stageSeconds[5];updateProgress();stop();return;}
      selectStage(state.stage+1,false);
    }
    updateProgress();frame=requestAnimationFrame(tick);
  }
  frame=requestAnimationFrame(tick);
}
function setSupervision(visible){state.supervision=visible;$('#supervision-panel').hidden=!visible;$('#supervision-toggle').setAttribute('aria-pressed',String(visible));$('#supervision-toggle').textContent=visible?'Hide training supervision':'Show training supervision';renderArchitecture();renderDetail();}
function setTheme(theme){
  document.documentElement.dataset.theme=theme;$('#theme-label').textContent=theme==='dark'?'Dark':'Light';$('#theme-toggle').setAttribute('aria-pressed',String(theme==='dark'));$('#theme-toggle').setAttribute('aria-label',`Switch to ${theme==='dark'?'light':'dark'} theme`);$('#site-icon').href=`static/images/branding/jepa-icon-${theme}.svg`;
  try{localStorage.setItem('djepa-theme',theme);}catch(_){}
}

$('#candidate-controls').insertAdjacentHTML('beforeend',CANDIDATES.map((id,i)=>`<button type="button" data-candidate="${i}" aria-label="Trace candidate ${id}" aria-pressed="${i===state.candidate}">${id}</button>`).join(''));
document.addEventListener('click',event=>{
  const stage=event.target.closest('#chapters [data-stage], #architecture [data-stage]');if(stage){selectStage(Number(stage.dataset.stage));return;}
  const candidate=event.target.closest('[data-candidate]');if(candidate){state.candidate=Number(candidate.dataset.candidate);render();}
});
document.addEventListener('keydown',event=>{
  const target=event.target;
  if((event.key==='Enter'||event.key===' ')&&target.matches('g[role=button]')){event.preventDefault();target.dispatchEvent(new MouseEvent('click',{bubbles:true}));return;}
  if(target.matches('input,select,textarea,button,a,summary')||target.closest('video'))return;
  if(event.key==='ArrowRight'){event.preventDefault();selectStage(state.stage+1);}
  if(event.key==='ArrowLeft'){event.preventDefault();selectStage(state.stage-1);}
  if(event.code==='Space'){event.preventDefault();play();}
});
$('#play').addEventListener('click',play);
$('#start-tour').addEventListener('click',()=>{stop();selectStage(0,false);play();});
$('#previous').addEventListener('click',()=>selectStage(state.stage-1));
$('#next').addEventListener('click',()=>selectStage(state.stage+1));
$('#reset').addEventListener('click',()=>{
  stop();cancelAnimationFrame(slideFrame);Object.assign(state,{stage:0,candidate:0,sources:2,bound:.2,head:0,lifting:'ordinal',progress:1,elapsed:0,supervision:false,speed:1});$('#strength').value=.2;$('#speed').value='1';$('#stage-options').dataset.stage='';setSupervision(false);render();
});
$('#speed').addEventListener('change',e=>state.speed=Number(e.target.value));
$('#strength').addEventListener('input',e=>{stop();state.bound=Number(e.target.value);render();});
$$('[data-sources]').forEach(button=>button.addEventListener('click',()=>{stop();state.sources=Number(button.dataset.sources);render();}));
$('#supervision-toggle').addEventListener('click',()=>setSupervision(!state.supervision));
$('#theme-toggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('storage',event=>{if(event.key==='djepa-theme'&&(event.newValue==='light'||event.newValue==='dark'))setTheme(event.newValue);});
reducedMotion.addEventListener('change',()=>{stop();cancelAnimationFrame(slideFrame);state.progress=1;render();});
const hash=location.hash.match(/^#stage-([1-6])$/);if(hash)state.stage=Number(hash[1])-1;
setTheme(document.documentElement.dataset.theme||'dark');render();
