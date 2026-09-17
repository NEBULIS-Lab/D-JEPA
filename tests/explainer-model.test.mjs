import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { sampleState } from '../docs/static/js/explainer-scenes.mjs';
import { liftingGeometry, liftingProgress, liftingStep, decisionProgress } from '../docs/static/js/explainer-motion.mjs';
import { CANDIDATES, example, attention, realizedPoint, nativeCost, transportPoint } from '../docs/static/js/explainer-model.mjs';

test('bounded corrections, zero-bound base recovery and gated rank identity', () => {
  for (const sources of [2, 4]) for (const bound of [0, .005, .05, .1, .15, .2]) {
    const data = example({sources, bound});
    assert.equal(new Set(data.finalOrder).size, CANDIDATES.length);
    assert.equal(data.finalOrder[0], data.winner);
    assert.ok(data.delta.every(x => Math.abs(x) <= bound));
    if (bound === 0) assert.equal(data.winner, data.baseWinner);
    const native = data.ordinal.map((rank, i) => nativeCost(realizedPoint(i, rank)));
    const nativeOrder = native.map((v,i) => [v,i]).sort((a,b) => a[0]-b[0]).map(x=>x[1]);
    assert.deepEqual(nativeOrder, data.finalOrder);
    data.ordinal.forEach((rank, i) => assert.ok(Math.abs(native[i] - (rank/7)**2) < 1e-12));
  }
});
test('the two illustrative configurations show a genuine choice change', () => {
  for (const sources of [2,4]) {
    const data=example({sources});
    assert.notEqual(data.winner, data.baseWinner);
  }
});
test('illustrative attention is a normalized full-set matrix', () => {
  for (const sources of [2,4]) for(let head=0;head<4;head++) {
    const matrix=attention(head,sources);
    assert.equal(matrix.length,6);
    for(const row of matrix){assert.equal(row.length,6);assert.ok(row.every(x=>x>0));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);}
  }
});
test('temporal transport keeps every same-step displacement within the bound', () => {
  for(let i=0;i<6;i++) for(let t=0;t<5;t++) {
    const p=transportPoint(i,t);
    assert.ok(Math.hypot(p.refined[0]-p.source[0],p.refined[1]-p.source[1])<=.1+1e-12);
    assert.deepEqual(transportPoint(i,t,0).refined,p.source);
  }
});

test('recorded replay retains start, full duration, terminal state and outcomes', () => {
  const record=JSON.parse(readFileSync(new URL('../docs/static/data/explainer-pusht.json', import.meta.url)));
  assert.equal(record.candidates.length,3);
  for(const candidate of record.candidates){
    assert.equal(candidate.states.length,126);
    assert.equal(candidate.times[0],0);
    assert.equal(candidate.times.at(-1),2.5);
    assert.deepEqual(sampleState(candidate,0),candidate.states[0]);
    const terminal=sampleState(candidate,1);
    terminal.forEach((value,i)=>assert.ok(Math.abs(value-candidate.states.at(-1)[i])<1e-10));
    candidate.states[0].forEach((value,i)=>assert.ok(Math.abs(value-record.candidates[0].states[0][i])<1e-5));
    const distance=Math.hypot(...terminal.slice(0,4).map((value,i)=>value-record.goal[i]));
    const angle=Math.abs(Math.atan2(Math.sin(terminal[4]-record.goal[4]),Math.cos(terminal[4]-record.goal[4])));
    assert.equal(distance<20 && angle<Math.PI/9,candidate.success);
    assert.equal(candidate.sha256.length,64);
  }
});

test('recorded replay interpolates periodic angles along the shorter arc', () => {
  const candidate={states:[[0,0,0,0,Math.PI*2-.1],[2,2,2,2,.1]]};
  const middle=sampleState(candidate,.5);
  assert.equal(middle[0],1);
  assert.ok(Math.abs(middle[4]-Math.PI*2)<1e-10);
});

test('lifting shows rank-to-radius conversion, retained direction and exact native order', () => {
  for(const sources of [2,4]){
    const d=example({sources});
    const before=liftingGeometry(d,0),after=liftingGeometry(d,1);
    assert.equal([...before].sort((a,b)=>a.cost-b.cost)[0].id,d.baseWinner);
    assert.deepEqual([...after].sort((a,b)=>a.cost-b.cost).map(g=>g.id),d.finalOrder);
    for(const g of after){assert.ok(Math.abs(g.radius-g.rank/7)<1e-12);assert.ok(Math.abs(g.cost-g.radius**2)<1e-12);}
    for(let p=0;p<=1;p+=.05){
      for(const g of liftingGeometry(d,p))assert.ok(g.radius>=Math.min(g.original,g.target)-1e-12&&g.radius<=Math.max(g.original,g.target)+1e-12);
    }
  }
  assert.equal(liftingProgress(.24),0);assert.equal(liftingProgress(1),1);
  assert.deepEqual([.08,.24,.58,1].map(liftingStep),[0,1,2,3]);
});

test('decision timing pauses just after the competing candidates cross', () => {
  for(const sources of [2,4]){
    const d=example({sources}),p=decisionProgress(.38,d);
    assert.equal(p,decisionProgress(.46,d));
    assert.ok(d.base[d.winner]+p*d.delta[d.winner]<d.base[d.baseWinner]+p*d.delta[d.baseWinner]);
    assert.equal(decisionProgress(0,d),0);assert.equal(decisionProgress(1,d),1);
  }
});
