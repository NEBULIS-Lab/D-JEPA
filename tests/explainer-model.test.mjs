import test from 'node:test';
import assert from 'node:assert/strict';
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
