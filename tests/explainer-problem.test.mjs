import test from 'node:test';
import assert from 'node:assert/strict';
import { DIAGNOSTIC, problemPhase } from '../docs/static/js/explainer-problem.mjs';
test('diagnostic endpoints match the published within-start values',()=>{
  assert.deepEqual(DIAGNOSTIC.map(x=>[x.model,x.all,x.shortlist]),[['LeWM',.90,.11],['TD-JEPA',.80,.13]]);
});
test('preview stops at 0.8 seconds; shortlist precedes diagnostic reveal',()=>{
  assert.equal(problemPhase(0).motion,0);
  assert.equal(problemPhase(1).motion*2.5,.8);
  assert.equal(problemPhase(.5).focus,1);
  assert.equal(problemPhase(.5).reveal,0);
  assert.equal(problemPhase(1).reveal,1);
  for(let i=0;i<=100;i++) assert.ok(problemPhase(i/100).motion<=.32);
});
