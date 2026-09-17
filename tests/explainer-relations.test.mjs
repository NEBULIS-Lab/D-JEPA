import test from 'node:test';
import assert from 'node:assert/strict';
import {relationMix,relationPhase} from '../docs/static/js/explainer-relations.mjs';
test('message accumulation equals the displayed normalized weighted sum',()=>{
 for(const sources of [2,4])for(let head=0;head<4;head++)for(let query=0;query<6;query++){
  const d=relationMix(query,head,sources);
  assert.ok(Math.abs(d.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
  d.mixed.forEach((v,j)=>assert.ok(Math.abs(v-d.weights.reduce((s,w,i)=>s+w*d.values[i][j],0))<1e-12));
  assert.ok(relationMix(query,head,sources,0).mixed.every(v=>v===0));
 }
});
test('aggregation precedes correction and different heads produce different evidence',()=>{
 assert.equal(relationPhase(.3).head,0);assert.equal(relationPhase(.72).head,0);
 assert.ok(relationPhase(.72).messages.every(v=>v===1));assert.equal(relationPhase(1).head,1);
 assert.notDeepEqual(relationMix(0,0,2).mixed,relationMix(0,1,2).mixed);
});
