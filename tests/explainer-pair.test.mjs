import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pairPhase, pairPoints, projectPoint, latentIntro } from '../docs/static/js/explainer-pair.mjs';
const pair=JSON.parse(readFileSync(new URL('../docs/static/data/explainer-pair.json',import.meta.url)));
const traces=JSON.parse(readFileSync(new URL('../docs/static/data/explainer-pusht.json',import.meta.url)));
test('intro camera visibly sweeps and returns without modifying measured radii',()=>{
  assert.equal(latentIntro(0).t,0);
  assert.ok(latentIntro(.055).yawOffset>.6);
  assert.ok(latentIntro(.165).yawOffset<-.6);
  assert.ok(Math.abs(latentIntro(.22).yawOffset)<1e-12);
  assert.equal(latentIntro(1).t,1);
});
test('same recorded start, opposing outcomes and authoritative score-to-radius identity',()=>{
  assert.equal(pair.start,traces.start);
  for(const c of pair.candidates){
    const r=traces.candidates.find(r=>r.id===c.id);
    assert.equal(r.sha256,c.trace_sha256);assert.equal(r.success,c.success);
    assert.ok(Math.abs(c.rms_distance**2-c.cost)<1e-12);
  }
  assert.deepEqual(pair.candidates.map(c=>[c.id,c.success]),[[79,false],[23,true]]);
  assert.ok(pair.candidates[0].cost<pair.candidates[1].cost);
});
test('rotating the latent view retains native radii and the stored opening angle',()=>{
  const points=pairPoints(pair);
  for(const yaw of [-1.22,-.25,0,.5,1.22]){
    const rotated=points.map(p=>projectPoint(p,yaw));
    rotated.forEach((p,i)=>assert.ok(Math.abs(Math.hypot(...p)-pair.candidates[i].rms_distance)<1e-12));
    const dot=rotated[0].reduce((s,v,i)=>s+v*rotated[1][i],0);
    assert.ok(Math.abs(Math.acos(dot/(Math.hypot(...rotated[0])*Math.hypot(...rotated[1])))-pair.angle_rad)<1e-12);
  }
});
test('execution completes before success/failure disclosure and aggregate diagnosis',()=>{
  assert.equal(pairPhase(0).motion,0);assert.equal(pairPhase(.2).motion,0);
  assert.equal(pairPhase(.67).outcome,0);assert.equal(pairPhase(.68).motion,1);
  assert.equal(pairPhase(.76).outcome,1);assert.equal(pairPhase(.76).diagnostic,0);
  assert.equal(pairPhase(1).diagnostic,1);assert.equal(pairPhase(1).motion,1);
});
