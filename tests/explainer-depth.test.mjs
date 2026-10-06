import test from 'node:test';
import assert from 'node:assert/strict';
import * as motion from '../docs/static/js/explainer-motion.mjs';
import { example, realizedPoint } from '../docs/static/js/explainer-model.mjs';

test('descriptor pieces assemble without overlapping their readable faces', () => {
  assert.equal(typeof motion.tokenAssemblyFrame, 'function');
  assert.deepEqual(motion.tokenAssemblyFrame(0,0),{x:550,y:99,width:164});
  assert.deepEqual(motion.tokenAssemblyFrame(2,1),{x:671,y:145,width:54});
  for(let p=0;p<=1;p+=.005){
    const boxes=[0,1,2].map(i=>motion.tokenAssemblyFrame(i,p));
    for(let i=0;i<3;i++)for(let j=i+1;j<3;j++){
      const a=boxes[i],b=boxes[j];
      assert.ok(a.x+a.width<=b.x||b.x+b.width<=a.x||a.y+26<=b.y||b.y+26<=a.y,
        'readable faces must remain disjoint at '+p);
    }
  }
});

test('presentation focus follows the active computation without hiding context', () => {
  assert.equal(typeof motion.computationFocus, 'function');
  for (const stage of [1, 2, 4]) {
    const first=motion.computationFocus(stage,0), last=motion.computationFocus(stage,1);
    assert.equal(first[0],1); assert.equal(last[2],1);
    assert.ok(first[0]>first[2]); assert.ok(last[2]>last[0]);
    for (let t=0;t<=1;t+=.01) {
      const now=motion.computationFocus(stage,t), next=motion.computationFocus(stage,t+.001);
      assert.ok(now.every(v=>v>=.6&&v<=1));
      assert.ok(now.every((v,i)=>Math.abs(v-next[i])<.02),'no focus jump at operation boundaries');
    }
    assert.deepEqual(motion.computationFocus(stage,.5),motion.computationFocus(stage,.5));
  }
});

test('oblique presentation retains the goal, radial direction and native-cost identity', () => {
  assert.equal(typeof motion.latentPlanePoint, 'function');
  assert.deepEqual(motion.latentPlanePoint([0,0]),[358,191]);
  const data=example({sources:2,bound:.2});
  for(let i=0;i<6;i++){
    const base=realizedPoint(i,1+data.base[i]*5),target=realizedPoint(i,data.ordinal[i]);
    const a=motion.latentPlanePoint(base),b=motion.latentPlanePoint(target);
    const cross=(a[0]-358)*(b[1]-191)-(a[1]-191)*(b[0]-358);
    assert.ok(Math.abs(cross)<1e-8,'display projection must not change candidate direction');
    const before=[...target];
    motion.latentPlanePoint(target);
    assert.deepEqual(target,before,'rendering must never mutate latent coordinates');
    assert.ok(Math.abs((target[0]**2+target[1]**2)/2-(data.ordinal[i]/7)**2)<1e-10);
  }
});
