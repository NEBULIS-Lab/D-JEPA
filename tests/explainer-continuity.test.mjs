import test from 'node:test';
import assert from 'node:assert/strict';
import * as motion from '../docs/static/js/explainer-motion.mjs';
import { relationPhase, relationMix } from '../docs/static/js/explainer-relations.mjs';
import { validationFrame } from '../docs/static/js/explainer-validation.mjs';

test('candidate carry reaches its destination, holds, and rewinds deterministically', () => {
  assert.equal(typeof motion.candidateCarryFrame, 'function');
  const from={x:80,y:320},to={x:40,y:110};
  assert.deepEqual(motion.candidateCarryFrame(from,to,0),{x:80,y:320,halo:0,done:false});
  const middle=motion.candidateCarryFrame(from,to,.3);
  assert.ok(middle.x<80&&middle.x>40&&middle.y<320&&middle.y>110);
  assert.deepEqual(motion.candidateCarryFrame(from,to,1),{x:40,y:110,halo:0,done:true});
  assert.deepEqual(motion.candidateCarryFrame(from,to,.3),middle);
});

test('carry is limited to adjacent teaching stages, never to recorded candidates or seeks', () => {
  assert.equal(typeof motion.canCarryCandidate, 'function');
  assert.equal(motion.canCarryCandidate(1,2,true),true);
  assert.equal(motion.canCarryCandidate(3,4,true),true);
  for(const [from,to,running] of [[0,1,true],[4,5,true],[1,4,true],[2,1,true],[1,2,false]])
    assert.equal(motion.canCarryCandidate(from,to,running),false);
});

test('relation messages arrive before contributing to context; pulses do not loop', () => {
  const early=relationPhase(.34);
  assert.ok(Array.isArray(early.travel));
  assert.ok(early.travel[0]>0&&early.travel[0]<1);
  assert.equal(early.messages[0],0);
  assert.ok(relationMix(0,0,2,.34).mixed.every(v=>v===0));
  const arrive=relationPhase(.445);
  assert.equal(arrive.travel[0],1);
  assert.ok(arrive.messages[0]>0&&arrive.arrivals[0]>0);
  for(const t of [0,.9,1,3])assert.ok(relationPhase(t).arrivals.every(v=>v===0));
  assert.ok(relationPhase(.72).messages.every(v=>v===1));
  assert.deepEqual(relationPhase(.445),arrive);
});

test('lifting cues separate radius preview, geometry motion, landing and native readout', () => {
  assert.equal(typeof motion.liftingCues, 'function');
  const preview=motion.liftingCues(.3);
  assert.ok(preview.target>0);assert.equal(preview.move,0);assert.equal(preview.read,0);
  const moving=motion.liftingCues(.55);
  assert.ok(moving.move>0&&moving.move<1);assert.equal(moving.arrival,0);
  assert.ok(motion.liftingCues(.79).arrival>0);
  assert.equal(motion.liftingCues(.79).move,1);
  assert.equal(motion.liftingCues(1).arrival,0);
  assert.equal(motion.liftingCues(1).read,1);
  assert.deepEqual(motion.liftingCues(.55),moving);
});

test('wall entrances settle without overshoot or changing recorded-video time', () => {
  const hidden=validationFrame(8).tasks[0];
  assert.ok(Number.isFinite(hidden.offsetY));assert.equal(hidden.offsetY,16);
  const entering=validationFrame(10.2).tasks[0];
  assert.ok(entering.offsetY>0&&entering.offsetY<16);
  assert.equal(entering.time,0);
  const settled=validationFrame(12).tasks[0];
  assert.equal(settled.offsetY,0);assert.equal(settled.scale,1);
  assert.ok(Math.abs(settled.time-1.35)<1e-10);
  assert.deepEqual(validationFrame(10.2).tasks[0],entering);
});
