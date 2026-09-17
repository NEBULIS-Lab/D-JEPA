import test from 'node:test';
import assert from 'node:assert/strict';
import {locateNarration,narrationFrame} from '../docs/static/js/explainer-narration-model.mjs';
const segment={duration:12,stage:4,lifting:'ordinal',sentences:[
  {start:.1,end:3,text:'Read the rank.'},{start:4,end:8,text:'Rewrite the terminal future.'}],
  keyframes:[{time:0,elapsed:0},{time:2,elapsed:2},{time:5,elapsed:2},
    {time:9,elapsed:12},{time:12,elapsed:12}]};
test('audio timing maps visual holds without stopping subtitles',()=>{
  assert.equal(narrationFrame(segment,3).elapsed,2);
  assert.equal(narrationFrame(segment,4.5).elapsed,2);
  assert.equal(narrationFrame(segment,3).sentence.text,'Read the rank.');
  assert.equal(narrationFrame(segment,4.5).sentence.text,'Rewrite the terminal future.');
  assert.equal(narrationFrame(segment,4.5).rate,0);
  assert.equal(narrationFrame(segment,7).elapsed,7);
  assert.equal(narrationFrame(segment,7).rate,2.5);
  assert.equal(narrationFrame(segment,12).elapsed,12);
});
test('seeking restores the correct segment, sentence and mechanism',()=>{
  assert.deepEqual(locateNarration([segment,segment],14),{index:1,time:2});
  assert.deepEqual(locateNarration([segment,segment],12),{index:1,time:0});
  assert.deepEqual(locateNarration([segment,segment],100),{index:1,time:12});
  assert.deepEqual(locateNarration([segment],-1),{index:0,time:0});
  assert.equal(narrationFrame(segment,7).lifting,'ordinal');
  assert.equal(narrationFrame({...segment,lifting:'transport'},7).lifting,'transport');
});
