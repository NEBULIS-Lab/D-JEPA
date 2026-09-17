import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {narrationFrame} from '../docs/static/js/explainer-narration-model.mjs';
const root=new URL('../docs/',import.meta.url);
const manifest=JSON.parse(readFileSync(new URL('static/data/explainer-narration.json',root)));
test('eight unique narration recordings match their manifest and sentence cues',()=>{
  assert.equal(manifest.segments.length,8);
  assert.equal(new Set(manifest.segments.map(s=>s.sha256)).size,8);
  let sentences=0;
  for(const segment of manifest.segments){
    const bytes=readFileSync(new URL(segment.audio,root));
    assert.equal(createHash('sha256').update(bytes).digest('hex'),segment.sha256);
    assert.ok(segment.duration>10&&segment.duration<40);
    assert.equal(segment.keyframes[0].time,0);
    assert.equal(segment.keyframes.at(-1).time,segment.duration);
    for(let i=0;i<segment.sentences.length;i++){
      const s=segment.sentences[i];assert.ok(s.text.length>10);
      assert.ok(s.start>=0&&s.end>s.start&&s.end<=segment.duration+.05);
      if(i)assert.ok(s.start>segment.sentences[i-1].start);
      assert.equal(narrationFrame(segment,s.start+.001).sentence.text,s.text);
      sentences++;
    }
    for(let i=1;i<segment.keyframes.length;i++){
      assert.ok(segment.keyframes[i].time>segment.keyframes[i-1].time);
      assert.ok(segment.keyframes[i].elapsed>=segment.keyframes[i-1].elapsed);
    }
  }
  assert.equal(sentences,36);
});
test('narration retains paired outcomes, exact realization, and complete task-wall timing',()=>{
  const parts=manifest.segments;
  assert.deepEqual(parts.map(p=>p.stage),[0,0,1,2,3,4,4,5]);
  assert.equal(narrationFrame(parts[0],parts[0].duration).elapsed,4);
  assert.equal(narrationFrame(parts[1],0).elapsed,4);
  assert.equal(narrationFrame(parts[1],parts[1].duration).elapsed,13.36);
  assert.equal(parts[5].lifting,'ordinal');assert.equal(parts[6].lifting,'transport');
  assert.equal(narrationFrame(parts[7],parts[7].duration).elapsed,26);
  const preview=parts[7].keyframes.filter(p=>p.elapsed>=8);
  for(let i=1;i<preview.length;i++){
    const a=preview[i-1],b=preview[i];
    assert.ok((b.elapsed-a.elapsed)/(b.time-a.time)<=1.001,'Recorded task previews must not be accelerated by narration mapping.');
  }
});
