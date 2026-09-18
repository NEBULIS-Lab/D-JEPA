import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { VALIDATION_TASKS, validationFrame, setValidationVisibility } from '../docs/static/js/explainer-validation.mjs';
test('full original PushT replay finishes before window shrinks',()=>{
  for(const t of [0,1,3,6,7,8]){
    const p=Math.max(0,Math.min(1,(t/8-.08)/.8));
    assert.equal(validationFrame(t).pushProgress,p*p*(3-2*p));
    assert.equal(validationFrame(t).shrink,0);
  }
  assert.equal(validationFrame(8).pushProgress,1);
  assert.ok(validationFrame(9).shrink>0&&validationFrame(9).shrink<1);
  assert.equal(validationFrame(10).shrink,1);
});
test('five task windows enter in order and retain bounded recorded-video times',()=>{
  const f=validationFrame(10.5);
  assert.ok(f.tasks[0].reveal>f.tasks[1].reveal);
  assert.equal(f.tasks[4].reveal,0);
  validationFrame(26).tasks.forEach((f,i)=>{
    assert.equal(f.reveal,1);assert.ok(f.time>=0&&f.time<VALIDATION_TASKS[i].duration);
  });
  assert.ok(VALIDATION_TASKS.every(t=>existsSync(new URL('../docs/static/videos/'+t.id+'.mp4',import.meta.url))));
});
test('actual video layers stay hidden until reveal, including backward seeks',()=>{
  const children = [{style:{}}, {style:{}}];
  const attributes = {};
  const card = {style:{}, querySelectorAll:()=>children, setAttribute:(k,v)=>{attributes[k]=v;}};
  for (const seconds of [0, 8, 10.3, 15, 0, 26, 7]) {
    const reveal = validationFrame(seconds).tasks[0].reveal;
    setValidationVisibility(card, reveal);
    assert.equal(card.style.display, reveal > 0 ? '' : 'none');
    assert.equal(attributes['aria-hidden'], String(reveal === 0));
    for (const child of children) {
      assert.equal(child.style.display, reveal > 0 ? '' : 'none');
      assert.equal(child.style.visibility, reveal > 0 ? 'visible' : 'hidden');
      assert.equal(child.style.opacity, String(reveal));
    }
  }
});
