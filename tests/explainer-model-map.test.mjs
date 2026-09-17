import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {modelMapCue,modelMapMarkup} from '../docs/static/js/explainer-model-map.mjs';
const segments=JSON.parse(readFileSync(new URL('../docs/static/data/explainer-narration.json',import.meta.url))).segments;
test('architecture cues precede detail and preserve every recording duration',()=>{
  assert.equal(segments.reduce((n,s)=>n+s.duration,0),206.496);
  const cues=segments.flatMap(s=>(s.architecture||[]).map(c=>({s,c})));
  assert.deepEqual(cues.map(({c})=>c.mode),['overview','predictor','lifting','transport']);
  for(const {s,c} of cues){
    assert.ok(c.start>=0&&c.end<=s.duration&&c.end>c.start);
    assert.equal(modelMapCue(s,c.start).mode,c.mode);
    assert.equal(modelMapCue(s,c.end),null);
    assert.ok(modelMapCue(s,c.end-.01).progress>.95);
  }
  assert.equal(modelMapCue(segments[5],segments[5].sentences[1].start),null);
  assert.equal(modelMapCue(segments[6],segments[6].sentences[1].start),null);
  assert.equal(modelMapCue(segments[4],segments[4].sentences[4].start),null);
});
test('adaptation remains a complementary proposal with a distinct relational default',()=>{
  const svg=modelMapMarkup('predictor');
  assert.match(svg,/data-map-node="final-block" class="map-node map-active"/);
  assert.match(svg,/data-map-node="projection" class="map-node map-active"/);
  assert.doesNotMatch(svg,/data-map-node="earlier-blocks" class="map-node map-active"/);
  for(const text of ['Relational branch','Default decision','Native-distance','Calibrated composition'])assert.ok(svg.includes(text));
  assert.doesNotMatch(svg,/data-map-node="lifting"/);
});
test('realization receives future, rank and goal; transport ends in a representation study',()=>{
  const ordinal=modelMapMarkup('lifting'),transport=modelMapMarkup('transport');
  for(const id of ['future-lifting','goal-lifting','order-lifting'])assert.ok(ordinal.includes(`data-map-flow="${id}"`));
  assert.match(ordinal,/Aligned choice/);
  assert.match(transport,/Five refined steps/);assert.match(transport,/representation study/);
  assert.doesNotMatch(transport,/Aligned choice/);
  for(const mode of ['overview','predictor','lifting','transport'])assert.doesNotMatch(modelMapMarkup(mode),/NaN|undefined/);
});
