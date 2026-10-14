import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const root = new URL('../', import.meta.url);
const read = path => readFileSync(new URL(path, root), 'utf8');
const dataPath = 'docs/static/data/decision_local_ranking.json';

// Only the browser DOM and local fetch are substituted; the real page script runs.
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.attributes = {}; this.value = ''; }
  set textContent(value) { this.value = value; this.children = []; }
  get textContent() { return this.value + this.children.map(child => child.textContent).join(' '); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.value = ''; this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  focus() { this.focused = true; }
  showModal() { this.open = true; }
  close() { this.open = false; this.events.close?.(); }
}

function mount(data) {
  const link = new Element('a');
  const match = read('docs/index.html').match(/<a href="static\/data\/decision_local_ranking.json">([^<]+)<\/a>/);
  assert.ok(match, 'the metric panel must remain accessible from the homepage');
  link.textContent = match[1];
  link.href = 'static/data/decision_local_ranking.json';
  const document = {
    body: new Element('body'),
    createElement: tag => new Element(tag),
    querySelectorAll: () => [link],
  };
  vm.runInNewContext(read('docs/static/js/site-dialogs.js'), {
    document,
    fetch: async url => {
      assert.equal(url, link.href);
      return {ok: true, json: async () => data};
    },
  });
  return {link, dialog: document.body.children[0]};
}

test('metric dialog shows scientific definitions without exposing source metadata', async () => {
  const data = JSON.parse(read(dataPath));
  data.sources = [{id: 'internal-source', sha256: 'a'.repeat(64)}];
  const {link, dialog} = mount(data);
  let prevented = false;
  await link.events.click({preventDefault() { prevented = true; }});
  assert.ok(prevented);
  assert.ok(dialog.open);
  for (const heading of ['Recorded pair', 'Distance gap', 'Success–failure pair inversions', 'Rank correlation']) {
    assert.ok(dialog.textContent.includes(heading), heading);
  }
  assert.doesNotMatch(dialog.textContent, /[a-f0-9]{32,64}|sha256|internal-source/i);
  dialog.children[0].children[1].events.click();
  assert.equal(dialog.open, false);
  assert.ok(link.focused);
});

test('downloadable metric data retains scientific results without file fingerprints', () => {
  const text = read(dataPath);
  const data = JSON.parse(text);
  assert.equal(data.margin_summary.length, 2);
  assert.equal(data.margin_summary[0].starts, 96);
  assert.equal(data.margin_summary[0].failed_preferred, 8);
  assert.ok(data.definitions.gap);
  assert.ok(data.pair_error_summary.length);
  assert.doesNotMatch(text, /[a-f0-9]{32,64}|sha256|checksum|fingerprint/i);
});
