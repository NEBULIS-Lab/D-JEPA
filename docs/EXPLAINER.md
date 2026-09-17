# Interactive D-JEPA explainer

Open [Inside D-JEPA](explainer.html) from the project page. The page is a single
viewport-sized workbench, with a compact computation path, a large stage, an
inspector and a shared playback timeline. It defaults to dark and shares the
main site's saved light/dark preference. Assets are local; no model download or
browser inference is required.

## Six stages

1. **Candidate futures:** synchronized PushT executions from a common start,
   or a separately labelled six-candidate teaching example.
2. **Predictive evidence:** source-specific descriptors, ordinal evidence and
   candidate-token construction, revealed in computational order.
3. **Relational alignment:** candidate-to-candidate attention, inspectable matrix
   cells and the bounded correction head.
4. **Decision rule:** base scores, bounded corrections, animated rank crossings
   and the calibrated margin gate.
5. **Future representation:** ordinal realization or five-step temporal transport;
   displayed native distances track the moving latent points.
6. **Action selection:** the original method-selected actions replayed in sync.

Candidate identity is linked across the teaching diagrams. Recorded executions
retain their original numeric candidate IDs. The recording and the six-candidate
mechanism illustration are separate data contexts, explicitly labelled.

The lifting stage is explicitly located between aligned ranks and native-distance
planning. Its four clickable operations read the final rank, compute the RMS
radius, rewrite the terminal future along its retained direction, and read the
native distance. The trajectory strip distinguishes retained steps t1–t4 from
the rewritten terminal t5. Original/current radius, current cost and nearest
candidate update together; temporal transport retains its separate diagnostic
identity.

## Recorded PushT case

The replay exports existing start-176 trajectories for TD-JEPA (candidate 15),
LeWM (79) and D-JEPA (23). Each contains 126 recorded states covering all 25
control actions over 2.50 seconds. Position and shortest-arc angle interpolation
smooth playback between recorded states; no new actions, physics rollouts or
outcomes are generated. The common viewport, goal pose, shape geometry and
original outcomes are retained. Colours follow the project palette.

The compact JSON stores source filenames and SHA-256 hashes. Its export tool is:

```bash
python scripts/export_explainer_trace.py /path/to/recorded/case \
  docs/static/data/explainer-pusht.json
```

This case illustrates different executed choices; it is not an aggregate
success-rate estimate. The physical scenes are recorded state reconstructions,
not decoded JEPA predictions.

## Teaching data and method identity

The six-candidate costs, descriptors, attention, correction outputs and transport
coefficients are deterministic illustrations, not checkpoint activations.
Attention and correction outputs illustrate separate operations; they do not
constitute a browser-executed trained forward pass. Source selection switches
between illustrative two- and four-geometry configurations, with 386- and
388-dimensional tokens respectively. The four-geometry instance has its own
learned parameters and a JEPA-WM base. The schematic margin threshold is −0.03.

Ordinal realization uses final gated ranks, unit-RMS directions and radii
π/(K+1). Its existing PushT checkpoint includes predictive and relational
computation. Reacher physical selection uses the relational score; temporal
transport is a separate five-step representation diagnostic. Complementary
predictor adaptation and calibrated composition are explained in the decision
stage. Training supervision opens in a dialog.

## Playback and accessibility

- Play/pause, previous/next, speed, reset and fullscreen.
- Scrub the whole tour or just the current step; replay a step without advancing.
- Trace candidates, inspect an attention head/cell, change source configuration
  and vary the schematic correction bound.
- Switch representation mechanism and inspect intermediate native distances.
- Hold the before-alignment button (pointer, Space or Enter) in the decision or
  lifting stage, then release to restore the exact progress and playback state.
- Jump to key moments via the timeline markers. The 53-second tour gives extra
  time to ranking crossings and lifting, including a brief boundary-switch hold.
- Hover a candidate to follow its evidence, rank path or latent point while
  dimming unrelated entries. Matrix hover highlights its candidate row/column.
- Arrow keys navigate stages; Space controls playback outside form controls.
- Keyboard-accessible diagrams, a training dialog with Escape support and
  reduced-motion handling for decorative transitions. Playback is user-initiated
  and stops when the tab is hidden.
- Desktop layout targets 1366×768, 1440×900 and 1920×1080 viewports; small screens
  use a stacked inspector.

## Implementation and references

- `static/js/explainer.js`: stage rendering, controls and shared playback clock.
- `static/js/explainer-model.mjs`: pure numerical teaching example.
- `static/js/explainer-motion.mjs`: deterministic presentation timing and
  rank-to-radius geometry, with endpoint and decision-boundary tests.
- `static/js/explainer-scenes.mjs`: interpolation and SVG recorded-state replay.
- `static/data/explainer-pusht.json`: compact recorded states and provenance.
- `static/css/explainer.css`: responsive workbench and both themes.

Interaction concepts draw on
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and
[ViT-Explainer](https://vit-explainer.vercel.app/). Their model bundles, analytics
and image assets are not redistributed. The implementation is original to this
project and covered by the repository's Apache-2.0 license.

Checks: `node --test tests/explainer-model.test.mjs` and
`python3 scripts/check_site.py`. Preview using
`bash scripts/preview_site.sh 8000`, then open `/explainer.html`.
