# Interactive D-JEPA explainer

Open [`explainer.html`](explainer.html) from the project's navigation, resource
buttons or method section. It shares the `djepa-theme` preference with the main
site and defaults to dark. All assets and scripts are local; there is no build
step, external runtime, analytics or model download.

## What the animation explains

1. Shared context, goal and candidate action sequences.
2. Source-specific future–goal descriptors and ordinal coordinates.
3. Complete-set relational attention and the bounded 64 → 8 → 1 head.
4. Base scores, corrections and the calibrated relational gate.
5. PushT exact ordinal realization or Reacher bounded temporal transport.
6. Candidate identity and the corresponding action-selection interface.

Two expandable sections explain complementary predictor adaptation/composition
and the four-geometry configuration. A separate recorded PushT video illustrates
executed outcomes.

The six-candidate scores, descriptor cells, attention matrices, corrections,
transport coefficients and T-shaped scenes are deterministic teaching examples,
not trained-checkpoint activations or experiment results. The illustrative gate
threshold is −0.03. The two-source example uses ordinal fusion; the four-source
example uses a JEPA-WM base and represents a separately trained configuration.
Training supervision is an explanatory overlay, not in-browser training.

The ordinal realization calculation uses strict final gated ranks, unit-RMS
directions and radii π/(K+1). Earlier predicted steps are retained. The current
PushT checkpoint contains predictive models and relational computation. Reacher
physical selection uses the relational score; temporal transport is presented
as the separate five-step representation diagnostic. Schematic physical scenes
are labelled illustrative; the recorded comparison below uses the existing
published experiment video without editing.

## Controls

- Play/pause, previous/next, playback speed and reset.
- Click a stage or a diagram module to inspect its internals.
- Trace a candidate via the A–F buttons; inspect one of four attention heads.
- Switch the predictive configuration between two and four sources.
- Move the illustrative correction bound to zero to recover the base decision.
- Toggle training supervision and replay either representation transformation.
- Arrow keys navigate stages and Space toggles playback when focus is outside
  an interactive control. Reduced-motion preferences disable animated transitions.
- The tour is user-initiated and pauses when the browser tab becomes hidden.

## Implementation and design references

- `static/js/explainer-model.mjs`: pure numerical teaching example.
- `static/js/explainer.js`: SVG diagrams, state, playback and accessibility.
- `static/css/explainer.css`: responsive layout and light/dark colour roles.

The interaction design draws on the explorable matrices, module selection and
progressive data flow of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
and [ViT-Explainer](https://vit-explainer.vercel.app/). Their model implementations,
bundles, illustrations and datasets are not redistributed here. D-JEPA code and
SVG diagrams are original to this project, under the repository's Apache-2.0 license.

Run `node --test tests/explainer-model.test.mjs` for mathematical invariants, and
`python3 scripts/check_site.py` for local links and existing media integrity.
Preview with `bash scripts/preview_site.sh 8000`, then open `/explainer.html`.
