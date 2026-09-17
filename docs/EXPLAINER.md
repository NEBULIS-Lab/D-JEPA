# Interactive D-JEPA explainer

Open [Inside D-JEPA](explainer.html) from the project page. The page is a single
viewport-sized workbench, with prominent chapter navigation, a large stage, an
inspector and a shared playback timeline. It defaults to dark and shares the
main site's saved light/dark preference. Assets are local; no model download or
browser inference is required.

## Narrated tour

The purple **Narrated tour** button plays the authors' eight English voiceover
segments with sentence-by-sentence subtitles in the header. It is separate from
the gradient **Play tour** button, which retains the freely explorable silent
animation. The header expands during narration while the workbench remains
within the desktop viewport. The complete voiceover lasts **3 minutes 26 seconds**
at its original speed and contains 36 sentence cues.

Audio time drives both the subtitle and the visual frame. Transformations finish
and hold where the explanation needs more time; the later recorded task previews
retain their playback speed. **Sound / Muted** changes only audibility, so muted
narration still advances the captions and animation. Pause, the timeline and the
previous/next controls operate on narration while this mode is active. Changing
a chapter or candidate pauses the voice; resuming restores its scripted scene.
Play tour or the header's close control exits narrated mode. Space toggles the
active mode, and browser-tab visibility changes pause playback.

The audio files in `static/audio/narration/` are unchanged author-supplied MP3s.
`static/data/explainer-narration.json` contains their hashes, actual durations,
sentence boundaries and visual keyframes. Word timestamps were matched to the
exact supplied script, then sentence boundaries and cue transitions checked;
the segments were not divided into equal-duration subtitle chunks. Files load
on demand, and the next segment preloads during playback. No external speech
service, model download or browser transcription is involved.

`explainer-narration-model.mjs` implements the pure time mapping;
`explainer-narration.mjs` manages audio, captions and mode transitions. The
narration tests check the source hashes, unique segments, sentence timing,
visual holds, mechanism selection and preserved task-preview timing.

## Six stages

### Architecture locator

**Locate in the world model** opens a shared architecture view inside the existing
canvas. The encoders, action-conditioned predictor and future representations stay
in consistent locations across four views: overview, predictor adaptation, ordinal
realization and temporal transport. Moving highlights follow the computation;
the original detailed animation returns when the locator closes. The light/dark
themes, viewport height, source recordings and original audio remain unchanged.

During narration the view opens at four sentence-aligned intervals:

| Voiceover segment | Local audio time | Architectural focus |
| --- | --- | --- |
| 03 | 0–4.80 s | Observation and goal encoding, prediction, candidate futures. |
| 05 | 13.86–20.50 s | Final TD-JEPA predictor block and projection; complementary native-distance proposal. |
| 06 | 0–5.18 s | Original future, goal and aligned rank entering ordinal realization. |
| 07 | 0–5.74 s | Matched predictive futures entering bounded temporal transport. |

These intervals precede the corresponding detailed operations. Adaptation remains
a separate proposal combined with the relational default; it is not drawn as a
mandatory stage before relational alignment. Ordinal realization ends in native
distance and the recovered aligned choice. Temporal transport ends in five refined
future steps and retains its Reacher representation-study identity.

The locator is also available for manual inspection during the silent tour.
Opening it pauses playback; tabs select a view, and **Back to detail** or Escape
returns to the detailed animation. Resuming narration restores the audio-specified
view. Timing is stored in the existing narration manifest's `architecture` cues;
`explainer-model-map.mjs` contains the diagrams, cue lookup and viewport controller.

### Detailed operations

1. **The prediction gap:** two recorded candidates from the same LeWM space have
   nearby predicted goal distances but opposing physical outcomes. A rotatable
   radial view leads into their full synchronized executions, then the measured
   population-level diagnosis. Three clickable operations advance this sequence.
   Average within-start Spearman correlations over 96 matched starts fall from
   0.90 to 0.11 for LeWM and 0.80 to 0.13 for TD-JEPA (all 63 candidates → top four;
   rounded values from the project-page diagnostic). The single curated pair and
   the aggregate diagnostic have distinct evidence identities. Stage 06 begins
   with the original three-method replay, then expands to cross-task validation.
2. **Predictive evidence:** source-specific descriptors, ordinal evidence and
   candidate-token construction, revealed in computational order. Three clickable
   operations animate goal subtraction and LayerNorm, within-source cost sorting
   and rank normalization, then descriptor/rank concatenation and shared encoding.
   The selected candidate's descriptor segments pack into one token, followed by
   the six-candidate 64D token bank. Four-source mode adds two rank coordinates,
   keeping the two 192D descriptors.
3. **Relational alignment:** candidate-to-candidate attention, inspectable matrix
   cells and the bounded correction head. Three independently replayable operations
   reveal the comparison matrix, pass weighted messages into a query-specific
   context, and animate the bounded correction. Select a matrix row or use Trace
   to change the query, and switch heads to inspect different attention patterns.
   Eight displayed value components are actually aggregated with the illustrative
   normalized attention weights. The displayed head output remains a separate
   teaching value, not a trained forward pass through the illustrated single head.
   Multi-head combination, residual and feed-forward processing connect that
   illustrative head context to the shared 64 → 8 → 1 correction head.
4. **Decision rule:** base scores, bounded corrections, animated rank crossings
   and the calibrated margin gate.
5. **Future representation:** ordinal realization or five-step temporal transport;
   displayed native distances track the moving latent points.
6. **Action selection and cross-task validation:** the original method-selected
   PushT actions replay in sync for eight presentation seconds, including the
   complete recorded horizon and its ending hold. The whole PushT view then
   scales down and moves to the upper-left tile; Reacher, Granular manipulation,
   bimanual grasping, driving and unseen-shape windows enter sequentially. The
   final wall connects one decision to evaluations across distinct physical
   systems. Task videos share the tour clock and pause/seek together. Select a
   window to open its full published comparison in a larger player.

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

Stage 01 uses candidates **A = 79** and **B = 23** from this same recording.
Their original LeWM MSE costs are 0.04107749 and 0.04695161 (pool ranks 1 and 3).
The radial view uses the square roots, 0.20267583 and 0.21668319, as RMS distances.
A fails and B succeeds after the original 25 actions. Outcomes are revealed only
after the 2.50-second recorded horizon completes. Shell radii retain their ratio
under camera rotation; the opening angle is recovered from the stored float16
goal-relative latents. Camera orientation is illustrative, not a learned 3D
embedding. Projected screen lengths vary with the view; numeric radii remain fixed.

`static/data/explainer-pair.json` records scores, metric definition, candidate IDs,
outcomes, trace hashes and source dataset keys. Re-export from an existing case
and its corresponding compact pool:

```bash
python scripts/export_explainer_pair.py /path/to/recorded/case \
  /path/to/compact_pool.h5 docs/static/data/explainer-pair.json
```

This CPU export verifies cost/outcome agreement with the case, checks latent MSE
within float16 storage tolerance, and preserves the original replay data. If pair
metadata is unavailable, stage 01 falls back to the earlier aggregate diagnostic.

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
- In ordinal lifting, each of the four operation buttons plays only its own
  interval and then pauses. Rank rows reveal in order, rank fractions fill a
  target-radius ruler, and the selected terminal point leaves a short radial
  movement trail. The native-cost bars update from the same instantaneous
  squared RMS radii and highlight the current nearest future. Their rows remain
  in aligned order so the final increasing cost sequence can be checked directly.
- The five-step strip keeps the first four illustrative component cards fixed
  while the fifth responds to the terminal radius. Temporal transport shows
  five signed coefficient gauges over the same ±0.1 bound; their numbers are
  the currently applied coefficients, reaching the full teaching-example values
  at the end. These remain illustrative computations, not experiment outputs.
- Hold the before-alignment button (pointer, Space or Enter) in the decision or
  lifting stage, then release to restore the exact progress and playback state.
- Rotate the stage-01 radial view by dragging, its angle slider, or Left/Right
  while the space is focused. View rotation does not change distances or outcomes.
- The first stage-01 operation turns the camera in one direction, draws the two
  measured-radius rays in sequence, and reveals the candidate points. Selecting
  that operation replays its animation and pauses before physical execution.
  Its comparison now takes 4.00 seconds (previously 2.64), with eased entry/exit
  and a final 0.40-second settled view. The subsequent recorded-execution and
  diagnostic intervals keep their previous durations. The final camera angle
  holds rather than reversing. Wave-grid lines taper gently toward the viewport
  edges, candidate letters have a background outline, and the goal has a small
  anchor ring and ground projection.
  Reduced-motion preference suppresses the camera sweep. Numerical radii remain
  fixed throughout. The primary play button uses the project-page purple/peach/gold
  gradient; chapter navigation has larger, higher-contrast text.
- The radial comparison sits in an open, depth-sorted purple point cloud with
  sparse local links, a gently undulating open grid and ground projections.
  The mesh extends beyond the viewport with a soft edge fade; bright gold/blue
  candidate beacons gradually illuminate without changing their measured radii.
  These 228 backdrop
  points are deterministic spatial illustration, not exported embeddings,
  candidate counts, affinities or success clusters. A/B remain the two separately
  measured candidates. Thin radius arcs replace the old enclosing wire spheres;
  traveling ray tips, gentle camera pitch and depth-dependent point opacity make
  the geometry easier to follow. Reduced-motion preference fixes the camera and
  backdrop positions while retaining the sequential distance readout.
- Inspector titles use a quieter 19–24 px hierarchy. Narration subtitles use
  17 px on desktop and 15 px on mobile, with theme gold (#FFDB67) in dark mode
  and a darker golden tone for readable contrast in light mode.
- Peripheral context strips, the top status line and below-canvas captions are
  removed from the visible workbench. Evidence identity and measurement notes
  remain available in the inspector. Candidate Trace controls remain visible for
  the computation stages; the stage-01 motion clock/reset strip is removed.
- Jump to key moments via the timeline markers. The approximately 80-second tour gives extra
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
- `static/js/explainer-problem.mjs`: measured diagnostic endpoints and bounded
  fallback shortlist-reveal timing; no interpolated statistics.
- `static/js/explainer-pair.mjs`: radial 3D projection and paired-replay timing.
- `static/data/explainer-pair.json`: verified latent costs, radii and pair provenance.
- `static/js/explainer-validation.mjs`: task-wall entrance timing and video-clock
  synchronization. Stage 06 lasts 26 seconds; its first eight seconds retain the
  original PushT playback timing.
- `static/videos/explainer-wall/`: lightweight camera-focused previews and a
  provenance manifest. Reacher, Granular and shape previews retain baseline/D-JEPA
  camera panes and omit peripheral text; shape omits the middle fusion pane.
  Bimanual video retains both camera panes; driving retains recorded camera
  context and simulated trajectories. Baseline is left and D-JEPA is right.
  The wall shows excerpts; the modal opens the unchanged full published video.
  No physical-robot footage is substituted with simulated footage.
- `static/js/explainer-model.mjs`: pure numerical teaching example.
- `static/js/explainer-relations.mjs`: message timing and weighted aggregation of
  the displayed illustrative value vectors, with normalization and summation tests.
- `static/js/explainer-motion.mjs`: deterministic presentation timing and
  rank-to-radius geometry, with endpoint and decision-boundary tests.
- `static/js/explainer-evidence.mjs`: illustrative vector subtraction, LayerNorm
  and evidence-stage timing. Eight of 192 descriptor components are displayed.
- `static/js/explainer-scenes.mjs`: interpolation and SVG recorded-state replay.
- `static/data/explainer-pusht.json`: compact recorded states and provenance.
- `static/css/explainer.css`: responsive workbench and both themes.

Interaction concepts draw on
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and
[ViT-Explainer](https://vit-explainer.vercel.app/). Their model bundles, analytics
and image assets are not redistributed. The implementation is original to this
project and covered by the repository's Apache-2.0 license.

The open point-cloud presentation also takes visual inspiration from the
[TensorFlow Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)
and the [three.js point-sprite example](https://threejs.org/examples/webgl_points_sprites.html).
No code, images, textures or libraries from these references are redistributed.
The stage uses original lightweight SVG projection, with no added 3D dependency
or network asset request.

Checks: `node --test tests/explainer-model.test.mjs` and
`python3 scripts/check_site.py`. Preview using
`bash scripts/preview_site.sh 8000`, then open `/explainer.html`.

Rebuild the task-wall previews with `python scripts/build_explainer_wall_media.py`
(requires `imageio-ffmpeg`). The five preview videos total under 1 MB, retain the
original playback speeds, and load on demand. Fully buffered preview blobs make
scrubbing work with a simple local HTTP server as well as the hosted site; they
are released when the stage is removed. Original published videos remain intact.
