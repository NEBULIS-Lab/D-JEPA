# Project website maintenance

The standalone [interactive method explainer](explainer.html) is linked from
the header, resource buttons and method section. It supports the site's two
themes, a six-stage guided tour and inspectable SVG diagrams. Implementation,
illustrative-data conventions and controls are documented in [EXPLAINER.md](EXPLAINER.md).

`index.html` is a static project page inspired by the section-based layout of
NEBULIS Lab's BRACE page. Content, CSS and JavaScript are specific to D-JEPA;
BRACE experiments, publication metadata, scripts and third-party bundles are
not copied. No frontend build system, CDN, analytics or backend is required.

## Preview

From the repository root, run `bash scripts/preview_site.sh 8000` and open
`http://127.0.0.1:8000`. This does not publish or configure the domain. All site
asset paths are relative so the same folder can be served under `/D-JEPA/`.
Existing Markdown documents remain technical documentation, not page sections.

## Replace the author-artwork placeholders

1. Add final artwork to `static/images/` with a stable descriptive filename.
2. Set the corresponding `figures.<name>.src` in `site-content.json` to that
   relative path, update `alt` and set `status` to `ready`.
3. Run `python scripts/check_site.py` from the repository root.

The overview, motivation and two detail slots are intentionally marked as
awaiting author artwork. JavaScript replaces a slot only after its actual image
loads. Missing artwork never produces a broken-image rectangle. Main content
and all ten video panels remain readable without JavaScript. With JavaScript,
accessible task/scene buttons show one comparison at a time; arrow keys, Home
and End navigate the tabs. Playback is user-initiated and switching tabs pauses
the previous clip. The gallery precedes the diagnostic and numerical results.

## Palette and media

Shared tokens are in `static/css/site.css`: purple #A45AA8, secondary purple
#8464A5, lavender gray #777083, peach #ECA896, champagne #F6C17F and gold #FFDB67.
Purple-to-gold gradients highlight the primary resource button and divider.
Light-mode text uses darker colors for readability. Figures and videos retain
their original white background in both themes; no inversion or recoloring.

Brand masters stay in `assets/branding/` at repository root. The four identical
copies under `docs/static/images/branding/` make a docs-only deployment portable.
If a master changes, update its site copy and `ASSET_MANIFEST.json` together.

Published comparison clips are selected paired examples, not a random sample.
They retain their declared execution horizons and recorded stopping points. Numerical
figures retain their checked source measurements and geometry.
The page uses 15 manuscript quantitative SVG panels and the baseline-only
ranking diagnostic. Numeric panels have common data-area alignment, restrained
bar hatching and preserved source values. Row-label positions and axis-title
baselines retain their original alignment. Desktop rows contain five equal square
panels; narrower screens use two. Method colours, uncertainty definitions and
evaluation populations match the manuscript, including the separate independent
and mechanism evaluations. Table notes are enclosed in `tfoot` with a closing rule.

RoboTwin and driving use byte-identical copies of the current manuscript PNGs,
with the original experimental frames and bird's-eye geometry retained. Provenance,
transformations and hashes are recorded in `ASSET_MANIFEST.json`. The gallery
includes PushT, Reacher, Granular manipulation, bimanual grasping, three driving
scenes, unseen object geometry, changed object appearance and Cube coverage.
The two appendix driving videos compose saved camera observations with the same
simulated candidates as the paper, at 1x playback. Cube shows LeWM and TD-JEPA
both succeeding; it is task coverage, not a D-JEPA comparison. A seven-condition
observed-input plate and an eight-timepoint Cube plate accompany the results.
The three original core-task clips have header-only
terminology updates to TD-JEPA and DINO-WM, retaining every frame and the original
dimensions and frame rate. Their linked timelines have label-only updates with
byte-identical scene pixels. Source hashes and transformations are recorded;
lightweight posters are decoded frames, resized proportionally without retouching.
Clips are not preloaded, keeping initial page loading lightweight.

Non-title typography uses three shared sizes (14, 16 and 18 px). Metric values
and section headings have their own compact scale. Navigation, footer and cards
support light/dark themes; subtle entry and hover effects respect reduced-motion
preferences. Updating these local files does not deploy the website.

The hero alone uses the author-provided `d-jepa-full-logo2-light.svg` and
`d-jepa-full-logo2-dark.svg`; existing logo masters and other placements are
unchanged. The title area lists all seven authors, their three affiliations and
the corresponding-author email; the same authors are exposed as structured
`ScholarlyArticle` metadata. Gallery, diagnostic, coverage plates and the five-column results
grid share the main content width. The header/footer share a lavender surface;
the reproduction section has a separate background. Theme switching uses a
sun/moon control. First visits default to dark, independent of the operating
system theme; a saved manual choice takes precedence on later visits.
Each video has a large, keyboard-accessible play overlay that
hides during playback and returns on pause, alongside native video controls.
The hero Paper button currently opens the public manuscript-source repository;
replace that single URL with the arXiv record when it becomes available.
Hero resource buttons use local SVGs: the existing D-JEPA mark for the project,
Simple Icons v16 for arXiv and GitHub, and Hugging Face's official logo for the
model and dataset. No icon is loaded from a CDN at runtime.

Body paragraphs and long figure captions use justified alignment, with the
last line left aligned; hero text, headings, short plot labels and metric cards
retain their intended alignment. The grasping clip uses the same white canvas,
grey-blue/purple method rules and regular-weight labels as the other comparisons.
It retains every recorded frame at 2× playback and explicitly labels the held
successful terminal frame while the reference continues.

The page has no final-paper download, final citation or venue badge. Add them
only when author-approved material is available.
