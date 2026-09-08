# Project website maintenance

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
and all three video panels remain readable without JavaScript.

## Palette and media

Shared tokens are in `static/css/site.css`: purple #A64CA6, secondary purple
#8464A5, lavender gray #777083, peach #ECA896, champagne #F6C17F and gold #FFDB67.
Purple-to-gold gradients highlight the primary resource button and divider.
Light-mode text uses darker colors for readability. Figures and videos retain
their original white background in both themes; no inversion or recoloring.

Brand masters stay in `assets/branding/` at repository root. The four identical
copies under `docs/static/images/branding/` make a docs-only deployment portable.
If a master changes, update its site copy and `ASSET_MANIFEST.json` together.

Published comparison clips are selected baseline-failure / D-JEPA-success cases,
not a random sample. They retain full declared fixed-action horizons. Existing
numerical figures and media were copied byte-for-byte from the checked visual
library; provenance and hashes are in `ASSET_MANIFEST.json`. No new simulation,
video edit, frame synthesis or rendering was performed for this page.

The page has no final-paper download, final citation, venue badge or unfinished
experiment results. Add them only when author-approved material is available.
