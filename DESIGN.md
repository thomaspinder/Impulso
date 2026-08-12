---
name: Impulso Documentation
description: Warm-paper, hairline-ruled documentation where the figures carry the ink and the chrome recedes.
colors:
  oxblood: "#7a2e2a"
  oxblood-deep: "#5f231f"
  oxblood-wash: "#f4e7e4"
  ink: "#2a2320"
  body: "#423b36"
  muted: "#6b6058"
  hairline: "#e6dfd6"
  surface: "#f3efe9"
  paper: "#faf9f7"
  lifted-oxblood: "#8a332e"
  ledger-blue: "#31649e"
  ochre: "#9a7020"
  plum: "#8f4a85"
  olive: "#5a7a2e"
typography:
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Oxygen, Ubuntu, Droid Sans, Helvetica Neue, sans-serif"
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace"
  figure-claim:
    fontFamily: "Spectral, Georgia, Palatino, DejaVu Serif, serif"
    fontSize: "11.5pt"
    fontWeight: 600
  figure-evidence:
    fontFamily: "Public Sans, Helvetica Neue, Arial, DejaVu Sans, sans-serif"
    fontSize: "9pt"
rounded:
  card: "8px"
---

# Design System: Impulso Documentation

## Overview

**Creative North Star: "The Calibrated Ledger"**

The documentation inherits its worldview from its own figures. The Calibrated Ledger — warm
off-white paper, near-black umber ink, hairline rules, and a single oxblood accent on the number
that matters — is fully realised in the qc-core matplotlib style that every tutorial loads, and the
site chrome exists to serve those figures, not to compete with them. The system is quiet, rigorous,
and warm: off-white paper instead of clinical white, umber ink instead of pure black, rigour
without coldness.

Two layers make up the surface. The figure layer is normative and precisely specified (the
`ledger.mplstyle` palette, light and dark). The chrome layer is a themed Sphinx site (shibuya,
Radix scales, system fonts) that should read as recessive infrastructure: navigation, search, and
code chrome disappear behind the evidence. Where the two layers once disagreed — the theme's
pink-leaning Radix crimson accent versus the ledger's oxblood — the ledger now wins in code:
`docs/stylesheets/extra.css` re-tones the theme's entire accent scale to oxblood in both modes.

**Key Characteristics:**
- Warm paper ground (#faf9f7) with umber ink (#2a2320) — never pure white or pure black
- One accent: oxblood marks the number that matters; everything else is ink on paper
- Hairline rules (1px, #e6dfd6) do the structural work shadows would do elsewhere
- Serif states the claim, sans carries the evidence
- Executed notebooks: every figure on the site is reproduced at build time, not pasted

## Colors

A warm monochrome ledger — paper, ink, and hairlines — with oxblood as the sole accent and a short,
muted categorical cycle for multi-series figures.

### Primary
- **Oxblood** (#7a2e2a): THE accent. Single-series emphasis in figures, the credible-interval band,
  the estimate that matters. Normatively also the chrome accent family (links, active states).
- **Oxblood Deep** (#5f231f): hover/pressed shade of the accent.
- **Oxblood Wash** (#f4e7e4): pale tint for distribution fills and full-posterior backgrounds.

### Neutral
- **Ink** (#2a2320): near-black warm umber. Figure titles, emphasis, the median rule.
- **Body** (#423b36): running text and axis labels.
- **Muted** (#6b6058): metadata — ticks, captions, and the "Other" series when folding extras.
- **Hairline** (#e6dfd6): 1px rules — spines, grids, dividers, borders.
- **Surface** (#f3efe9): insets and asides, one warm step up from paper.
- **Paper** (#faf9f7): the page ground. White-adjacent, never cream, never #fff.

### Categorical Series
The validated five-slot cycle for multi-series figures (passes lightness-band, chroma-floor, CVD
ΔE ≥ 8, normal-vision ΔE ≥ 15, and 3:1-contrast-on-paper checks):
- **Lifted Oxblood** (#8a332e): slot 1 — brand oxblood raised one lightness step to sit in the
  series band; the true oxblood stays reserved for single-series emphasis.
- **Ledger Blue** (#31649e): slot 2.
- **Ochre** (#9a7020): slot 3.
- **Plum** (#8f4a85): slot 4.
- **Olive** (#5a7a2e): slot 5.

### Legacy & Drift
- **Legacy Brand Red** (#870d14, retired 2026-08-12): the pre-ledger brand red. Fully removed
  from the docs surface — the landing-page CTA card and the seven tutorial copies are gone, along
  with the `--linkcolor` support in `docs/stylesheets/extra.css`. It survives only in `README.md`,
  which is outside this surface and a separately recorded open decision. Do not reintroduce.
- **Radix crimson (resolved drift)**: shibuya's `accent_color: "crimson"` once rendered
  pink-leaning links (≈#e93d82). As of 2026-08-12, `extra.css` overrides the full `--crimson-*`
  scale — Radix step semantics, anchored on Oxblood #7a2e2a (light) and #cf6f60 (dark) — so every
  accent-derived element (links, active nav, hovers, focus, banner) speaks oxblood. The
  `accent_color` setting merely names the token family being overridden.

Dark mode re-lights the same roles on warm near-black: paper #201b19, surface #2a2320, ink #f2ece6,
body #d8cfc7, muted #9a8d84, hairline #3a322e, oxblood lifts to #cf6f60 (deep #a35043, wash
#2f211e), and the cycle becomes #cf6f60 / #5b8fd6 / #b3883b / #b877ab / #78933f. The site chrome
follows the reader's preference (`color_mode: "auto"`).

**The One Accent Rule.** Oxblood is the only voice raised above the ink. When one number matters,
it wears oxblood; nothing else on the page competes for that channel.

**The Re-Lit, Not Inverted Rule.** Dark mode re-lights the same named roles on warm near-black. It
never inverts hues or swaps roles.

## Typography

**Chrome Font:** system sans stack (-apple-system, BlinkMacSystemFont, Segoe UI, …)
**Figure Claim Font:** Spectral (with Georgia, Palatino fallbacks)
**Figure Evidence Font:** Public Sans (with Helvetica Neue, Arial fallbacks)
**Mono Font:** ui-monospace stack (SFMono-Regular, Menlo, Consolas, …)

**Character:** The chrome deliberately has no typographic personality — the system sans disappears.
Type becomes expressive only inside figures, where Spectral appears at the exact moment a claim is
made.

### Hierarchy
- **Figure claim** (semibold 600, 11.5pt, left-aligned): the figure title. Written as a finding
  ("Low water halves Rhine tonnage"), never as a variable name. Set via `serif_title()`.
- **Figure evidence** (regular, 9pt base / 8.5pt ticks and legends): axis labels in body
  (#423b36), tick labels and captions in muted (#6b6058).
- **Chrome headings and body**: theme-governed system sans; sizes are shibuya defaults and are not
  independently specified.
- **Code** (mono): inline and block code in the mono stack; API names in tutorial code are
  auto-linked to the reference (sphinx-codeautolink).

**The Serif States, Sans Measures Rule.** Spectral (serif) states the claim — titles only. Public
Sans carries the evidence — labels, ticks, legends. No third role exists.

## Layout

The chrome is shibuya's three-region documentation layout: left navigation, a reading column, and a
right in-page table of contents, all theme-governed. The project's own layout decisions live in the
figures and content structure:

- Figures are compact and consistent: 5.5in × 3in at 250 dpi, constrained layout, saved tight with
  transparent background so they sit directly on the page's paper ground in either mode.
- Figures, tables, and equations are numbered (`numfig`) and cross-referenced; equations are
  labelled `$$…$$ (label)` and cited with `{eq}`.
- Content follows Diátaxis: tutorials (executed notebooks), how-to guides, explanation, reference.
- Legends sit outside the data area — one entry per row to the right (`legend_right`) or a single
  row below (`legend_below`); the canvas grows rather than the axes shrinking.

## Elevation & Depth

Flat, hairline-ruled. Depth is conveyed by 1px rules (#e6dfd6) and one warm surface step
(#f3efe9 on #faf9f7) — never by shadows at rest. Shadows exist only on transient overlays
(dropdowns, popovers, search) and belong to the theme, not the system. In figures, the same
doctrine: no drop shadows, no 3D, structure from horizontal hairline grid rules only (0.6pt,
y-axis only — the ledger's lines), with top and right spines removed.

**The Flat-At-Rest Rule.** Surfaces are flat at rest. A shadow may appear only on a transient
overlay, never as decoration on static content.

## Shapes

Gently rounded containers (8px radius on cards), square figures, and rounded line ends. Line work
is the form language: hairline spines (1.0pt) on the bottom and left only, outward tick marks
(3.5pt) as calibration marks, and round cap-styles on data lines (1.8pt) so series end softly.
Bars and stacked segments separate with paper-colored edges (0.8pt) — a surface gap, not an
outline. Scatter points wear paper edges for the same reason.

## Components

The component philosophy is **recessive and precise**: chrome components are quiet instruments;
the figures and code are the interface.

### Links
- **Accent:** the oxblood family, delivered through the overridden theme scale — Oxblood #7a2e2a
  in light, #cf6f60 in dark — for links, current nav items, and active TOC entries.
- **Hover:** deepens in light (Oxblood Deep #5f231f), lifts in dark (#dd8171); no underline games.

### Cards / Containers
- **Corner Style:** gently rounded (8px).
- **Background:** paper, or surface (#f3efe9) for insets.
- **Border:** 1px hairline (#e6dfd6); a heavier left rule may mark a container's voice.
- **Shadow Strategy:** none at rest (see Elevation & Depth).

### CTA Card (retired 2026-08-12)
Removed from the entire docs surface — landing page and all seven tutorial copies — per
PRODUCT.md's no-explicit-CTAs posture, together with its `.consulting-cta` styles and
`--linkcolor`. Do not build new components in this pattern.

### Admonitions
Theme-supplied (shibuya). Keep them recessive: they support the prose, they do not decorate it.
Callout text stays faithful to the surrounding prose — no added claims.

### Code Blocks
Mono stack on a quiet ground; copy button (sphinx-copybutton) appears on hover. API names inside
tutorial code link to their reference pages — code is navigation, not just illustration. Executed
notebook cells carry a quiet oxblood-wash left rule (the accent scale's a6 step) marking them as
run-at-build source; the marker whispers, it never takes the solid accent.

### The Estimate Figure (signature)
The brand's signature chart, mirrored by `qc_core.plotting.estimate_plot`: a full posterior in
Oxblood Wash, the credible interval in translucent oxblood (30%), the density outline stroked in
oxblood (1.8pt, round caps), the median as an ink rule, a hairline baseline, and no left spine or
y-ticks — the density axis means nothing to the reader, so it isn't drawn.

### Figure Legends
Frameless — identity comes from the mark swatch; text stays in body ink. Never boxed.

## Do's and Don'ts

### Do:
- **Do** load the ledger style (`plotting.use_ledger_style()`) in every notebook that draws; it is
  the visual system.
- **Do** write figure titles as claims, left-aligned, in Spectral via `serif_title()` — state the
  finding, not the variable.
- **Do** keep scatter plots to at most three series (only the first three cycle slots pass the
  all-pairs separation test); fold a sixth-plus series into Muted as "Other" or facet the chart.
- **Do** use warm neutrals everywhere: paper #faf9f7 and ink #2a2320, with hairline #e6dfd6 rules.
- **Do** let dark mode re-light the same roles (the palette defines every dark counterpart);
  figures save transparent so they sit on either ground.

### Don't:
- **Don't** reintroduce the Radix crimson pink (≈#e93d82) or extend Legacy Brand Red (#870d14);
  oxblood #7a2e2a is the accent, and `extra.css` re-tones the theme's crimson scale to keep it so.
- **Don't** use pure #fff or #000 anywhere; the system's ground is warm paper and its ink is umber.
- **Don't** frame legends, add drop shadows to static content, or draw top/right spines — the
  ledger is flat and hairline-ruled.
- **Don't** invent a sixth categorical hue; the five-slot cycle is validated and closed.
- **Don't** add CTA-styled blocks; the surviving consulting card is legacy pending removal, not a
  pattern.
