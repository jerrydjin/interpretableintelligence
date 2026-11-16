# Viewer design: handover notes

These notes cover the visual language of the `ii` viewer, the browser front end in `src/interpretableintelligence/static/`. That front end is shared by `ii view`, the Rust server, `ii export` and `ii export-site`. Read this before changing how the viewer looks.

The source of truth for every value below is the comment block and `:root` tokens at the top of `static/style.css`.

## Who it is for

The first user is a learner. The viewer should make mechanistic interpretability feel easy and inviting, never dense or technical. Every feature can be explored and steered. The strict scoreboard gates only decide the "verified" badge, and a failed check always suggests a next experiment.

## Type: two fonts, five sizes, two weights

| Role | Font | Weight | Used for |
| --- | --- | --- | --- |
| Sans | Geist | 400 and 500 | Everything you read and use. Titles are 500 with tight tracking (`--serif-track`, plus tighter values on the demo's display and section heads); sentences, ledes and labels are 400 |
| Mono | Geist Mono | 400 only | Code and commands (`code`, the quickstart block). Never for labels or numbers in the viewer |

The sizes are tokens: `--t-small` 13px, `--t-body` 16px, `--t-lead` 21px, `--t-head` 30px and `--t-display` 52px. Below 860px the head, display and lead sizes shrink.

Until 16 Nov 2026 titles were set in Newsreader, a serif. Together with the ivory paper, the clay-orange activation colour and pill buttons, that read as a copy of Anthropic's site, so the serif was dropped. `--serif` is kept as an alias of `--display` (Geist) so older panels inherit the new title role.

Some things are deliberately absent: there are no uppercase labels and no third weight. Labels are sentence-case Geist at 13px in grey. Do not add a new size or font without removing one.

All fonts are vendored as woff2 under the SIL OFL in `static/fonts/`, so the viewer works offline.

## Colour

| Token | Value | Meaning |
| --- | --- | --- |
| `--paper` | `#f6f6f3` | Soft neutral page (not ivory) |
| `--sheet` / `--well` | `#ffffff` / `#ecece7` | Raised cards / quiet fills |
| `--ink`, `--ink-2`, `--ink-3` | `#111110`, `#51514c`, `#85857e` | Text, from strongest to quietest |
| `--signal` | `#f2a900` (saffron) | Activation. The only colour for "where a feature fires". `--signal-ink` `#8f6200` if it ever has to be text |
| `--steer` | `#5b4ae0` (violet) | Steering and the playground |
| `--real` (+ `--real-wash`, `--real-line`) | `#23704a` | Only where something is verified: the tick, a passed check, the check ring, a verified feature's standing card and the playground's "Verified feature" mark |
| `--unv-bg` (sand) | `#f6efdf` | The next experiment for a feature that is not verified yet |

The heatmap reads `--signal-rgb` and `--steer-rgb` from the stylesheet. `isoTerrain()` in `app.js` hard-codes its block colours, and the logo is inline SVG in `static/index.html` and `static/favicon.svg`, so update them alongside.

## Shapes and signature elements

- The nav is plain text links; the current page sits on a small white tile with a hairline. On the demo pages the header is sticky with a hairline under it, and below 860px the links become one row that scrolls sideways inside itself.
- Corners are small: chips and badges 6px (`--chip`), buttons 8px (`--radius-sm`), cards 10px (`--radius`). Cards use a 1px line shadow instead of a border; hover darkens the line rather than lifting the card. There are no pills except the Numbers switch track.
- Ledes are two-tone: the claim in ink, the context in grey.
- The signature graphic is **the patch grid**: a real frame with the model's patch grid over it, dimmed, and the patches where a verified feature fires lit in saffron. The home page hero is one (picked from the DROID export by `_hero_figure` in `export_site.py`, preferring "white robot arm" on a held-out clip), and the viewer's heatmap is the same idea. The isometric activation blocks stay as the feature cards' art.
- The logo is two isometric columns, the two i's drawn as activations: a short saffron one and a tall ink one.
- There is one button shape, an ink block (`.button`), plus a quiet white variant on the demo home. The playground's Steer button uses the ink one.
- The **check ring** is the verified seal: one arc per check (dictionary, many clips, name holds, beats random, learned), green when passed. A verified feature closes the ring and a tick pops into the middle; anything else shows `4/5` in the middle. The arcs draw in once, and not at all under `prefers-reduced-motion`.

## Plain words first, numbers on request

The viewer shows a plain-language reading by default. Raw measurements stay in the page but carry the class `num`, and only show when the reader turns on the **Numbers** switch in the header. The switch sets `data-numbers="on"` on `<html>` and is remembered in `localStorage` (`ii.numbers`). The page works the same if storage is blocked.

Behind the switch:

- The header's model sha, layer, split sizes and `ev_heldout` / `loss_recovered`. By default the header shows only "model · dictionary usable" with a coloured dot, and the full explanation is in its tooltip.
- The frame readout (`frame · t · window`). By default the player shows "Frame 3 of 16".
- Estimate effects on feature rows. The ablation chip always stays visible, with a tooltip that says what it measures.
- The "3 real · 9 unverified" count on the features page.

The feature page's "The numbers" section (the score tiles and the full top-clip table) stays visible, but sits last, after the plain-language checks. When you add a raw value anywhere, give it the `num` class or a plain-language companion.

## Page layouts

- **Explore** (a clip): clips | player | what the model notices. Each clip in the rail has a thumbnail of its first frame, and held-out clips are tagged "unseen". The playground sits under the feature list, so the order reads: watch, pick, steer. With no feature in the URL, the first **verified** feature on the frame is selected, so a newcomer starts on something real.
- **Features**: the hardest moments (when the run has them), then a dictionary callout, then two card grids, **Verified** and **Still exploring**. Each card shows the frame where the feature fires hardest beside its isometric blocks. Cards load their art when they scroll into view, so a run with thousands of features stays fast.
- **A feature**: the title and verdict, then the **standing card**, the hero, then three sections in order: what the checks say, where it fires hardest (six frames), and the numbers. The standing card is the reward or the invitation. It shows the check ring and a trail of the five checks, then two columns.
  - Verified (green wash): what the badge earns (it beat its random look-alikes on unseen clips, `ii explain` can cite it, the next bar is `ii steer-scorecard` or a rollout), and an edit to try in the playground.
  - Not verified (white, next experiment on sand): what held it back (the first failed check in order, else the first one that did not run), and the next experiment from `guide.js`.
  - Its one button opens the top clip with a playground edit set up and run (`?try=scale:3`), or just watches the clip when the playground is off. A failed check never ends the page: there is always something to do next.
- **Playground**: a verified feature gets a green "Verified feature" mark and a green idea card; any other feature gets an "An experiment to try" card on sand. Both come from `IIGuide.experiments()`, which picks two edits from what the checks said. After one runs, the card says "Next, try" and offers the other, so there is always a next experiment.
- **Below 1180px** the clip rail becomes a strip that scrolls sideways inside itself. **Below 860px** everything is one column and the header drops the status. **Below 560px** the wordmark shows only the logo. No page may scroll sideways at 390px; a test checks this.

## The demo site

`ii export-site` builds `demo-site/`. Its fake viewer page is the viewer's own `index.html` with a slim demo bar above it, so the viewer and the demo never drift apart. `static/demo.css` styles only the demo prose pages (`.prose`, `.demo-bar`, `.score-line`) on the same tokens, and must never restyle viewer classes. Regenerate the site with `make demo-site` whenever `static/` changes.

## References Jerry chose

He picked these on 13 Nov 2026, from screenshots of about 28 homepages:

- **Anthropic**: formal warmth. It was the main influence until 16 Nov 2026, when Jerry said the site looked like a straight copy; what remains is the generous space and plain, warm copy.
- **Cursor**: clean layout. Now the main influence: one sans, mono for code, hairlines, small corners, a quiet sticky header.
- **Baseten**: graphics. This led to the isometric blocks.
- **1X**: good visuals and imagery. This led to the real-footage hero on the demo home.

## What was rejected, and why

1. **Dark "instrument"** (mono readouts, registration ticks) and **serif "specimen catalogue"**: rejected as too information-dense.
2. **Light "highlighter" cards**: rejected as "the most generic presentation". Jerry wants a real brand identity, with no shadcn-style components.
3. **Risograph halftone** (fluoro inks, Bricolage Grotesque): rejected as horrendous.
4. **The first version of the current direction** (PR #18): its direction was accepted, but it had too many type variations (3 families and 19 sizes). PR #19 cut it down to the system above.
5. **Jargon up front** (the first pass of PR #19): the type was right, but every row and the header led with `ablation 1.49e-8`, `ev_heldout`, `t · window`, and the feature list was names and badges only. The learner pass moved raw numbers behind the Numbers switch, added clip thumbnails and feature cards, put verified features first, and fixed sideways scrolling on phones.

The lessons:

- Ask for concrete references before proposing a direction.
- Prefer fewer elements and plain language, with raw numbers one click away.
- Keep the type system strict.

## Constraints that tests rely on

The Playwright tests in `tests/test_view.py`, `test_steer.py` and `test_guide.py` pin the following. Restyle freely, but keep these:

- Every `data-testid`.
- The badge text "real" / "unverified".
- A visible ablation chip on frame-feature rows.
- A steer caption containing "prediction change".
- A `guide-verdict` element containing "Verified".
- Exactly one `[data-testid=badge]` on a feature page: the check ring and the playground's verified mark are not badges.
- The `standing`, `standing-next`, `standing-count`, `check-trail`, `steer-idea`, `steer-idea-run` and `steer-verified` test ids.
- `data-real` on `[data-testid=guide]`.
- A visible top-clip table.
- The exact `frame-readout` text (`frame 6 · t 1 · window 0`) and `sae-summary` text. Both may be hidden behind the Numbers switch, but their text must not change.
- A feature selected from the URL (`?feature=`) stays selected, even if it is not verified.
- No sideways scroll at 390px wide on the clip, features and feature pages.

The Rust server embeds `static/` at compile time. After a CSS change, rebuild it with `cargo build -p interpretableintelligence-server`, or `tests/test_contract.py::test_static_asset_bytes_match` will fail against a stale local binary.
