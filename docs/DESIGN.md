# Viewer design: handover notes

These notes cover the visual language of the `ii` viewer, the browser front end in `src/interpretableintelligence/static/`. That front end is shared by `ii view`, the Rust server, `ii export` and `ii export-site`. Read this before changing how the viewer looks.

The source of truth for every value below is the comment block and `:root` tokens at the top of `static/style.css`.

## Who it is for

The first user is a learner. The viewer should make mechanistic interpretability feel easy and inviting, never dense or technical. Every feature can be explored and steered. The strict scoreboard gates only decide the "verified" badge, and a failed check always suggests a next experiment.

## Type: two fonts, five sizes, two weights

| Role | Font | Weight | Used for |
| --- | --- | --- | --- |
| Serif | Newsreader | 400 only | Page and section titles, ledes, sentences you read |
| Sans | Geist | 400 and 500 | Everything you use: names, buttons, labels, numbers, tables |

The sizes are tokens: `--t-small` 13px, `--t-body` 16px, `--t-lead` 21px, `--t-head` 30px and `--t-display` 52px. Below 860px the head, display and lead sizes shrink.

Serif text uses one tracking value, `--serif-track`. Sans text is never tracked.

Some things are deliberately absent: there is no monospace, no uppercase labels and no third weight. Labels are sentence-case Geist at 13px in grey. Do not add a new size or font without removing one.

All fonts are vendored as woff2 under the SIL OFL in `static/fonts/`, so the viewer works offline.

## Colour

| Token | Value | Meaning |
| --- | --- | --- |
| `--paper` | `#faf9f5` | Warm ivory page |
| `--sheet` / `--well` | `#ffffff` / `#f0eee6` | Raised cards / quiet fills |
| `--ink`, `--ink-2`, `--ink-3` | `#141413`, `#5e5d59`, `#8f8d86` | Text, from strongest to quietest |
| `--signal` | `#e8582a` (ember) | Activation. The only colour for "where a feature fires" |
| `--steer` | `#3a5bd9` (cobalt) | Steering and the playground |
| `--real` | `#2f6b4b` | Only on the verified tick |

The heatmap reads `--signal-rgb` and `--steer-rgb` from the stylesheet. `isoTerrain()` in `app.js` hard-codes its block colours, so update it alongside them.

## Shapes and signature elements

- The nav is a row of pills, with the active page filled in ink. There is no header rule.
- Cards use a 14px radius (`--radius`) and a 1px line shadow instead of a border. Lists use hover fills rather than dividers.
- Ledes are two-tone: the claim in ink, the context in grey.
- The signature graphic is **isometric activation blocks**: a feature's activation on its top frame, drawn as a little terrain beside the frame.
- The logo is two isometric columns, the two i's drawn as activations: a short ember one and a tall ink one.
- There is one button shape, an ink pill (`.button`). The playground's Steer button uses it too.

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
- **A feature**: the title and verdict, one "Watch it fire" button (or "Watch and steer it"), the hero, then three sections in order: what the checks say, where it fires hardest (six frames), and the numbers.
- **Below 1180px** the clip rail becomes a strip that scrolls sideways inside itself. **Below 860px** everything is one column and the header drops the status. **Below 560px** the wordmark shows only the logo. No page may scroll sideways at 390px; a test checks this.

## The demo site

`ii export-site` builds `demo-site/`. Its fake viewer page is the viewer's own `index.html` with a slim demo bar above it, so the viewer and the demo never drift apart. `static/demo.css` styles only the demo prose pages (`.prose`, `.demo-bar`, `.score-line`) on the same tokens, and must never restyle viewer classes. Regenerate the site with `make demo-site` whenever `static/` changes.

## References Jerry chose

He picked these on 13 Nov 2026, from screenshots of about 28 homepages:

- **Anthropic**: formal warmth, and the main influence. It shows in the serif plus sans pairing, the ivory paper, the pill nav and the generous space.
- **Cursor**: clean layout.
- **Baseten**: graphics. This led to the isometric blocks.
- **1X**: good visuals and imagery.

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
- `data-real` on `[data-testid=guide]`.
- A visible top-clip table.
- The exact `frame-readout` text (`frame 6 · t 1 · window 0`) and `sae-summary` text. Both may be hidden behind the Numbers switch, but their text must not change.
- A feature selected from the URL (`?feature=`) stays selected, even if it is not verified.
- No sideways scroll at 390px wide on the clip, features and feature pages.

The Rust server embeds `static/` at compile time. After a CSS change, rebuild it with `cargo build -p interpretableintelligence-server`, or `tests/test_contract.py::test_static_asset_bytes_match` will fail against a stale local binary.
