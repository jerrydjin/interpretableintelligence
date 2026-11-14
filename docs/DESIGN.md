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

The Rust server embeds `static/` at compile time. After a CSS change, rebuild it with `cargo build -p interpretableintelligence-server`, or `tests/test_contract.py::test_static_asset_bytes_match` will fail against a stale local binary.
