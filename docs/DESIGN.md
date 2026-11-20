# Viewer design: handover notes

These notes cover the visual language of the `ii` viewer, the browser front end in `src/interpretableintelligence/static/`. That front end is shared by `ii view`, the Rust server, `ii export` and `ii export-site`. Read this before changing how the viewer looks.

The source of truth for every value below is the comment block and `:root` tokens at the top of `static/style.css`.

## Who it is for

The first user is a learner. The viewer should make mechanistic interpretability feel easy and inviting, never dense or technical. Every feature can be explored and steered. The strict scoreboard gates only decide the "verified" badge, and a failed check always suggests a next experiment.

## Type: two fonts, five sizes, two weights

| Role | Font | Weight | Used for |
| --- | --- | --- | --- |
| Serif | Newsreader | 400 only | Page and section titles, and a few sentences you read (the steering target) |
| Sans | Geist | 400 and 500 | Everything else: ledes, names, buttons, labels, numbers, tables, inline commands |

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
| `--real` (+ `--real-wash`, `--real-line`) | `#2f6b4b` | Only where something is verified: the tick, a passed check, the check ring, a verified feature's standing card and the playground's "Verified feature" mark |
| `--unv-bg` (sand) | `#f7efdc` | The next experiment for a feature that is not verified yet |

The heatmap reads `--signal-rgb` and `--steer-rgb` from the stylesheet. `isoTerrain()` in `app.js` hard-codes its block colours, so update it alongside them.

## Shapes and signature elements

- The nav is plain text links; the current section is underlined in ember. There is no header rule.
- Cards use a 10px radius (`--radius`) and a 1px hairline instead of a border, and do not lift on hover. Lists use hover fills rather than dividers.
- Status tags (`.badge`) are small outlined labels with 6px corners (`--radius-sm`), not filled pills.
- Inline code is Geist in ink with no background. Commands to copy sit in a dark block (`--plate`).
- Ledes are two-tone: the claim in ink, the context in grey.
- The signature graphic is **isometric activation blocks**: a feature's activation on its top frame, drawn as a little terrain beside the frame.
- The logo is two isometric columns, the two i's drawn as activations: a short ember one and a tall ink one.
- There is one button shape: ink with small corners (`.button`); `.button-quiet` is its outlined second choice. The playground's Steer button uses it too.
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

`ii export-site` builds `demo-site/`. The public site (home, Explore, your-model page, Research blog) and the live playground share one stylesheet, `static/site.css`: the same tokens, one header, one button, one card and one code block. The playground's own `live.css` holds only its components. The site does not load the viewer's `style.css`. Regenerate the site with `make demo-site` whenever `static/site.css` changes.

The site has four sections, and every page belongs to one; the nav underlines that section.

| Section | Pages | What it is |
| --- | --- | --- |
| Home (`/`) | | One sentence, three ways in, the latest result, what is coming |
| Try it | `/droid/` | A real run (DINOv2 on DROID) in the viewer, with a precomputed playground |
| Research | `/results/` hub, `/groot/`, `/vjepa/` | Studies on real models, each as question, answer, evidence, how to rerun. `/explore/` forwards to `/groot/` |
| Run it yourself | `/start/` | Install, `ii try`, a real model, the commands |
| Demos | `/demos/`, `/fake/`, `/picker/` | Synthetic examples with a known answer, kept apart from the research |

Pages inside a section show a small link back to it above the title. When in doubt, remove rather than add: labels, tags, borders and accents.

## References Jerry chose

He picked these on 13 Nov 2026, from screenshots of about 28 homepages:

- **Anthropic**: formal warmth, and the main influence. It shows in the serif plus sans pairing, the ivory paper and the generous space. On 17 Nov Jerry asked to keep the colours but make the components look less like Anthropic's, which is why the pill nav, pill buttons and filled tags went.
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
- Exactly one `[data-testid=badge]` on a feature page: the check ring and the playground's verified mark are not badges.
- The `standing`, `standing-next`, `standing-count`, `check-trail`, `steer-idea`, `steer-idea-run` and `steer-verified` test ids.
- `data-real` on `[data-testid=guide]`.
- A visible top-clip table.
- The exact `frame-readout` text (`frame 6 · t 1 · window 0`) and `sae-summary` text. Both may be hidden behind the Numbers switch, but their text must not change.
- A feature selected from the URL (`?feature=`) stays selected, even if it is not verified.
- No sideways scroll at 390px wide on the clip, features and feature pages.

The Rust server embeds `static/` at compile time. After a CSS change, rebuild it with `cargo build -p interpretableintelligence-server`, or `tests/test_contract.py::test_static_asset_bytes_match` will fail against a stale local binary.
