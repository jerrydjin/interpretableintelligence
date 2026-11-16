# First five minutes: friction log

A newcomer's walk through the README quickstart, `ii try`, `ii get-clips` and the DROID playground (`ii run hf --model facebook/dinov2-small --no-score --steer`). Each entry says where the newcomer got confused or stuck and what changed. No verification gate, threshold or scoreboard rule changed; the fixes are to copy, empty states and errors only.

It was run on a fresh clone in a 4-core Linux x86_64 container with empty caches, not on a Mac. Mac-only steps (MPS, the `brew install uv` path) were not exercised. Timings: `uv sync` 11 s, `ii try` 9–14 s, `ii get-clips` 3 min (mostly the 175 MB download), and the DROID `--no-score --steer` run 4 min (capture 25 s, SAE 2.4 min, features 1 min).

## README

1. **`ii` is not on PATH.** The quickstart runs `uv run ii try`, but every hint the CLI printed afterwards said plain `ii get-clips`, which gives "command not found" after `uv sync`. *Fixed:* the hints print `uv run ii …` when the CLI was started through `uv run` and `ii …` otherwise, and the README says why and how to activate the venv.
2. **No vocabulary.** Feature, SAE/dictionary, ablation, held-out ("unseen"), verified/`real` and playground all appear in the first minute with no definition. *Fixed:* a five-line "Words you will see" list before the real-model step.
3. **The HF_TOKEN warning looks like a required step.** Both `ii get-clips` and `ii run hf` print "You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN…", right after a README line saying no account is needed. The warning is sent by the Hub, so the tool cannot reword it. *Fixed:* `ii get-clips` says beforehand that no account is needed and the warning only concerns rate limits; the README says the same.

## `ii try`

4. **The suggested decoy was not on screen.** Step 2 said "click an unverified feature on the right, such as flicker", but the page opens on a frame that lists band, stripe and speckle. *Fixed:* the step names the strongest unverified feature on the opening frame (band, on the default run).
5. **"Under the video" was the wrong place.** On a laptop-width window the Playground sits in the right column under the feature list. *Fixed:* "In the Playground panel" and "in the list above the Playground".
6. **Verified or real?** The terminal said "marked verified" and the page's badge says `real`. *Fixed:* the terminal names both ("verified (the page marks them 'real')"; the rest are "unverified").
7. **"Coloured squares" is ambiguous.** The practice clips are themselves coloured squares, so "the coloured squares on the video show where it fires" did not say which squares. *Fixed:* "the orange overlay on the frame marks where it fires".
8. **The steering result was two e-notation numbers.** "Prediction error went from 1.1e-4 to 2.8e-3" leaves a newcomer to work out whether that is a lot, which is the whole point of step 2. *Fixed:* the result adds "(about 26× the unedited error)" for stripe and "(about the same)" for band. On `hf` runs, where the reference is the model's own clean output, it says why the error starts at 0.

## `ii get-clips`

9. **`--n` below 20 produced clips no run accepts.** *Fixed:* `--help` says `ii run` needs at least 20, and a smaller download ends with a note to rerun with `--n 20` or more.
10. **The next step did not mention a second download.** *Fixed:* the hint says DINOv2-small is fetched once (about 90 MB) and that `--no-score` skips the checks, which is why nothing will be verified.
11. **Offline, the error did not say what was needed.** *Fixed:* after a connection error, a line says the first download needs huggingface.co and later runs use the cache.

## `ii run hf … --no-score --steer`

12. **"Scoring the 10" with `--no-score`.** The run opened with "3072 SAE features: scoring the 10 with the largest attribution", then scored nothing. *Fixed:* under `--no-score` it says nothing is checked now and that a run without `--no-score` checks those 10.
13. **SAE training looked stuck.** It logged every 500 steps with no estimate, about 2 minutes of silence each time here. *Fixed:* SAE training logs about every tenth of the way with "about N min left", like capture already did. The final line, which is stored with the SAE, is unchanged.
14. **The run ended on a cost report.** The last thing on screen was `video_s: 396.200000` and `stage score: not run`, with the viewer URL scrolled away. *Fixed:* after the report the run repeats the viewer URL, says whether the playground is on, and says what is verified: nothing under `--no-score` (and how to add the checks), otherwise how many features passed and how many got the ablation check.
15. **A missing clips folder gave a bare usage error.** "Invalid value for '--clips': Directory 'clips/droid' does not exist." *Fixed:* `clips folder not found`, plus the `ii get-clips --out …` (or `ii generate-clips` for `fake`) command that makes one.
16. **Too few clips did not say why 20.** *Fixed:* the error explains that some clips are held out to check features and points at `ii get-clips`.

## DROID playground

17. **The Playground was three screens down.** A real model lists up to 40 features per frame, each a card, so the Playground (and its Steer button) sat far below the fold. *Fixed:* past 12 features the list shows the strongest 6 (plus the selected one) and a "Show all 40 features firing on this frame" link. Runs with 12 or fewer features per frame, such as the practice run, are unchanged.
18. **"SAE is not marked usable" on a run that was never checked.** On an unscored run the Playground warned the SAE was not usable, which reads as a failed check. *Fixed:* unscored runs say no check has run and no feature is verified, so what you see is a lead, not a finding.
19. **"Each missed at least one check" on unscored features.** The Features page said all 3072 features had failed a check when none had been checked. *Fixed:* an unscored run says none has been checked yet. On scored runs the note now also covers features the ablation check skipped.
20. **An empty frame was a dead end.** "No feature fired on this frame" now adds "Move the slider or pick another clip."

## Elsewhere

21. **A mistyped run folder in `ii view`.** "run folder not found: runs/tyr" now lists the run folders next to it, or says that `ii run` / `ii try` writes one.

## Not changed

- Thresholds, gates, scoring, `scoreboard.json` and `split.json` are untouched. Every message above describes the gates as they already are.
- Clip and feature names on the DROID run are `feature 2934` and so on until scoring labels them. That is correct for an unscored run, and the Features page already says how to score it.
- The README's "about 10 minutes" for the DROID `--no-score` run was 4 minutes here. It is left as is, since a laptop CPU can be slower than this container.
