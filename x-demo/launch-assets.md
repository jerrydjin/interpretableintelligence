# X launch assets: steering a verified feature live

A 30 to 45 second screen recording, a four-image thread and a short post. British English, no em dashes.

## Read this first

**Inputs that were missing.** `x-demo/shot-list.md` is not in the repo on any branch, or in Notion or Linear, so this draft is built from the viewer itself. The five verified DROID features are not committed anywhere either. To reproduce them I ran the README's DROID path in a fresh session (`ii get-clips`, then `ii run hf --model facebook/dinov2-small` with default settings, seed 0). It verified **one** feature, not five. That session had no vision-model API key, so the labels came from the CLIP word list instead of explain-then-simulate, and 9 of the 10 shortlisted features failed the label-holdout check. The copy below therefore leaves the feature names as fields to fill from your run (`{F1}` and so on). I have not written any feature name or effect size I could not check.

**Framing that has to stay accurate.**

- DINOv2-small is a **vision model**, not a robot model. The clips are real robot video (DROID). Say "a vision model watching real robot video". If your five came from `lerobot/smolvla_base`, you can say "the vision encoder a robot policy sees through", but still not that the robot does anything differently.
- Steering moves the model's **output** (its final patch tokens), not the robot's actions. The viewer itself says "Exploration only. Steering never marks a feature real; only the scored gates do." The copy keeps to that.
- "Verified" means the five offline checks the viewer lists: the dictionary is usable, the feature shows up across many clips, its name holds on clips the model never trained on, removing it matters more than removing any of 100 random features, and an untrained model has nothing like it. It is not a rollout finding.

**Something the recording has to work around.** In the reproduction, removing an *unverified* feature (2934) on the same unseen frame moved the output twice as much as removing the verified one: mean cosine shift 0.033 against 0.016. A "remove the verified one, then a decoy, and watch the difference" beat would show the opposite of what it claims. The script uses that honestly instead: a big move is not the evidence, the checks are. See shot 5.

**Layout problem.** The playground panel sits at the bottom of the per-frame feature list, about 2,000 px below the video (40 features at 1600 x 900). You cannot see the Steer button and the blue "moved" overlay in one frame. The script cuts between them. Moving the panel under the video in the viewer would make this one continuous take.

Run-level numbers below come from that same default run (100 clips, seed 0). Labelling does not change capture or the SAE, so they should match yours. Check them against your `scoreboard.json` before posting.

| Number | Value | Where it comes from |
| --- | --- | --- |
| Clips | 100 DROID clips, 4 s each (80 train, 20 unseen) | `scoreboard.json` `run` |
| Dictionary size | 3,072 features | README, `hf` SAE default |
| Prediction quality kept with the SAE swapped in | 86.9% (bar is 70%) | `sae.loss_recovered` |
| Random controls per feature | 100 | `n_random` |
| Verified | `{N}` (you said 5) | Features page, "Verified" count |

### Fields to fill

| Field | What to put | Where to find it |
| --- | --- | --- |
| `{F1}` | Hero feature label. Pick the verified feature whose label is concrete and visibly matches where it fires (the gripper, the arm, a hand), and whose strongest moment is on an unseen clip. | Features page, then the feature's page |
| `{F1_ID}` | Its feature id | Feature page header |
| `{F1_CLIP}`, `{F1_FRAME}` | Its strongest moment | "Opens droid-NNN at frame N, where it is strongest." |
| `{F1_SHIFT}` | Mean cosine shift when you press Steer with Remove on that frame | Playground readout, "the prediction moved X on average" |
| `{DECOY}`, `{DECOY_SHIFT}` | An **unverified** feature on the same frame whose Remove moves the output at least as much as `{F1}` | Same frame's list, then Steer |
| `{F2}` to `{F5}` | The other four verified labels | Features page |

If no unverified feature on that frame moves the output as much as `{F1}`, use the alternative shot 5 in the script.

## 1. Screen recording script (about 40 s)

**Setup.** 1600 x 900 browser window, light mode, "Numbers" toggle off, cursor highlight on. Have the run finished and the viewer open before you hit record:

```bash
uv run ii view runs/droid --clips clips/droid --steer
```

Open the hero feature's strongest moment directly: `http://127.0.0.1:7860/#/clips/{F1_CLIP}?frame={F1_FRAME}&feature={F1_ID}`. Press Steer once off camera so the model is warm. Each Steer runs the model live on CPU and returns in a second or two.

Captions are burned in, bottom left, one line each. No voice-over needed. If you add one, read the captions.

| Time | Shot | On screen | Caption |
| --- | --- | --- | --- |
| 0 to 4 s | 1. Cold open | Features page, scrolled to the "Verified" heading and its `{N}` cards. Slow push in. | A vision model watching real robot video. `{N}` features it learned, each verified. |
| 4 to 10 s | 2. One feature | Click the `{F1}` card. Hold on the title, the green "real" badge and the 3D map of where it fires. | This one is `{F1}`. The bars show where it fires. |
| 10 to 15 s | 3. The moment | Click "Watch and steer it". The clip opens on the unseen frame with `{F1}` highlighted. Let two or three frames play, then pause on `{F1_FRAME}`. | A clip the model never trained on, at the moment it fires hardest. |
| 15 to 24 s | 4. Steer it | **Cut** to the playground panel. Edit = Remove. Click Steer. **Cut** back up to the video: the blue grid appears where the model's output moved. Hold. | Remove it, live. Blue is where the model's output moved. |
| 24 to 32 s | 5. The catch | **Cut** to the list. Pick `{DECOY}` (grey "unverified" badge). Remove, Steer. Cut to the video: the blue is as big or bigger. Show the readout: `{DECOY_SHIFT}` against `{F1_SHIFT}`. | This unverified one moves it just as much. Steering is the fun part. It isn't the evidence. |
| 32 to 40 s | 6. The checks | Back to the `{F1}` page. Slow scroll past the "Verified." line and the five checks. End card: repo URL and the two commands. | Verified means 5 checks, against 100 random features and an untrained model. `ii run`, then `ii view --steer`. |

**Alternative shot 5** (no decoy moves it as much): switch the edit to Scale at 3 and press Steer. The blue spreads. Caption: "Boost it 3 times and the change spreads. Exploration, not evidence: only the checks mark it real."

**Do not** cut the decoy beat to imply that verified features move the output more. In the default run they did not.

## 2. Four-image thread

Each image is a 1600 x 900 screenshot at 2x (3200 x 1800). Reference frames from the default run, with its single verified feature standing in for `{F1}`, are in `x-demo/reference/`. Retake them on your run with your features.

**Post 1** (image: `1-feature.png`, the `{F1}` feature page with the "real" badge and the 3D map)

> I pointed a vision model at 100 clips of real robot arms working in kitchens and offices, and pulled out the features it learned.
>
> {N} survived every check. This one is "{F1}".
>
> Then you can steer it live. 🧵

**Post 2** (image: `2-remove-verified.png`, the unseen clip with the blue "moved" grid, plus `2-remove-verified-panel.png` cropped in beside it)

> Pick a feature, remove it, press Steer. The model runs again on that frame, and the blue squares show where its output moved.
>
> This clip was held out. The dictionary never saw it in training.

**Post 3** (image: `3-remove-decoy.png` and its panel: an unverified feature moving the output as much or more)

> The catch: this unverified feature moves the output just as much.
>
> A big move is not evidence. Steering is for exploring. What marks a feature real is a set of checks it has to pass first.

**Post 4** (image: `4-scoreboard.png`, the Features page: "dictionary usable", why the rest are unverified, the Verified row)

> The five checks: the dictionary has to keep 86.9% of the model's quality, the feature has to fire across many clips, its name has to hold on unseen clips, removing it has to beat 100 random features, and an untrained model can't have it.
>
> Open source: {REPO_URL}

## 3. Short post (goes with the video)

> A vision model watching real robot video. I pulled out the features it learned, kept the {N} that passed every check, and steered one live.
>
> Blue is where its output moved. Steering is for exploring; the checks are the evidence.
>
> {REPO_URL}

Shorter alternative:

> Steer a feature inside a vision model, live, on real robot video. Only {N} passed all five checks, and the page tells you which.
>
> {REPO_URL}

## Before posting

- [ ] Fill every `{…}` field from your run and retake the four images on it.
- [ ] Confirm the run-level numbers (86.9%, 100 controls, 100 clips) against your `scoreboard.json`.
- [ ] Check that `{DECOY_SHIFT}` really is at least `{F1_SHIFT}` on the frame you recorded, or use the alternative shot 5.
- [ ] Make sure `{REPO_URL}` is public, and open the demo site in a private window if you link it.
- [ ] DROID attribution: the clips are from DROID (Khazatsky et al.), CC BY 4.0 upstream. Add "Clips: DROID (CC BY 4.0)" to the end card or a reply.
- [ ] Re-read once for "robot model", "the robot does", "proves" and "causal". None of them belong in this copy.
