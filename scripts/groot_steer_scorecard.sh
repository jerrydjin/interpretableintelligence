#!/usr/bin/env bash
# Steering scorecard on the real GR00T model (task 1028), end to end, on any GPU box set up with
# scripts/groot_gpu_setup.sh (Colab, or a rented GPU with a shell). Guide: docs/groot-steer-scorecard.md.
#
#   scripts/groot_steer_scorecard.sh CAPTURE [WORK]
#
# CAPTURE  the unpacked groot-1028-v9-n100 capture, its .tar.xz, or an `ii retrain-sae` output. Its sae/ is the
#          dictionary whose feature ids are measured (default 6262,7353,7775: the PR 50 picks on the capture SAE).
# WORK     output folder (default runs/groot-1028-steer): clips/, run/, run/steering/.
#
# Environment knobs: FEATURES, EPISODES (24), HELDOUT_FRAC (0.75), CONTROLS (19), CONTROLS_FROM (random|features), BOOST (ii default),
# SEED0 (60000), PLUS_DIR (LIBERO-plus checkout), MAX_STEPS, RESULTS (a folder to copy the small result files to).
# Every step resumes: rerun the same command after a disconnect.
set -euo pipefail

CAPTURE="${1:?usage: groot_steer_scorecard.sh CAPTURE [WORK]}"
WORK="${2:-runs/groot-1028-steer}"
FEATURES="${FEATURES:-6262,7353,7775}"
EPISODES="${EPISODES:-24}"
HELDOUT_FRAC="${HELDOUT_FRAC:-0.75}"
CONTROLS="${CONTROLS:-19}"
SEED0="${SEED0:-60000}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
CAPTURE_SHA256="0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e"  # data/groot-1028.md

mkdir -p "$WORK"
if [ -f "$CAPTURE" ]; then
  TOP="$(tar -tJf "$CAPTURE" 2>/dev/null | head -1 | cut -d/ -f1 || true)"
  case "$TOP" in ""|.) echo "cannot tell the top folder of $CAPTURE; unpack it and pass the folder" >&2; exit 2;; esac
  if [ ! -d "$WORK/$TOP/sae" ]; then
    echo "== unpack $CAPTURE"
    if [ "$(basename "$CAPTURE")" = "groot-1028-v9-n100.tar.xz" ]; then
      echo "$CAPTURE_SHA256  $CAPTURE" | sha256sum -c -
    fi
    tar -xJf "$CAPTURE" -C "$WORK"
  fi
  CAPTURE="$WORK/$TOP"
fi
[ -f "$CAPTURE/sae/config.json" ] || [ -f "$CAPTURE/config.json" ] || { echo "no SAE under $CAPTURE" >&2; exit 2; }

echo "== 1/3 record $EPISODES closed-loop episodes on LIBERO-plus 1028 (seeds $SEED0+i)"
# --trim-out: ii run needs every clip to yield the same windows, so the run reads each episode's first N steps,
# N = the shortest episode in whole 16-step chunks.
ii record-libero --out "$WORK/clips" --task 1028 --episodes "$EPISODES" --seed0 "$SEED0" --trim-out "$WORK/clips-run" \
  ${PLUS_DIR:+--plus-dir "$PLUS_DIR"} ${MAX_STEPS:+--max-steps "$MAX_STEPS"}

echo "== 2/3 capture DiT block 5 on those clips, with the capture's SAE frozen in"
ii run groot --clips "$WORK/clips-run" --out "$WORK/run" --sae-from "$CAPTURE" --heldout-frac "$HELDOUT_FRAC" --no-score --no-serve

echo "== 3/3 steering scorecard on held-out episodes"
ii steer-scorecard "$WORK/run" --clips "$WORK/clips-run" --features "$FEATURES" --controls "$CONTROLS" ${CONTROLS_FROM:+--controls-from "$CONTROLS_FROM"} ${BOOST:+--boost "$BOOST"}

python3 - "$WORK" "$FEATURES" <<'PYEOF'
import json, sys
from pathlib import Path

work, features = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split(",") if x]
stats = json.loads((work / "run/features/stats.json").read_text())
record = json.loads((work / "clips-run/record.json").read_text())
freq = {int(r["id"]): int(r["freq"]) for r in stats["features"]}
print()
print(f"episodes: {len(record['episodes'])} recorded, {record['n_success']} successes, each cut to its first {record['trimmed_to']} steps; "
      f"input parity pass={record['parity']['pass']}")
print(f"SAE on the new clips: held-out EV {stats['ev_heldout']:.4f}, dead fraction {stats['dead_frac']:.3f} "
      "(capture held-out eval: EV ~0.998, ~456 of 8192 alive, so dead ~0.94). Far off means the hook or inputs differ.")
print(f"measured features, held-out tokens they fire on (of {stats['n_heldout']}):", {f: freq.get(f) for f in features})
PYEOF

if [ -n "${RESULTS:-}" ]; then
  mkdir -p "$RESULTS"
  cp "$WORK/run/steering/scorecard.json" "$WORK/run/steering/effects.parquet" "$WORK/clips-run/record.json" "$RESULTS/"
  cp "$WORK/run/features/stats.json" "$RESULTS/features-stats.json"
  cp "$WORK/run/manifest.json" "$RESULTS/run-manifest.json"
  echo "copied results to $RESULTS"
fi
