#!/usr/bin/env bash
# Run the README quickstart exactly as written, then check what a newcomer would see.
#
#   uv sync
#   uv run ii try
#
# `ii try` serves the viewer until Ctrl+C and opens a browser on it. Here BROWSER points at a tiny
# recorder instead of a real browser, so the script learns the exact URL `ii try` opened, fetches
# that page and the API calls the page makes, presses Steer once, then stops the server with
# SIGINT (Ctrl+C). CI runs this on fresh Linux and macOS runners (.github/workflows/ci.yml); it
# also works on a laptop from a fresh clone. Needs bash, curl and uv; nothing else.
set -euo pipefail

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
log="$work/try.log"
opened="$work/opened-url"
pid=""

cleanup() {
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

fail() {
  echo "--- ii try output ---"
  cat "$log" 2>/dev/null || true
  echo "---------------------"
  echo "quickstart: FAIL: $*" >&2
  exit 1
}

printf '#!/bin/sh\nprintf "%%s\\n" "$1" > "%s"\n' "$opened" > "$work/browser"
chmod +x "$work/browser"
export BROWSER="$work/browser"

echo "quickstart: \$ uv sync"
uv sync

echo "quickstart: \$ uv run ii try"
started=$SECONDS
# Job control on: the server gets its own process group with SIGINT not ignored (a plain `&` in a
# script ignores it), so Ctrl+C below is the same signal a terminal sends.
set -m
uv run ii try >"$log" 2>&1 &
pid=$!
set +m

deadline=$((SECONDS + ${QUICKSTART_TIMEOUT:-900}))
until [ -s "$opened" ]; do
  kill -0 "$pid" 2>/dev/null || fail "ii try exited before it opened the viewer"
  [ "$SECONDS" -lt "$deadline" ] || fail "ii try did not open the viewer within ${QUICKSTART_TIMEOUT:-900}s"
  sleep 1
done
url="$(head -n 1 "$opened")"
base="${url%%#*}"
base="${base%/}"
echo "quickstart: ii try opened $url after $((SECONDS - started))s"
cat "$log"

# The page the browser was sent to, and the files and API calls it loads first.
page="$(curl -fsS "$base/")" || fail "GET $base/ failed"
case "$page" in *"/static/app.js"*) ;; *) fail "the viewer page does not load app.js" ;; esac
for asset in static/app.js static/style.css static/guide.js; do
  curl -fsS -o /dev/null "$base/$asset" || fail "GET /$asset failed"
done
curl -fsS "$base/api/run" >"$work/run.json" || fail "GET /api/run failed"

# The route ii try opened: #/clips/<clip>?frame=<n>&feature=<id>
fragment="${url#*#}"
clip="$(printf '%s' "$fragment" | sed -n 's|^/clips/\([^?]*\)?.*|\1|p')"
frame="$(printf '%s' "$fragment" | sed -n 's|.*[?&]frame=\([0-9]*\).*|\1|p')"
feature="$(printf '%s' "$fragment" | sed -n 's|.*[?&]feature=\([0-9]*\).*|\1|p')"
[ -n "$clip" ] && [ -n "$frame" ] && [ -n "$feature" ] || fail "unexpected start route: $fragment"
curl -fsS "$base/api/clips/$clip" >"$work/clip.json" || fail "GET /api/clips/$clip failed"
curl -fsS "$base/api/features/$feature" >"$work/feature.json" || fail "GET /api/features/$feature failed"
curl -fsS "$base/api/clips/$clip/heatmap?frame=$frame&feature=$feature" >"$work/heat.json" \
  || fail "GET heatmap failed"
curl -fsS -o "$work/frame.png" "$base/api/clips/$clip/frames/$frame.png" || fail "GET frame png failed"

# Step 1 the terminal suggests: Remove the feature and press Steer.
curl -fsS -X POST -H 'Content-Type: application/json' \
  -d "{\"clip_id\": \"$clip\", \"window\": 0, \"features\": [$feature], \"kind\": \"ablate\"}" \
  "$base/api/steer" >"$work/steer.json" || fail "POST /api/steer failed"

# The practice model has three planted features; the run must mark exactly those verified.
uv run --no-sync python - "$work" runs/try/scoreboard.json "$feature" <<'PY' || fail "result checks failed"
import json, sys
from pathlib import Path

work, scoreboard_path, opened_feature = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
run = json.loads((work / "run.json").read_text())
assert run.get("steer", {}).get("enabled") is True, f"playground is off: {run.get('steer')}"
scoreboard = json.loads(scoreboard_path.read_text())
real = sorted(f["label"] for f in scoreboard["features"] if f.get("real") is True)
assert real == ["corner", "pulse", "stripe"], f"verified features are {real}, expected corner, pulse, stripe"
real_ids = {f["id"] for f in scoreboard["features"] if f.get("real") is True}
assert opened_feature in real_ids, f"ii try opened feature {opened_feature}, which is not verified"
feature = json.loads((work / "feature.json").read_text())
assert feature.get("id") == opened_feature, feature
assert json.loads((work / "heat.json").read_text()).get("heatmap"), "empty heatmap"
assert (work / "frame.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "frame is not a PNG"
steer = json.loads((work / "steer.json").read_text())
assert isinstance(steer, dict) and "error" not in steer, steer
devices = sorted({str(stage.get("device")) for stage in json.loads((scoreboard_path.parent / "manifest.json").read_text()).get("stages", {}).values() if isinstance(stage, dict)})
print(f"quickstart: device {', '.join(devices)}; verified features {real}; opened feature {opened_feature}; steer returned {sorted(steer)}")
PY

# Ctrl+C, as the terminal tells the user to do: SIGINT to the whole foreground group.
kill -INT -- "-$pid"
for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$pid" 2>/dev/null; then fail "ii try did not stop on Ctrl+C"; fi
pid=""
echo "quickstart: OK"
