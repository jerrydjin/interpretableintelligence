const state = {
  run: null,
  clip: null,
  clipId: null,
  frame: 0,
  featureId: null,
  playing: false,
  timer: null,
  token: 0,
  heatToken: 0,
  heatPack: null,
  heatClip: null,
  steer: { kind: "scale", value: 3, result: null, busy: false, show: false, error: "", token: 0 },
};

const STEER_KINDS = {
  ablate: { label: "Remove", help: "Set the feature to zero wherever it fires.", min: 0, max: 0, step: 1, value: 0 },
  scale: { label: "Scale", help: "Multiply the feature by this amount where it fires. 0 removes it, 2 doubles it.", min: -5, max: 10, step: 0.1, value: 3 },
  clamp: { label: "Clamp", help: "Force the feature to this value on every patch, even where it was silent.", min: -5, max: 10, step: 0.1, value: 1 },
  add: { label: "Add", help: "Add this amount of the feature on every patch.", min: -5, max: 10, step: 0.1, value: 1 },
};

const app = document.getElementById("app");

function iiRoot() {
  const root = document.documentElement.getAttribute("data-ii-root");
  return root ? root.replace(/\/$/, "") : "";
}

function apiUrl(path) {
  const root = iiRoot();
  if (!root) return path;
  if (path === "/api/run") return `${root}/run.json`;
  let match = path.match(/^\/api\/features\/(\d+)$/);
  if (match) return `${root}/features/${match[1]}.json`;
  match = path.match(/^\/api\/features\/(\d+)\/viz$/);
  if (match) return `${root}/features/${match[1]}.png`;
  match = path.match(/^\/api\/clips\/([^/]+)$/);
  if (match) return `${root}/clips/${match[1]}.json`;
  match = path.match(/^\/api\/clips\/([^/]+)\/frames\/(\d+)\.png$/);
  if (match) return `${root}/clips/${match[1]}/frames/${match[2]}.png`;
  match = path.match(/^\/api\/clips\/([^/]+)\/media$/);
  if (match) return `${root}/clips/${match[1]}/media.mp4`;
  return path;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function fmtActivation(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function fmtEffect(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 10) return value.toFixed(3);
  if (magnitude >= 0.001) return value.toFixed(4);
  if (magnitude >= 0.0001) return value.toFixed(6);
  return value.toExponential(2);
}

function fmtUnit(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function fmtSmall(value) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) >= 0.01 ? value.toFixed(3) : value.toExponential(1);
}

function fmtFlag(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "—";
}

function fmtPercentile(value) {
  if (!Number.isFinite(value)) return "—";
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function displayLabel(feature) {
  return feature.label ? feature.label : `feature ${feature.id}`;
}

function badge(feature) {
  const real = feature.badge === "real";
  const status = !real && typeof feature.status === "string" ? feature.status.trim() : "";
  const shown = real ? "real" : status && status.toLowerCase() !== "real" ? status : "";
  const kind = real ? "real" : shown ? "status" : "unverified";
  return el("span", { class: `badge badge-${kind}`, "data-testid": "badge", "data-kind": kind }, shown || "unverified");
}

async function boot() {
  try {
    const response = await fetch(apiUrl("/api/run"), { cache: "no-store" });
    if (!response.ok) throw new Error(`run request failed (${response.status})`);
    state.run = await response.json();
  } catch (error) {
    renderError(error.message || "Could not load the run.");
    return;
  }
  renderHeader();
  window.addEventListener("hashchange", route);
  route();
}

function renderHeader() {
  const run = state.run;
  document.getElementById("run-id").textContent = run.run_id;
  const model = run.model || {};
  const name = model.name || model.adapter || "unknown model";
  const sha = typeof model.sha === "string" && model.sha ? model.sha.slice(0, 8) : "";
  const modelEl = document.getElementById("run-model");
  modelEl.textContent = sha ? `${name} · ${sha}` : name;
  modelEl.title = model.sha || name;
  document.getElementById("run-layer").textContent = `layer ${run.layer}`;
  const split = run.split || {};
  document.getElementById("split-sizes").textContent = `train ${split.n_train} · held-out ${split.n_heldout}`;
  const saeEl = document.getElementById("sae-summary");
  if (run.scoreboard === false || !run.sae) {
    saeEl.dataset.usable = "false";
    saeEl.textContent = "no scoreboard";
  } else {
    const sae = run.sae;
    const usable = sae.usable === true;
    saeEl.dataset.usable = usable ? "true" : "false";
    saeEl.textContent = `${usable ? "usable" : "not usable"} · ev_heldout ${fmtUnit(sae.ev_heldout)} · loss_recovered ${fmtUnit(sae.loss_recovered)}`;
  }
  if (guide()) saeEl.title = guide().sae(run.scoreboard === false ? null : run.sae).result;
  document.title = `${run.run_id} · interpretableintelligence`;
}

function route() {
  stop();
  const raw = (location.hash || "#/").replace(/^#/, "");
  const [path, query = ""] = raw.split("?");
  const params = new URLSearchParams(query);
  markNav(path.startsWith("/features") ? "features" : "explore");
  if (path === "/features") {
    renderIndex();
    return;
  }
  const feature = path.match(/^\/features\/(\d+)$/);
  if (feature) {
    showFeature(Number(feature[1]));
    return;
  }
  const clip = path.match(/^\/clips\/(.+)$/);
  if (clip) {
    showClip(decodeURIComponent(clip[1]), Number(params.get("frame") || 0), params.get("feature"));
    return;
  }
  const first = state.run.clips[0];
  if (!first) {
    renderError("This run has no clips.");
    return;
  }
  location.replace(`#/clips/${encodeURIComponent(first.clip_id)}`);
}

function markNav(section) {
  for (const link of document.querySelectorAll(".nav a[data-nav]")) {
    if (link.dataset.nav === section) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

// Theme colours live in style.css; the canvas-free heatmap reads them from there.
function themeRgb(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function renderError(message) {
  app.replaceChildren(el("p", { class: "error" }, message));
}

function renderIndex() {
  const features = state.run.features;
  const real = features.filter((feature) => feature.badge === "real").length;
  const moments = renderMoments();
  const list = el("ul", { class: "index-list" });
  for (const feature of features) {
    const link = el(
      "a",
      {
        href: `#/features/${feature.id}`,
        "data-testid": "feature-index-link",
        "data-feature-id": feature.id,
        "data-label": feature.label || "",
        "data-candidate": feature.candidate ? "true" : "false",
      },
      [
        el("span", {}, [el("span", { class: "index-id" }, String(feature.id).padStart(2, "0")), displayLabel(feature)]),
        badge(feature),
      ],
    );
    list.append(el("li", {}, link));
  }
  const page = [
    moments,
    el("section", { class: "feature-index", "data-testid": "feature-index" }, [
      el("h1", {}, "Features"),
      el("p", { class: "lede" }, [
        el("b", {}, real === 1 ? "One verified feature." : `${real === 0 ? "No" : real} verified features.`),
        ` ${features.length - real} more to explore. Every one can be opened and steered; the checks only decide the badge.`,
      ]),
      el("p", { class: "kicker", "data-testid": "real-count" }, `${real} real · ${features.length - real} unverified`),
      indexGuide(features),
      list,
    ]),
  ].filter(Boolean);
  app.replaceChildren(...page);
  document.title = `features · ${state.run.run_id} · interpretableintelligence`;
}

function renderMoments() {
  const moments = (state.run.moments || []).slice(0, 10);
  if (!moments.length) return null;
  const rows = moments.map((moment) => {
    const task = moment.task_text ? el("span", { class: "moment-task" }, moment.task_text) : el("span", { class: "moment-task" }, "");
    return el("li", {}, el("a", {
      href: `#/clips/${encodeURIComponent(moment.clip_id)}?frame=${moment.frame}`,
      "data-testid": "hardest-moment",
      "data-clip-id": moment.clip_id,
      "data-clip-name": moment.clip_name || "",
      "data-frame": moment.frame,
      "data-loss": moment.loss,
    }, [
      el("span", { class: "moment-name" }, moment.clip_name || moment.clip_id),
      task,
      el("span", { class: "moment-range" }, `${moment.frame_start}–${moment.frame_end}`),
      el("span", { class: "moment-loss" }, Number(moment.loss).toFixed(4)),
    ]));
  });
  return el("section", { class: "hardest", "data-testid": "hardest-moments" }, [
    el("h2", {}, "Hardest moments"),
    el("p", { class: "lede" }, [el("b", {}, "Where the model was most surprised."), " Start here: these frames have the highest prediction error."]),
    el("ol", { class: "hardest-list" }, rows),
  ]);
}

async function showFeature(id) {
  const token = ++state.token;
  const response = await fetch(apiUrl(`/api/features/${id}`), { cache: "no-store" });
  if (token !== state.token) return;
  if (!response.ok) {
    renderError(`Feature ${id} is not in this run.`);
    return;
  }
  const feature = await response.json();
  if (token !== state.token) return;
  renderFeature(feature);
}

function renderFeature(feature) {
  const rows = feature.top_clips.length
    ? feature.top_clips.map((clip) =>
        el("tr", { "data-testid": "top-clip", "data-clip-id": clip.clip_id }, [
          el("td", {}, el("a", { href: `#/clips/${encodeURIComponent(clip.clip_id)}?frame=${clip.frame}&feature=${feature.id}` }, clip.clip_id)),
          el("td", {}, clip.split || "—"),
          el("td", {}, String(clip.frame)),
          el("td", {}, `${clip.window} / ${clip.t}`),
          el("td", {}, fmtActivation(clip.value)),
        ]),
      )
    : [el("tr", {}, el("td", { colspan: "5" }, "This feature does not fire on any clip."))];
  const strip = el("div", { class: "thumb-strip", "data-testid": "top-thumbs" });
  for (const clip of feature.top_clips) {
    const image = el("img", {
      alt: `${clip.clip_id} frame ${clip.frame}`,
      src: apiUrl(`/api/clips/${encodeURIComponent(clip.clip_id)}/frames/${clip.frame}.png`),
      "data-testid": "top-thumb-image",
    });
    image.addEventListener("error", () => {
      image.replaceWith(el("span", { class: "thumb-missing", "data-testid": "top-thumb-missing" }, "no frame"));
    });
    strip.append(el("a", {
      class: "thumb",
      href: `#/clips/${encodeURIComponent(clip.clip_id)}?frame=${clip.frame}&feature=${feature.id}`,
      "data-testid": "top-thumb",
      "data-clip-id": clip.clip_id,
      "data-frame": clip.frame,
    }, [image, el("span", {}, `${clip.clip_id} · ${fmtActivation(clip.value)}`)]));
  }
  app.replaceChildren(
    el("article", {
      class: "feature-page",
      "data-testid": "feature-page",
      "data-feature-id": feature.id,
      "data-label": feature.label || "",
      "data-badge": feature.badge,
    }, [
      el("p", { class: "kicker" }, `Feature ${feature.id}${feature.label_model ? ` · named by ${feature.label_model}` : ""}`),
      el("div", { class: "feature-title" }, [el("h1", {}, displayLabel(feature)), badge(feature)]),
      featureLede(feature),
      featureHero(feature),
      feature.viz
        ? el("img", { class: "viz", alt: "", src: apiUrl(`/api/features/${feature.id}/viz`), "data-testid": "feature-viz" })
        : null,
      el("h2", { class: "section-title" }, "What the checks say"),
      feature.scored === false
        ? el("p", { class: "kicker", "data-testid": "scoreboard-missing" }, "No scoreboard record. Shown as unverified.")
        : null,
      featureGuide(feature),
      el("h2", { class: "panel-title" }, "The numbers"),
      el("dl", { class: "score", "data-testid": "scoreboard" }, [
        scoreItem("distinct_clips_top20", fmtMaybe(feature.distinct_clips_top20, String), feature.distinct_clips_top20),
        scoreItem("label_holds_heldout", fmtFlag(feature.label_holds_heldout), feature.label_holds_heldout),
        scoreItem("ablation_effect", fmtEffect(feature.ablation_effect), feature.ablation_effect),
        scoreVersus(feature),
        scoreItem("absent_in_random_init", fmtFlag(feature.absent_in_random_init), feature.absent_in_random_init),
      ]),
      el("h2", { class: "section-title" }, "Where it fires hardest"),
      feature.top_clips.length ? strip : null,
      el("table", { class: "top-table" }, [
        el("thead", {}, el("tr", {}, ["clip", "split", "frame", "window / t", "activation"].map((label) => el("th", {}, label)))),
        el("tbody", {}, rows),
      ]),
    ]),
  );
  document.title = `${displayLabel(feature)} · interpretableintelligence`;
}

// Plain-English guide (static/guide.js). The badge never comes from here.
function guide() {
  return typeof window !== "undefined" && window.IIGuide ? window.IIGuide : null;
}

// Text with `code` spans.
function richText(text) {
  return String(text).split("`").map((part, index) => (index % 2 ? el("code", {}, part) : part));
}

function steerHrefFor(feature) {
  if (!steerEnabled() || !feature.top_clips || !feature.top_clips.length) return null;
  const top = feature.top_clips[0];
  return `#/clips/${encodeURIComponent(top.clip_id)}?frame=${top.frame}&feature=${feature.id}`;
}

function featureGuide(feature) {
  const g = guide();
  if (!g) return null;
  const run = state.run;
  const ctx = { steerHref: steerHrefFor(feature), sae: run.scoreboard === false ? null : run.sae };
  const read = g.feature(feature, ctx);
  const items = read.gates.map((gate) => {
    const mark = gate.passed === true ? "pass" : gate.passed === false ? "fail" : "skip";
    const symbol = { pass: "✓", fail: "✗", skip: "–" }[mark];
    const word = { pass: "Passed", fail: "Failed", skip: "Not run" }[mark];
    let next = null;
    if (gate.next) {
      const hint = !gate.next.href && /playground/.test(gate.next.text) && !steerEnabled() && !iiRoot()
        ? [" ", el("span", { class: "guide-hint" }, richText("Start the viewer with `ii view RUN --steer` to turn the playground on."))]
        : [];
      next = el("p", { class: "guide-next", "data-testid": "guide-next" }, [
        el("strong", {}, "Try next: "),
        ...richText(gate.next.text),
        gate.next.href ? [" ", el("a", { href: gate.next.href, "data-testid": "guide-steer" }, `${gate.next.action || "Open"} →`)] : null,
        ...hint,
      ].flat());
    }
    return el("li", { class: `guide-gate guide-${mark}`, "data-testid": "guide-gate", "data-gate": gate.key, "data-passed": mark }, [
      el("span", { class: "guide-mark", "aria-label": word, title: word }, symbol),
      el("div", { class: "guide-body" }, [
        el("h3", {}, gate.title),
        el("p", { class: "guide-result" }, richText(gate.result)),
        el("details", { class: "guide-asks" }, [el("summary", {}, "What this check asks"), el("p", {}, richText(gate.asks))]),
        next,
      ]),
    ]);
  });
  return el("section", { class: "guide", "data-testid": "guide", "data-real": read.real ? "true" : "false" }, [
    read.status ? el("p", { class: "guide-status", "data-testid": "guide-status" }, read.status) : null,
    el("ol", { class: "guide-gates" }, items),
    el("p", { class: "guide-note" },
      "These checks only decide the verified badge. Every feature stays open to explore and steer, whatever its badge."),
  ]);
}

function indexGuide(features) {
  const g = guide();
  if (!g) return null;
  const run = state.run;
  const dictionary = g.sae(run.scoreboard === false ? null : run.sae);
  const summary = g.summary(features);
  const reasons = summary.lines.length
    ? el("p", { class: "guide-why", "data-testid": "guide-why" }, [
        "Why the rest are unverified: ",
        summary.lines.map((line) => line.text).join(", "),
        ". Open a feature to see what to try next.",
      ])
    : null;
  return el("div", { class: "guide guide-index", "data-testid": "guide-index" }, [
    el("p", { class: `guide-sae guide-${dictionary.passed === true ? "pass" : dictionary.passed === false ? "fail" : "skip"}`, "data-testid": "guide-sae" }, [
      el("strong", {}, dictionary.passed === true ? "Dictionary: usable. " : dictionary.passed === false ? "Dictionary: not usable. " : "Dictionary: not checked. "),
      ...richText(dictionary.result),
      dictionary.next ? [" ", el("strong", {}, "Try next: "), ...richText(dictionary.next.text)] : [],
    ].flat()),
    reasons,
  ]);
}

// Two-tone lede: the first sentence in ink, the rest in grey.
function twoTone(text) {
  const match = String(text).match(/^(.+?[.!?])\s+(.*)$/s);
  return match ? [el("b", {}, match[1]), ` ${match[2]}`] : [el("b", {}, text)];
}

function featureLede(feature) {
  const g = guide();
  if (g) {
    const run = state.run;
    const read = g.feature(feature, { steerHref: steerHrefFor(feature), sae: run.scoreboard === false ? null : run.sae });
    return el("p", { class: "lede", "data-testid": "guide-verdict" }, twoTone(read.verdict));
  }
  if (feature.badge === "real") {
    return el("p", { class: "lede" }, [
      el("b", {}, "Verified."),
      " It shows up across many clips, its name holds on clips the model never trained on, removing it matters more than removing a random feature, and an untrained model has nothing like it.",
    ]);
  }
  if (feature.scored === false) {
    return el("p", { class: "lede" }, [el("b", {}, "Not scored yet."), " Treat it as exploratory: open its top clips and steer it to see what it does."]);
  }
  const status = typeof feature.status === "string" ? feature.status.trim() : "";
  const failed = status.toLowerCase().startsWith("failed:")
    ? status.slice(7).split(",").map((part) => part.trim()).filter(Boolean)
    : [];
  const count = failed.length ? `It misses ${failed.length === 1 ? "one check" : `${failed.length} checks`}` : "It has not passed every check";
  return el("p", { class: "lede" }, [
    el("b", {}, "Not verified yet."),
    ` ${count}, so treat it as exploratory. You can still look at where it fires and steer it.`,
  ]);
}

// Hero: the frame where the feature fires hardest, next to its activation as isometric blocks.
function featureHero(feature) {
  const top = feature.top_clips && feature.top_clips[0];
  if (!top) return null;
  const href = `#/clips/${encodeURIComponent(top.clip_id)}?frame=${top.frame}&feature=${feature.id}`;
  const shot = el("a", { class: "hero-shot", href, "data-testid": "feature-hero-shot" }, [
    el("img", { alt: `${top.clip_id} frame ${top.frame}`, src: apiUrl(`/api/clips/${encodeURIComponent(top.clip_id)}/frames/${top.frame}.png`) }),
  ]);
  shot.querySelector("img").addEventListener("error", (event) => event.target.remove());
  const terrain = el("div", { class: "hero-terrain", "data-testid": "feature-terrain" });
  const caption = el("p", { class: "hero-caption" }, `${top.clip_id} · frame ${top.frame} · peak ${fmtActivation(top.value)}`);
  const token = state.token;
  fetchHeat(top.clip_id, top.frame, feature.id).then((rows) => {
    if (token !== state.token || !rows || !rows.length) return;
    terrain.innerHTML = isoTerrain(rows);
  }).catch(() => {});
  return el("div", { class: "hero" }, [shot, el("div", { style: "position: relative" }, [terrain, caption])]);
}

async function fetchHeat(clipId, frame, featureId) {
  if (iiRoot()) {
    const pack = await heatPack(clipId);
    return (pack && pack[`${frame}:${featureId}`]) || null;
  }
  const response = await fetch(
    `/api/clips/${encodeURIComponent(clipId)}/heatmap?frame=${frame}&feature=${featureId}`,
    { cache: "no-store" },
  );
  if (!response.ok) return null;
  const payload = await response.json();
  return payload.heatmap || null;
}

// Isometric blocks, one per patch, height proportional to activation. Numbers only, so innerHTML is safe.
function isoTerrain(rows) {
  const h = rows.length;
  const w = rows[0].length;
  let max = 0;
  for (const row of rows) for (const value of row) if (value > max) max = value;
  const tile = Math.min(28, 360 / (w + h));
  const dx = tile * 0.866;
  const dy = tile * 0.5;
  const lift = tile * 4.2;
  const width = (w + h) * dx;
  const height = (w + h) * dy + lift + 4;
  const x0 = h * dx;
  const y0 = lift + 2;
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const base = [233, 230, 221];
  const hot = [242, 84, 27];
  const parts = [];
  for (let s = 0; s < w + h - 1; s += 1) {
    for (let r = 0; r < h; r += 1) {
      const c = s - r;
      if (c < 0 || c >= w) continue;
      const value = rows[r][c] > 0 && max > 0 ? rows[r][c] / max : 0;
      const z = value > 0 ? Math.max(2, value * lift) : 0;
      const k = 0.88;
      const cx = x0 + (c - r) * dx;
      const cy = y0 + (c + r) * dy;
      const top = [
        [cx, cy - z],
        [cx + dx * k, cy + dy * k - z],
        [cx, cy + 2 * dy * k - z],
        [cx - dx * k, cy + dy * k - z],
      ];
      const face = value > 0 ? mix([246, 205, 186], hot, Math.pow(value, 0.7)) : base;
      const pts = (list) => list.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
      if (z > 0) {
        const left = mix(face, [60, 30, 15], 0.22);
        const right = mix(face, [60, 30, 15], 0.1);
        parts.push(`<polygon points="${pts([top[3], top[2], [top[2][0], top[2][1] + z], [top[3][0], top[3][1] + z]])}" fill="${rgb(left)}"/>`);
        parts.push(`<polygon points="${pts([top[2], top[1], [top[1][0], top[1][1] + z], [top[2][0], top[2][1] + z]])}" fill="${rgb(right)}"/>`);
      }
      parts.push(`<polygon points="${pts(top)}" fill="${rgb(face)}"/>`);
    }
  }
  return `<svg viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}" role="img" aria-label="Activation on each patch, drawn as blocks">${parts.join("")}</svg>`;
}

function fmtMaybe(value, render) {
  if (value === null || value === undefined) return "—";
  return render(value);
}

const SCORE_NAMES = {
  distinct_clips_top20: "Different clips in its top 20",
  label_holds_heldout: "Name holds on unseen clips",
  ablation_effect: "Prediction shift when removed",
  random_percentile: "Beats random features (percentile)",
  absent_in_random_init: "Absent from an untrained model",
};

function scoreItem(name, text, raw) {
  const shown = raw === null || raw === undefined ? "null" : String(raw);
  return el("div", {}, [
    el("dt", { title: name }, SCORE_NAMES[name] || name),
    el("dd", { "data-testid": name, "data-value": shown }, text),
  ]);
}

function scoreVersus(feature) {
  const percentile = feature.random_percentile === null || feature.random_percentile === undefined
    ? "—"
    : fmtPercentile(feature.random_percentile);
  const randomCount = feature.n_random === null || feature.n_random === undefined ? "—" : String(feature.n_random);
  return el("div", {}, [
    el("dt", { title: "random_percentile" }, SCORE_NAMES.random_percentile),
    el("dd", {}, [
      el("span", { "data-testid": "random_percentile", "data-value": feature.random_percentile === null || feature.random_percentile === undefined ? "null" : String(feature.random_percentile) }, percentile),
      " ",
      el("span", { class: "of" }, "of "),
      el("span", { "data-testid": "n_random", "data-value": feature.n_random === null || feature.n_random === undefined ? "null" : String(feature.n_random) }, randomCount),
    ]),
  ]);
}

async function showClip(clipId, frame, featureParam) {
  const token = ++state.token;
  const same = Boolean(state.clip && state.clipId === clipId);
  if (!same) {
    const response = await fetch(apiUrl(`/api/clips/${encodeURIComponent(clipId)}`), { cache: "no-store" });
    if (token !== state.token) return;
    if (!response.ok) {
      renderError(`Clip ${clipId} is not in this run.`);
      return;
    }
    state.clip = await response.json();
    state.clipId = clipId;
  }
  if (token !== state.token) return;
  state.featureId = featureParam === null || featureParam === "" ? null : Number(featureParam);
  state.frame = clampFrame(frame);
  if (same && document.querySelector("[data-testid=stage]")) {
    renderFrame();
    return;
  }
  renderClip();
}

function clampFrame(frame) {
  const count = state.clip.n_frames;
  if (!Number.isFinite(frame) || count <= 0) return 0;
  return Math.min(Math.max(0, frame), count - 1);
}

function renderClip() {
  const clip = state.clip;
  const links = state.run.clips.map((entry) => {
    const anchor = el(
      "a",
      {
        href: `#/clips/${encodeURIComponent(entry.clip_id)}`,
        "data-testid": "clip-link",
        "data-clip-id": entry.clip_id,
      },
      [entry.clip_id, el("span", { class: "split-tag" }, entry.split || "")],
    );
    if (entry.clip_id === clip.clip_id) anchor.setAttribute("aria-current", "page");
    return el("li", {}, anchor);
  });
  const image = el("img", { alt: "", "data-testid": "frame-image" });
  const video = el("video", {
    "data-testid": "clip-video",
    playsinline: "",
    preload: "auto",
    muted: "",
  });
  video.hidden = true;
  const heatmap = el("div", { class: "heatmap", "data-testid": "heatmap" });
  const shade = el("div", { class: "crop-shade", "data-testid": "crop-shade", hidden: "" });
  const stage = el("div", { class: "stage", "data-testid": "stage" }, [image, video, shade, heatmap]);
  video.addEventListener("loadedmetadata", () => {
    const hw = state.clip?.hw;
    if (video.videoWidth && video.videoHeight && !(hw && hw[0] > 0 && hw[1] > 0)) {
      stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      stage.parentElement?.style.setProperty("--stage-aspect", String(video.videoWidth / video.videoHeight));
    }
    if (state.clip?.media === "video" && !state.playing) renderFrame({ seek: true });
  });
  video.addEventListener("timeupdate", () => {
    if (!state.playing || state.clip?.media !== "video") return;
    const fps = state.clip.fps > 0 ? state.clip.fps : 8;
    const frame = Math.floor(video.currentTime * fps + 1e-3);
    if (frame !== state.frame) setFrame(frame, { seek: false });
  });
  video.addEventListener("ended", stop);
  const message = el("p", { class: "pixels-message", "data-testid": "pixels-message" });
  stage.append(message);
  const play = el("button", { type: "button", id: "play", "data-testid": "play", "data-state": "paused" }, "Play");
  const scrubber = el("input", {
    id: "scrubber",
    "data-testid": "scrubber",
    type: "range",
    min: "0",
    max: String(Math.max(clip.n_frames - 1, 0)),
    step: "1",
    value: String(state.frame),
    "aria-label": "Frame",
  });
  const scrubberWrap = el("div", { class: "scrubber-wrap", "data-testid": "scrubber-wrap" }, scrubber);
  play.addEventListener("click", togglePlay);
  scrubber.addEventListener("input", () => {
    stop();
    setFrame(Number(scrubber.value));
  });
  app.replaceChildren(
    el("div", { class: "clip-page" }, [
      el("aside", {}, [
        el("h2", { class: "panel-title" }, "Clips"),
        el("ul", { class: "clip-list" }, links),
      ]),
      el("section", { class: "player" }, [
        el("div", { class: "clip-kicker" }, [
          el("h2", { "data-testid": "clip-title" }, clip.clip_id),
          el("span", { class: "split-tag" }, clip.split === "heldout" ? "held-out clip" : clip.split ? `${clip.split} clip` : "clip"),
          el("p", { class: "clip-task", "data-testid": "clip-task", hidden: "" }),
        ]),
        el("div", { class: "stage-frame" }, stage),
        el("div", { class: "transport" }, [
          play,
          scrubberWrap,
          el("p", { class: "frame-readout", "data-testid": "frame-readout" }),
        ]),
        el("p", { class: "heat-caption", "data-testid": "heatmap-label" }),
        steerPanel(),
      ]),
      el("section", { class: "notices" }, [
        el("h2", {}, "What the model notices"),
        el("p", { class: "lead" }, "Features firing on this frame, strongest first. Pick one to see where it fires."),
        el("ol", { class: "feature-list", "data-testid": "frame-features" }),
      ]),
    ]),
  );
  renderFrame();
  document.querySelector(".clip-list a[aria-current='page']")?.scrollIntoView({ block: "nearest" });
}

function clipMoments(clip) {
  const id = String(clip.clip_id);
  return (state.run.moments || []).filter((moment) => String(moment.clip_id) === id);
}

function showTask(clip, frame) {
  const node = document.querySelector("[data-testid=clip-task]");
  if (!node) return;
  const moment = clipMoments(clip).find((item) => frame >= item.frame_start && frame <= item.frame_end && item.task_text);
  if (!moment) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.hidden = false;
  node.textContent = moment.task_text;
}

function markMoments(clip) {
  const wrap = document.querySelector("[data-testid=scrubber-wrap]");
  if (!wrap) return;
  wrap.querySelectorAll("[data-testid=moment-range]").forEach((node) => node.remove());
  const span = Math.max(clip.n_frames - 1, 1);
  for (const moment of clipMoments(clip)) {
    const start = Math.max(0, Math.min(span, moment.frame_start));
    const end = Math.max(start, Math.min(span, moment.frame_end));
    const mark = el("span", {
      class: "moment-range",
      "data-testid": "moment-range",
      "data-start": moment.frame_start,
      "data-end": moment.frame_end,
    });
    mark.style.left = `${(start / span) * 100}%`;
    mark.style.width = `${Math.max((end - start) / span, 1 / span) * 100}%`;
    wrap.append(mark);
  }
}

function setFrame(frame, options) {
  state.frame = clampFrame(frame);
  renderFrame(options);
  const feature = state.featureId === null || Number.isNaN(state.featureId) ? "" : `&feature=${state.featureId}`;
  const hash = `#/clips/${encodeURIComponent(state.clipId)}?frame=${state.frame}${feature}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
}

function renderFrame(options) {
  const clip = state.clip;
  const frame = clip.frames[state.frame];
  const stage = document.querySelector("[data-testid=stage]");
  const image = document.querySelector("[data-testid=frame-image]");
  const video = document.querySelector("[data-testid=clip-video]");
  const message = document.querySelector("[data-testid=pixels-message]");
  const readout = document.querySelector("[data-testid=frame-readout]");
  const scrubber = document.getElementById("scrubber");
  const list = document.querySelector("[data-testid=frame-features]");
  if (!frame || !list) return;

  const seek = !options || options.seek !== false;
  const tubelet = frame.t === null || frame.t === undefined ? "—" : String(frame.t);
  scrubber.value = String(state.frame);
  readout.textContent = `frame ${state.frame} · t ${tubelet} · window ${frame.window}`;
  readout.dataset.frame = String(state.frame);
  readout.dataset.t = tubelet;
  readout.dataset.window = String(frame.window);
  list.dataset.frame = String(state.frame);
  image.alt = `${clip.clip_id} frame ${state.frame}`;
  markMoments(clip);
  showTask(clip, state.frame);

  const useVideo = clip.media === "video" && clip.pixels === "ok";
  stage.dataset.media = useVideo ? "video" : clip.pixels === "ok" ? "frames" : "none";
  let aspect = null;
  if (clip.hw && clip.hw[0] > 0 && clip.hw[1] > 0) aspect = [clip.hw[1], clip.hw[0]];
  else if (clip.width && clip.height) aspect = [clip.width, clip.height];
  if (aspect) {
    stage.style.aspectRatio = `${aspect[0]} / ${aspect[1]}`;
    stage.parentElement?.style.setProperty("--stage-aspect", String(aspect[0] / aspect[1]));
  }
  placeCrop(clip);
  if (useVideo) {
    stage.classList.remove("is-empty");
    message.hidden = true;
    image.hidden = true;
    image.removeAttribute("src");
    video.hidden = false;
    const src = apiUrl(`/api/clips/${encodeURIComponent(clip.clip_id)}/media`);
    if (video.getAttribute("src") !== src) video.src = src;
    if (seek && !state.playing) {
      const fps = clip.fps > 0 ? clip.fps : 8;
      const target = state.frame / fps;
      if (!Number.isFinite(video.currentTime) || Math.abs(video.currentTime - target) > 0.45 / fps) {
        try {
          video.currentTime = target;
        } catch (_error) {
          // metadata may not be ready yet; loadedmetadata retries the seek
        }
      }
    }
  } else if (clip.pixels === "ok") {
    stage.classList.remove("is-empty");
    message.hidden = true;
    if (video) {
      video.hidden = true;
      video.removeAttribute("src");
    }
    const src = apiUrl(`/api/clips/${encodeURIComponent(clip.clip_id)}/frames/${state.frame}.png`);
    if (image.getAttribute("src") !== src) image.src = src;
    image.hidden = false;
  } else {
    stage.classList.add("is-empty");
    image.hidden = true;
    image.removeAttribute("src");
    if (video) video.hidden = true;
    message.hidden = false;
    message.textContent = clip.message || "Clip pixels are not available.";
  }

  const requested = Number.isInteger(state.featureId) ? state.featureId : null;
  const listed = requested === null ? null : frame.features.find((feature) => Number(feature.id) === requested);
  let active = listed;
  if (requested !== null && !listed) {
    active = { id: requested, label: "", badge: "unverified", activation: 0, effects: [], inactive: true };
  } else if (requested === null) {
    active = frame.features[0] || null;
  }
  list.dataset.feature = active ? String(active.id) : "";
  const peak = frame.features.reduce((best, feature) => Math.max(best, Number(feature.activation) || 0), 0);
  list.replaceChildren(...(frame.features.length ? frame.features.map((feature) => renderFeatureRow(feature, active, peak)) : [
    el("li", { class: "empty" }, "No feature fired on this frame."),
  ]));
  state.activeFeature = active;
  paintSteer();
  if (steerShowing()) {
    paintSteerCaption();
    drawSteer();
  } else {
    paintCaption(active);
    renderHeat(active);
  }
  document.title = `${clip.clip_id} · frame ${state.frame} · interpretableintelligence`;
}

function paintCaption(feature) {
  const caption = document.querySelector("[data-testid=heatmap-label]");
  if (!caption) return;
  caption.replaceChildren();
  caption.dataset.mode = "activation";
  if (!feature) {
    caption.textContent = "Nothing fired on this frame.";
    return;
  }
  if (feature.inactive) {
    caption.textContent = `feature ${feature.id} not active at this frame`;
    return;
  }
  caption.append("Lit up where ", el("strong", {}, displayLabel(feature)), " fires");
}

function renderFeatureRow(feature, active, peak) {
  const selected = active && feature.id === active.id;
  const row = el("li", {
    class: "frame-feature",
    "data-testid": "frame-feature",
    "data-feature-id": feature.id,
    "data-label": feature.label || "",
    "data-badge": feature.badge,
    "data-selected": selected ? "true" : "false",
  });
  const link = el("a", {
    class: "feature-link",
    href: `#/features/${feature.id}`,
    "data-testid": "feature-link",
    "data-feature-id": feature.id,
  }, displayLabel(feature));
  link.addEventListener("click", (event) => event.stopPropagation());
  row.append(
    el("div", { class: "feature-head" }, [link, badge(feature)]),
    el("div", { class: "activation" }, [
      el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width: ${barWidth(feature.activation, peak)}%` })),
      el("b", { "data-testid": "activation" }, fmtActivation(feature.activation)),
    ]),
  );
  if (feature.effects && feature.effects.length) {
    row.append(el("div", { class: "effects" }, feature.effects.map(renderEffect)));
  }
  row.addEventListener("click", () => {
    state.featureId = feature.id;
    renderFrame();
    const hash = `#/clips/${encodeURIComponent(state.clipId)}?frame=${state.frame}&feature=${feature.id}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
  });
  return row;
}

function barWidth(value, peak) {
  if (!Number.isFinite(value) || !(peak > 0) || value <= 0) return 0;
  return Math.max(4, Math.min(100, (value / peak) * 100)).toFixed(1);
}

function renderEffect(effect) {
  const kind = effect.effect_kind === "ablation" || effect.effect_kind === "estimate" ? effect.effect_kind : "other";
  const className = kind === "other" ? "effect effect-other" : `effect effect-${kind}`;
  return el("span", {
    class: className,
    "data-testid": "effect",
    "data-kind": effect.effect_kind,
    "data-effect": String(effect.effect),
  }, [
    el("span", { class: "kind" }, effect.effect_kind),
    el("span", { class: "value" }, fmtEffect(effect.effect)),
  ]);
}

function placeCrop(clip) {
  const heat = document.querySelector("[data-testid=heatmap]");
  const shade = document.querySelector("[data-testid=crop-shade]");
  if (!heat) return;
  const hw = clip.hw;
  const crop = clip.crop;
  const known = hw && hw[0] > 0 && hw[1] > 0 && crop && crop.width > 0 && crop.height > 0;
  if (!known) {
    heat.style.left = "0";
    heat.style.top = "0";
    heat.style.width = "100%";
    heat.style.height = "100%";
    delete heat.dataset.cropX;
    delete heat.dataset.cropY;
    delete heat.dataset.cropWidth;
    delete heat.dataset.cropHeight;
    if (shade) shade.hidden = true;
    return;
  }
  const left = (crop.x / hw[1]) * 100;
  const top = (crop.y / hw[0]) * 100;
  const width = (crop.width / hw[1]) * 100;
  const height = (crop.height / hw[0]) * 100;
  for (const node of [heat, shade].filter(Boolean)) {
    node.style.left = `${left}%`;
    node.style.top = `${top}%`;
    node.style.width = `${width}%`;
    node.style.height = `${height}%`;
  }
  heat.dataset.cropX = String(crop.x);
  heat.dataset.cropY = String(crop.y);
  heat.dataset.cropWidth = String(crop.width);
  heat.dataset.cropHeight = String(crop.height);
  const covers = crop.x === 0 && crop.y === 0 && crop.width === hw[1] && crop.height === hw[0];
  if (shade) shade.hidden = covers;
}

async function heatPack(clipId) {
  if (state.heatClip === clipId && state.heatPack) return state.heatPack;
  const response = await fetch(`${iiRoot()}/clips/${encodeURIComponent(clipId)}/heat.json`, { cache: "no-store" });
  const pack = response.ok ? await response.json() : {};
  state.heatClip = clipId;
  state.heatPack = pack;
  return pack;
}

async function renderHeat(feature) {
  const grid = document.querySelector("[data-testid=heatmap]");
  if (!grid) return;
  const token = ++state.heatToken;
  const frame = state.frame;
  grid.replaceChildren();
  delete grid.dataset.steer;
  if (feature) grid.dataset.featureId = String(feature.id);
  else delete grid.dataset.featureId;
  if (!feature) return;
  if (feature.inactive) drawHeat(emptyHeat(state.clip));
  if (iiRoot()) {
    const pack = await heatPack(state.clipId);
    if (token !== state.heatToken) return;
    const rows = (pack && pack[`${frame}:${feature.id}`]) || emptyHeat(state.clip);
    drawHeat(rows);
    const hot = rows.some((row) => row.some((value) => value > 0));
    if (feature.inactive && hot) paintCaption({ ...feature, inactive: false });
    return;
  }
  const response = await fetch(
    `/api/clips/${encodeURIComponent(state.clipId)}/heatmap?frame=${frame}&feature=${feature.id}`,
    { cache: "no-store" },
  );
  if (token !== state.heatToken || !response.ok) return;
  const payload = await response.json();
  if (token !== state.heatToken) return;
  const rows = payload.heatmap || [];
  drawHeat(rows);
  const hot = rows.some((row) => row.some((value) => value > 0));
  if (feature.inactive && hot) paintCaption({ ...feature, inactive: false });
}

function emptyHeat(clip) {
  const height = clip?.grid?.h || 0;
  const width = clip?.grid?.w || 0;
  if (height <= 0 || width <= 0) return [];
  return Array.from({ length: height }, () => Array.from({ length: width }, () => 0));
}

function drawHeat(rows, rgb = themeRgb("--signal-rgb", "242, 84, 27")) {
  const grid = document.querySelector("[data-testid=heatmap]");
  if (!grid || !rows.length) return;
  grid.replaceChildren();
  const height = rows.length;
  const width = rows[0].length;
  let max = 0;
  for (const row of rows) for (const value of row) if (value > max) max = value;
  grid.dataset.dense = width * height > 400 ? "true" : "false";
  grid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  grid.style.gridTemplateRows = `repeat(${height}, 1fr)`;
  for (let h = 0; h < height; h += 1) {
    for (let w = 0; w < width; w += 1) {
      const value = rows[h][w];
      const alpha = max > 0 && value > 0 ? Math.pow(value / max, 0.6) * 0.78 : 0;
      const cell = el("i", {
        class: "heat-cell",
        "data-testid": "heat-cell",
        "data-h": h,
        "data-w": w,
        "data-value": String(value),
      });
      cell.style.background = alpha > 0 ? `rgba(${rgb}, ${alpha.toFixed(3)})` : "transparent";
      grid.append(cell);
    }
  }
}

function togglePlay() {
  const video = document.querySelector("[data-testid=clip-video]");
  const useVideo = state.clip?.media === "video" && state.clip.pixels === "ok" && video && !video.hidden;
  state.playing = !state.playing;
  const button = document.getElementById("play");
  button.textContent = state.playing ? "Pause" : "Play";
  button.dataset.state = state.playing ? "playing" : "paused";
  if (useVideo) {
    if (state.playing) {
      const play = video.play();
      if (play && typeof play.catch === "function") play.catch(() => stop());
    } else {
      video.pause();
    }
    return;
  }
  if (state.playing) tick();
  else if (state.timer) {
    clearTimeout(state.timer);
    state.timer = null;
  }
}

function tick() {
  if (!state.playing || !state.clip) return;
  setFrame((state.frame + 1) % state.clip.n_frames);
  const fps = state.clip.fps > 0 ? state.clip.fps : 8;
  state.timer = setTimeout(tick, 1000 / fps);
}

function stop() {
  state.playing = false;
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  const video = document.querySelector("[data-testid=clip-video]");
  if (video && !video.paused) video.pause();
  const button = document.getElementById("play");
  if (button) {
    button.textContent = "Play";
    button.dataset.state = "paused";
  }
}

boot();

// Playground: steer a feature and see where the model's prediction moves. Exploration only.

function steerEnabled() {
  return Boolean(state.run && state.run.steer && state.run.steer.enabled && !iiRoot());
}

function steerFrame() {
  return state.clip ? state.clip.frames[state.frame] : null;
}

function steerResultHere() {
  const result = state.steer.result;
  const frame = steerFrame();
  if (!result || !frame || result.clip_id !== state.clipId) return null;
  if (Number(result.window) !== Number(frame.window)) return null;
  return result;
}

function steerShowing() {
  return steerEnabled() && state.steer.show && Boolean(steerResultHere());
}

function steerPanel() {
  if (!steerEnabled()) return el("div", { hidden: "" });
  const kind = el("select", { "data-testid": "steer-kind", "aria-label": "Edit" },
    Object.entries(STEER_KINDS).map(([key, spec]) => el("option", { value: key }, spec.label)));
  kind.value = state.steer.kind;
  const value = el("input", {
    type: "range",
    "data-testid": "steer-value",
    "aria-label": "Strength",
  });
  const readout = el("output", { class: "steer-readout", "data-testid": "steer-readout" });
  const run = el("button", { type: "button", class: "steer-run", "data-testid": "steer-run" }, "Steer");
  const show = el("input", { type: "checkbox", "data-testid": "steer-show" });
  show.checked = state.steer.show;
  kind.addEventListener("change", () => {
    state.steer.kind = kind.value;
    state.steer.value = STEER_KINDS[kind.value].value;
    paintSteer();
  });
  value.addEventListener("input", () => {
    state.steer.value = Number(value.value);
    paintSteer();
  });
  run.addEventListener("click", runSteer);
  show.addEventListener("change", () => {
    state.steer.show = show.checked;
    renderFrame({ seek: false });
  });
  const usable = state.run.steer.sae_usable;
  return el("div", { class: "steer", "data-testid": "steer-panel" }, [
    el("h3", { class: "panel-title" }, "Playground"),
    el("p", { class: "steer-target", "data-testid": "steer-target" }),
    el("div", { class: "steer-controls" }, [kind, value, readout, run]),
    el("p", { class: "steer-help", "data-testid": "steer-help" }),
    el("label", { class: "steer-toggle" }, [show, " Show where the prediction moved"]),
    el("div", { class: "steer-result", "data-testid": "steer-result" }),
    usable ? null : el("p", { class: "steer-warn", "data-testid": "steer-warn" },
      "This run's SAE is not marked usable, so its features may not mean much yet. Steering still works."),
    el("p", { class: "steer-note" }, "Exploration only. Steering never marks a feature real; only the scored gates do."),
  ].filter(Boolean));
}

function paintSteer() {
  const panel = document.querySelector("[data-testid=steer-panel]");
  if (!panel) return;
  const spec = STEER_KINDS[state.steer.kind];
  const value = panel.querySelector("[data-testid=steer-value]");
  value.min = String(spec.min);
  value.max = String(spec.max);
  value.step = String(spec.step);
  value.value = String(state.steer.value);
  value.disabled = state.steer.kind === "ablate";
  panel.querySelector("[data-testid=steer-readout]").textContent = state.steer.kind === "ablate" ? "" : `${state.steer.value}`;
  panel.querySelector("[data-testid=steer-help]").textContent = spec.help;
  const target = panel.querySelector("[data-testid=steer-target]");
  const feature = state.activeFeature;
  target.replaceChildren(feature ? "Feature: " : "Pick a feature from the list to steer it.");
  if (feature) target.append(el("strong", {}, displayLabel(feature)));
  const run = panel.querySelector("[data-testid=steer-run]");
  run.disabled = !feature || state.steer.busy;
  run.textContent = state.steer.busy ? "Steering…" : "Steer";
  const box = panel.querySelector("[data-testid=steer-result]");
  box.replaceChildren();
  if (state.steer.error) {
    box.append(el("p", { class: "steer-error", "data-testid": "steer-error" }, state.steer.error));
    return;
  }
  const result = state.steer.result;
  if (!result || result.clip_id !== state.clipId) return;
  const sum = result.summary;
  const kindLabel = STEER_KINDS[result.kind]?.label || result.kind;
  const amount = result.kind === "ablate" ? "" : ` ${result.value}`;
  box.dataset.meanShift = String(sum.mean_shift);
  box.append(
    el("p", {}, [
      `${kindLabel}${amount} on feature ${result.features.join(", ")}, window ${result.window}: `,
      "the prediction moved ",
      el("strong", { "data-testid": "steer-mean" }, fmtSmall(sum.mean_shift)),
      " on average (cosine distance), up to ",
      el("strong", {}, fmtSmall(sum.max_shift)),
      ".",
    ]),
    el("p", {}, [
      "Prediction error went from ",
      el("strong", {}, fmtSmall(sum.loss_clean)),
      " to ",
      el("strong", {}, fmtSmall(sum.loss_edited)),
      `. The feature fired on ${sum.tokens_where_feature_fired} of ${sum.tokens_total} patches in this window.`,
    ]),
  );
  if (!steerResultHere()) box.append(el("p", { class: "steer-warn" }, "This frame is in another window. Steer again to see it here."));
}

function paintSteerCaption() {
  const caption = document.querySelector("[data-testid=heatmap-label]");
  if (!caption) return;
  const names = state.steer.result.features.map((id) => {
    const known = (state.run.features || []).find((feature) => Number(feature.id) === Number(id));
    return known ? displayLabel(known) : `feature ${id}`;
  });
  caption.dataset.mode = "steer";
  caption.replaceChildren("Showing the prediction change from steering ", el("strong", {}, names.join(", ")));
}

function drawSteer() {
  const result = steerResultHere();
  const frame = steerFrame();
  const grid = document.querySelector("[data-testid=heatmap]");
  if (!result || !frame || !grid) return;
  state.heatToken += 1;
  grid.replaceChildren();
  grid.dataset.steer = "true";
  const t = Number(frame.t);
  const rows = (result.shift[t] || []).map((row) => row.map((value) => (value === null ? 0 : value)));
  drawHeat(rows, themeRgb("--steer-rgb", "47, 85, 228"));
}

async function runSteer() {
  const feature = state.activeFeature;
  const frame = steerFrame();
  if (!feature || !frame || state.steer.busy) return;
  const token = ++state.steer.token;
  state.steer.busy = true;
  state.steer.error = "";
  paintSteer();
  try {
    const response = await fetch("/api/steer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip_id: state.clipId,
        window: Number(frame.window) || 0,
        features: [Number(feature.id)],
        kind: state.steer.kind,
        value: state.steer.value,
      }),
    });
    const payload = await response.json();
    if (token !== state.steer.token) return;
    if (!response.ok) {
      state.steer.error = (payload.detail && payload.detail.error) || payload.error || "Steering failed.";
      state.steer.result = null;
    } else {
      state.steer.result = payload;
      state.steer.show = true;
      const toggle = document.querySelector("[data-testid=steer-show]");
      if (toggle) toggle.checked = true;
    }
  } catch (error) {
    state.steer.error = `Steering failed: ${error}`;
  } finally {
    if (token === state.steer.token) state.steer.busy = false;
    renderFrame({ seek: false });
  }
}
