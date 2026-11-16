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
  steer: { kind: "ablate", value: 0, result: null, busy: false, show: false, error: "", token: 0, idea: null },
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

// The "Numbers" switch in the header: the plain-language view by default, raw
// measurements (class "num") on request. Remembered per browser, never required.
const NUMBERS_KEY = "ii.numbers";

function initNumbers() {
  let on = false;
  try {
    on = window.localStorage.getItem(NUMBERS_KEY) === "on";
  } catch (_error) {
    on = false;
  }
  setNumbers(on, false);
  const toggle = document.querySelector("[data-testid=numbers-toggle]");
  if (toggle) toggle.addEventListener("click", () => setNumbers(!numbersOn(), true));
}

function numbersOn() {
  return document.documentElement.dataset.numbers === "on";
}

function setNumbers(on, remember) {
  document.documentElement.dataset.numbers = on ? "on" : "off";
  const toggle = document.querySelector("[data-testid=numbers-toggle]");
  if (toggle) toggle.setAttribute("aria-pressed", on ? "true" : "false");
  if (!remember) return;
  try {
    window.localStorage.setItem(NUMBERS_KEY, on ? "on" : "off");
  } catch (_error) {
    // storage can be blocked; the switch still works for this page
  }
}

async function boot() {
  initNumbers();
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
  watchLive();
}

// A run still filling in (ii run --remote) carries `live` on /api/run. Show where it runs and
// what it has cost so far, and redraw when a newer stage lands (`live.generation` goes up).
const LIVE_POLL_MS = 3000;

function liveText(live) {
  const end = typeof live.finished_at === "number" ? live.finished_at : Date.now() / 1000;
  const cost = (Math.max(0, end - live.started_at) * Number(live.usd_per_hour || 0)) / 3600;
  const money = (value) => (value < 1 ? `$${value.toFixed(3)}` : `$${value.toFixed(2)}`);
  const where = [live.provider, live.gpu].filter(Boolean).join(" ");
  let what = live.state;
  if (live.state === "running") what = live.stage ? `${live.stage} ready, running` : "running";
  const cap = typeof live.max_cost_usd === "number" ? ` of ${money(live.max_cost_usd)}` : "";
  return `${where} · ${what} · ~${money(cost)}${cap}`;
}

function watchLive() {
  const node = document.getElementById("live-status");
  if (!node || !state.run || !state.run.live) return;
  let generation = state.run.live.generation;
  const paint = () => {
    const live = state.run.live;
    node.hidden = !live;
    if (!live) return;
    node.dataset.state = live.state;
    node.textContent = liveText(live);
    node.title = "Estimated from the container's list price. The provider's usage page has the exact bill.";
  };
  paint();
  const tick = setInterval(paint, 1000);
  const poll = setInterval(async () => {
    let fresh;
    try {
      const response = await fetch(apiUrl("/api/run"), { cache: "no-store" });
      if (!response.ok) return;
      fresh = await response.json();
    } catch (error) {
      return;
    }
    if (fresh.live && fresh.live.generation !== generation) {
      generation = fresh.live.generation;
      state.run = fresh;
      renderHeader();
      route();
    } else {
      state.run.live = fresh.live;
    }
    paint();
    if (!fresh.live || fresh.live.state !== "running") {
      clearInterval(poll);
      clearInterval(tick);
    }
  }, LIVE_POLL_MS);
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
  const explain = guide() ? guide().sae(run.scoreboard === false ? null : run.sae).result : "";
  if (explain) saeEl.title = explain;
  const status = document.getElementById("run-status");
  if (status) {
    const scored = run.scoreboard !== false && Boolean(run.sae);
    const usable = scored && run.sae.usable === true;
    status.dataset.state = !scored ? "none" : usable ? "ok" : "warn";
    status.title = explain || "";
    status.replaceChildren(
      el("span", { class: "run-dot", "aria-hidden": "true" }),
      `${name} · ${!scored ? "not scored yet" : usable ? "dictionary usable" : "dictionary not usable"}`,
    );
  }
  const steeringLink = document.querySelector('.nav a[data-nav="steering"]');
  if (steeringLink) steeringLink.hidden = !steeringCard();
  document.title = `${run.run_id} · interpretableintelligence`;
}

function route() {
  stop();
  const raw = (location.hash || "#/").replace(/^#/, "");
  const [path, query = ""] = raw.split("?");
  const params = new URLSearchParams(query);
  markNav(path.startsWith("/features") ? "features" : path === "/steering" ? "steering" : "explore");
  if (path === "/steering") {
    renderSteering();
    return;
  }
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
    showClip(decodeURIComponent(clip[1]), Number(params.get("frame") || 0), params.get("feature"), params.get("try"));
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
  const verified = features.filter((feature) => feature.badge === "real");
  const exploring = features.filter((feature) => feature.badge !== "real");
  const real = verified.length;
  const moments = renderMoments();
  const passed = real === 0
    ? "None has passed every check yet, and all of them are open to explore."
    : `${real === 1 ? "One has" : `${real} have`} passed every check. Open any feature to see where it fires and steer it.`;
  const page = [
    moments,
    el("section", { class: "feature-index", "data-testid": "feature-index" }, [
      el("h1", {}, "Features"),
      el("p", { class: "lede" }, [el("b", {}, "Patterns the model learned to look for."), ` ${passed}`]),
      el("p", { class: "kicker num", "data-testid": "real-count" }, `${real} real · ${exploring.length} unverified`),
      indexGuide(features),
      featureGroup("verified", "Verified", "Passed all five checks. These are the ones to trust.", verified),
      featureGroup(
        "exploring",
        "Still exploring",
        "Each missed at least one check. Open one to see which, and what experiment to try next.",
        exploring,
      ),
    ]),
  ].filter(Boolean);
  app.replaceChildren(...page);
  document.title = `features · ${state.run.run_id} · interpretableintelligence`;
}

function featureGroup(key, title, note, features) {
  if (!features.length) return null;
  const list = el("ul", { class: "index-list", "data-group": key });
  for (const feature of features) list.append(el("li", {}, featureCard(feature)));
  return el("section", { class: "index-group", "data-testid": `index-group-${key}` }, [
    el("h2", { class: "section-title" }, [title, " ", el("span", { class: "count" }, String(features.length))]),
    el("p", { class: "group-note" }, note),
    list,
  ]);
}

// One card per feature: the frame where it fires hardest beside its activation blocks,
// loaded only when the card scrolls into view.
function featureCard(feature) {
  const art = el("span", { class: "card-art", "aria-hidden": "true", "data-feature-id": feature.id });
  const link = el("a", {
    class: "feature-card",
    href: `#/features/${feature.id}`,
    "data-testid": "feature-index-link",
    "data-feature-id": feature.id,
    "data-label": feature.label || "",
    "data-candidate": feature.candidate ? "true" : "false",
  }, [
    art,
    el("span", { class: "card-body" }, [
      el("span", { class: "card-name" }, displayLabel(feature)),
      el("span", { class: "card-meta" }, [el("span", { class: "index-id" }, `#${feature.id}`), badge(feature)]),
    ]),
  ]);
  observeArt(art);
  return link;
}

let artObserver = null;

function observeArt(node) {
  if (typeof IntersectionObserver === "undefined") {
    loadArt(node);
    return;
  }
  if (!artObserver) {
    artObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        artObserver.unobserve(entry.target);
        loadArt(entry.target);
      }
    }, { rootMargin: "240px" });
  }
  artObserver.observe(node);
}

async function loadArt(node) {
  const id = Number(node.dataset.featureId);
  let feature = null;
  try {
    const response = await fetch(apiUrl(`/api/features/${id}`), { cache: "no-store" });
    feature = response.ok ? await response.json() : null;
  } catch (_error) {
    feature = null;
  }
  const top = feature && feature.top_clips && feature.top_clips[0];
  if (!top || !node.isConnected) {
    node.classList.add("is-empty");
    node.textContent = feature ? "Silent on every clip" : "";
    return;
  }
  const image = el("img", {
    alt: "",
    loading: "lazy",
    src: apiUrl(`/api/clips/${encodeURIComponent(top.clip_id)}/frames/${top.frame}.png`),
  });
  image.addEventListener("error", () => image.remove());
  const terrain = el("span", { class: "card-terrain" });
  node.replaceChildren(el("span", { class: "card-shot" }, image), terrain);
  const rows = await fetchHeat(top.clip_id, top.frame, id).catch(() => null);
  if (rows && rows.length && node.isConnected) terrain.innerHTML = isoTerrain(rows);
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
  // Pictures for the strongest few; the table below lists every top clip.
  for (const clip of feature.top_clips.slice(0, 6)) {
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
      featureStanding(feature) || featureActions(feature),
      featureHero(feature),
      feature.viz
        ? el("img", { class: "viz", alt: "", src: apiUrl(`/api/features/${feature.id}/viz`), "data-testid": "feature-viz" })
        : null,
      el("section", { class: "page-section" }, [
        el("h2", { class: "section-title" }, "What the checks say"),
        el("p", { class: "section-note" }, "Five questions decide the verified badge. Each answer is in plain words; open one to see exactly what it asks."),
        feature.scored === false
          ? el("p", { class: "kicker", "data-testid": "scoreboard-missing" }, "No scoreboard record. Shown as unverified.")
          : null,
        featureGuide(feature),
      ]),
      el("section", { class: "page-section" }, [
        el("h2", { class: "section-title" }, "Where it fires hardest"),
        el("p", { class: "section-note" }, "The frames where this feature is strongest. Open one to watch it in the clip."),
        feature.top_clips.length ? strip : el("p", { class: "empty" }, "This feature does not fire on any clip."),
      ]),
      steeringSection(feature.id),
      el("section", { class: "page-section numbers-section" }, [
        el("h2", { class: "section-title" }, "The numbers"),
        el("p", { class: "section-note" }, "The raw measurements behind the checks, and every top clip."),
        el("dl", { class: "score", "data-testid": "scoreboard" }, [
          scoreItem("distinct_clips_top20", fmtMaybe(feature.distinct_clips_top20, String), feature.distinct_clips_top20),
          scoreItem("label_holds_heldout", fmtFlag(feature.label_holds_heldout), feature.label_holds_heldout),
          scoreItem("ablation_effect", fmtEffect(feature.ablation_effect), feature.ablation_effect),
          scoreVersus(feature),
          scoreItem("absent_in_random_init", fmtFlag(feature.absent_in_random_init), feature.absent_in_random_init),
        ]),
        el("div", { class: "table-wrap" }, el("table", { class: "top-table" }, [
          el("thead", {}, el("tr", {}, ["clip", "split", "frame", "window / t", "activation"].map((label) => el("th", {}, label)))),
          el("tbody", {}, rows),
        ])),
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

// The one obvious next step: watch the feature on the clip where it fires hardest.
function featureActions(feature) {
  const top = feature.top_clips && feature.top_clips[0];
  if (!top) return null;
  const href = `#/clips/${encodeURIComponent(top.clip_id)}?frame=${top.frame}&feature=${feature.id}`;
  return el("div", { class: "feature-actions" }, [
    el("a", { class: "button", href, "data-testid": "feature-watch" }, steerEnabled() ? "Watch and steer it" : "Watch it fire"),
    el("span", { class: "feature-actions-note" }, `Opens ${top.clip_id} at frame ${top.frame}, where it is strongest.`),
  ]);
}

function guideCtx(feature) {
  const run = state.run;
  return { steerHref: steerHrefFor(feature), sae: run.scoreboard === false ? null : run.sae };
}

// The top clip with a playground edit set up (and run) on arrival.
function tryHref(feature, idea) {
  const top = feature.top_clips && feature.top_clips[0];
  if (!top) return null;
  const base = `#/clips/${encodeURIComponent(top.clip_id)}?frame=${top.frame}&feature=${feature.id}`;
  return steerEnabled() && idea ? `${base}&try=${idea.kind}:${idea.value}` : base;
}

function ideaName(idea) {
  const label = STEER_KINDS[idea.kind]?.label || idea.kind;
  return idea.kind === "ablate" ? label : `${label} ${idea.value}${idea.kind === "scale" ? "×" : ""}`;
}

// The check ring: one arc per check, filled when it passed. All filled, it closes
// into the verified seal with a tick in the middle. Numbers only, so innerHTML is safe.
function checkRing(trail, real, size = 64) {
  const n = trail.marks.length;
  const r = 26;
  const gap = n > 1 ? 18 : 0;
  const arc = (start, end) => {
    const rad = (deg) => ((deg - 90) * Math.PI) / 180;
    const [x1, y1] = [32 + r * Math.cos(rad(start)), 32 + r * Math.sin(rad(start))];
    const [x2, y2] = [32 + r * Math.cos(rad(end)), 32 + r * Math.sin(rad(end))];
    return `M${x1.toFixed(2)} ${y1.toFixed(2)}A${r} ${r} 0 ${end - start > 180 ? 1 : 0} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
  };
  const step = 360 / n;
  const arcs = trail.marks.map((mark, i) =>
    `<path class="ring-arc ring-${mark.mark}" style="--i:${i}" pathLength="1" d="${arc(i * step + gap / 2, (i + 1) * step - gap / 2)}"/>`).join("");
  const middle = real
    ? `<circle class="ring-core" cx="32" cy="32" r="16"/><path class="ring-tick" pathLength="1" d="M24.5 32.5l5 5 10-11"/>`
    : `<text class="ring-count" x="32" y="32" text-anchor="middle" dominant-baseline="central">${trail.passed}/${trail.total}</text>`;
  const label = real ? "Verified: every check passed" : `${trail.passed} of ${trail.total} checks passed`;
  return `<svg class="check-ring" width="${size}" height="${size}" viewBox="0 0 64 64" role="img" aria-label="${label}">${arcs}${middle}</svg>`;
}

const COUNT_WORDS = ["no", "one", "two", "three", "four", "five", "six"];

function countWord(n) {
  return COUNT_WORDS[n] || String(n);
}

function standingLine(trail, real) {
  if (real) return `All ${countWord(trail.total)} checks passed`;
  const left = trail.total - trail.passed;
  if (trail.passed === 0) return "A fresh lead, with every check still ahead";
  const head = countWord(trail.passed);
  return `${head.charAt(0).toUpperCase()}${head.slice(1)} of ${countWord(trail.total)} checks passed. ${left === 1 ? "One to go." : `${countWord(left).replace(/^./, (c) => c.toUpperCase())} to go.`}`;
}

function checkTrail(trail) {
  const symbol = { pass: "✓", fail: "×", skip: "–" };
  const word = { pass: "passed", fail: "missed", skip: "not run" };
  return el("ol", { class: "check-trail", "data-testid": "check-trail" }, trail.marks.map((mark) =>
    el("li", { class: `trail-${mark.mark}`, "data-gate": mark.key, title: `${mark.name}: ${word[mark.mark]}` }, [
      el("span", { class: "trail-mark", "aria-hidden": "true" }, symbol[mark.mark]),
      mark.name,
    ])));
}

// Where the feature stands, right under its title. Verified: the seal, what the badge
// earns, and what to try now. Not yet: how far it got, what held it back, and the next
// experiment, one click from the playground. Both always end in something to do.
function featureStanding(feature) {
  const g = guide();
  if (!g || !g.trail) return null;
  const ctx = guideCtx(feature);
  const read = g.feature(feature, ctx);
  const trail = g.trail(feature, ctx);
  const plan = g.experiments(feature, ctx);
  const idea = plan.ideas[0];
  const real = read.real;
  const top = feature.top_clips && feature.top_clips[0];
  const ring = el("span", { class: "standing-ring" });
  ring.innerHTML = checkRing(trail, real);
  const head = el("div", { class: "standing-head" }, [
    ring,
    el("div", { class: "standing-sum" }, [
      el("p", { class: "standing-count", "data-testid": "standing-count" }, standingLine(trail, real)),
      checkTrail(trail),
    ]),
  ]);
  const play = top
    ? el("a", { class: "button", href: tryHref(feature, idea), "data-testid": "feature-watch" },
        steerEnabled() ? (real ? "Steer it in the playground" : "Try it in the playground") : "Watch it fire")
    : null;
  const playNote = top
    ? el("p", { class: "standing-note" }, steerEnabled()
        ? `Opens ${top.clip_id} at frame ${top.frame}, where it is strongest, with ${ideaName(idea)} set up.`
        : `Opens ${top.clip_id} at frame ${top.frame}, where it is strongest.`)
    : null;
  let left;
  let right;
  if (real) {
    const n = Number.isFinite(feature.n_random) && feature.n_random > 0 ? `${feature.n_random} random look-alikes` : "random look-alikes";
    const steered = steeringCard() && steeringCard().rows.some((row) => row.feature_id === feature.id);
    left = el("div", { class: "standing-col" }, [
      el("h3", { class: "standing-title" }, "What the badge earns it"),
      el("ul", { class: "standing-list" }, [
        el("li", {}, `Switching it off moved the prediction more than ${n} did, on clips the model never trained on.`),
        el("li", {}, richText("`ii explain` can now cite it when it explains a new clip.")),
        el("li", {}, steered
          ? [el("a", { href: "#/steering" }, "Its steering scorecard"), " is below. For a robot policy, a closed-loop rollout is the next bar."]
          : richText("Next bar: `ii steer-scorecard RUN` measures removing and boosting it against random edits. For a robot policy, a closed-loop rollout.")),
      ]),
    ]);
    right = el("div", { class: "standing-col standing-next", "data-testid": "standing-next" }, [
      el("h3", { class: "standing-title" }, steerEnabled() ? idea.title : "Watch it work"),
      el("p", {}, steerEnabled() ? idea.text : "Play its strongest clip and see where it lights up."),
      play,
      playNote,
    ]);
  } else {
    const gate = read.gates.find((item) => item.key === plan.gate) || null;
    const hint = !steerEnabled() && !iiRoot() && top
      ? el("p", { class: "standing-note" }, richText("Start the viewer with `ii view RUN --steer` to turn the playground on."))
      : null;
    left = el("div", { class: "standing-col" }, [
      el("h3", { class: "standing-title" }, gate ? (gate.passed === false ? "What held it back" : "What has not run yet") : "Where it stands"),
      el("p", {}, gate ? richText(gate.result) : "Every check it could take, it passed. The scorer has not marked it verified."),
    ]);
    right = el("div", { class: "standing-col standing-next", "data-testid": "standing-next" }, [
      el("h3", { class: "standing-title" }, "Your next experiment"),
      el("p", {}, gate && gate.next ? richText(gate.next.text) : idea.text),
      play,
      playNote,
      hint,
    ]);
  }
  return el("section", {
    class: "standing",
    "data-testid": "standing",
    "data-real": real ? "true" : "false",
    "data-next": plan.key,
  }, [head, el("div", { class: "standing-body" }, [left, right])]);
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
  const base = [222, 218, 205];
  const hot = [232, 88, 42];
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

async function showClip(clipId, frame, featureParam, tryParam) {
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
  const tried = steerEnabled() ? parseTry(tryParam) : null;
  if (tried) {
    state.steer.kind = tried.kind;
    state.steer.value = tried.value;
  }
  if (same && document.querySelector("[data-testid=stage]")) {
    renderFrame();
  } else {
    renderClip();
  }
  // Arriving from "Try it in the playground": run the edit that was set up.
  if (tried && token === state.token) runSteer();
}

// `try=scale:3` in a clip link: a playground edit to set up on arrival.
function parseTry(raw) {
  if (!raw) return null;
  const [kind, value] = String(raw).split(":");
  if (!STEER_KINDS[kind]) return null;
  const number = kind === "ablate" ? 0 : Number(value);
  if (!Number.isFinite(number)) return null;
  return { kind, value: nearestPreset(kind, number) };
}

// Static exports only have precomputed strengths: snap to the closest one.
function nearestPreset(kind, value) {
  const presets = steerPresets(kind);
  if (!presets) return value;
  return presets.reduce((best, item) => (Math.abs(item - value) < Math.abs(best - value) ? item : best), presets[0]);
}

function clampFrame(frame) {
  const count = state.clip.n_frames;
  if (!Number.isFinite(frame) || count <= 0) return 0;
  return Math.min(Math.max(0, frame), count - 1);
}

function renderClip() {
  const clip = state.clip;
  const links = state.run.clips.map((entry) => {
    const thumb = el("span", { class: "clip-thumb", "aria-hidden": "true" });
    if (entry.pixels === undefined || entry.pixels === "ok") {
      const image = el("img", {
        alt: "",
        loading: "lazy",
        src: apiUrl(`/api/clips/${encodeURIComponent(entry.clip_id)}/frames/0.png`),
      });
      image.addEventListener("error", () => image.remove());
      thumb.append(image);
    }
    const heldout = entry.split === "heldout";
    const anchor = el(
      "a",
      {
        href: `#/clips/${encodeURIComponent(entry.clip_id)}`,
        "data-testid": "clip-link",
        "data-clip-id": entry.clip_id,
        "data-split": entry.split || "",
      },
      [
        thumb,
        el("span", { class: "clip-name" }, entry.clip_id),
        heldout ? el("span", { class: "split-tag", title: "Held out: never used to train the dictionary" }, "unseen") : null,
      ],
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
  const splitNote = clip.split === "heldout"
    ? "Unseen clip: never used to train the dictionary"
    : clip.split === "train"
      ? "Training clip"
      : "Clip";
  app.replaceChildren(
    el("div", { class: "clip-page" }, [
      el("aside", { class: "rail" }, [
        el("h2", { class: "panel-title" }, ["Clips ", el("span", { class: "count" }, String(state.run.clips.length))]),
        el("ul", { class: "clip-list" }, links),
      ]),
      el("section", { class: "player" }, [
        el("div", { class: "clip-kicker" }, [
          el("h2", { "data-testid": "clip-title" }, clip.clip_id),
          el("span", { class: "split-tag" }, splitNote),
          el("p", { class: "clip-task", "data-testid": "clip-task", hidden: "" }),
        ]),
        el("div", { class: "stage-frame" }, stage),
        el("div", { class: "transport" }, [
          play,
          scrubberWrap,
          el("p", { class: "frame-line" }, [
            el("span", { class: "frame-count", "data-testid": "frame-count" }),
            el("span", { class: "frame-readout num", "data-testid": "frame-readout" }),
          ]),
        ]),
        el("p", { class: "heat-caption", "data-testid": "heatmap-label" }),
      ]),
      el("section", { class: "notices" }, [
        el("h2", {}, "What the model notices"),
        el("p", { class: "lead" }, [
          "Each feature is a pattern the model learned. ",
          el("span", { class: "lead-quiet" }, "These fire on this frame, strongest first. Pick one and the frame glows where it fires."),
        ]),
        el("ol", { class: "feature-list", "data-testid": "frame-features" }),
        steerPanel(),
      ]),
    ]),
  );
  renderFrame();
  document.querySelector(".clip-list a[aria-current='page']")?.scrollIntoView({ block: "nearest", inline: "nearest" });
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
  const count = document.querySelector("[data-testid=frame-count]");
  if (count) count.textContent = `Frame ${state.frame + 1} of ${clip.n_frames}`;
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
    // Start a newcomer on something verified when the frame has one.
    active = frame.features.find((feature) => feature.badge === "real") || frame.features[0] || null;
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

const EFFECT_HELP = {
  ablation: "Switch this feature off and measure how far the model's prediction moves.",
  estimate: "A cheap estimate of the same thing, from gradients. Not evidence on its own.",
};

function renderEffect(effect) {
  const kind = effect.effect_kind === "ablation" || effect.effect_kind === "estimate" ? effect.effect_kind : "other";
  // The ablation effect stays in view; estimates and other effects are raw numbers.
  const className = kind === "ablation" ? "effect effect-ablation" : `effect effect-${kind} num`;
  return el("span", {
    class: className,
    "data-testid": "effect",
    "data-kind": effect.effect_kind,
    "data-effect": String(effect.effect),
    title: EFFECT_HELP[kind] || "",
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
  // Action tokens (GR00T) are steps in time, not places in the frame: draw them as a strip.
  heat.dataset.layout = clip.layout === "actions" ? "actions" : "patches";
  heat.title = clip.layout === "actions" ? "Robot state, then each planned action step, now → later" : "";
  if (!known || clip.layout === "actions") {
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
  return Boolean(state.run && state.run.steer && state.run.steer.enabled && (!iiRoot() || steerStatic()));
}

// A static export (`ii export`) ships precomputed edits instead of a model.
function steerStatic() {
  return Boolean(iiRoot() && state.run && state.run.steer && state.run.steer.static);
}

function steerPresets(kind) {
  const presets = steerStatic() && state.run.steer.presets ? state.run.steer.presets[kind] : null;
  return Array.isArray(presets) && presets.length ? presets : null;
}

function steerKey(window, featureId, kind, value) {
  return `${Number(window)}:${Number(featureId)}:${kind}:${Number(value)}`;
}

const steerPacks = {};

async function staticSteer(clipId, window, featureId, kind, value) {
  if (!(state.run.steer.clips || []).includes(clipId)) {
    return { error: "This export precomputed the playground on other clips. Open a top clip from a feature page, or run ii view RUN --steer locally to steer any clip." };
  }
  if (!steerPacks[clipId]) {
    const response = await fetch(`${iiRoot()}/steer/${encodeURIComponent(clipId)}.json`, { cache: "no-store" });
    steerPacks[clipId] = response.ok ? await response.json() : {};
  }
  const result = steerPacks[clipId][steerKey(window, featureId, kind, value)];
  if (!result) {
    return { error: "This export did not precompute that feature here. Try one of the strongest features on this frame, or run ii view RUN --steer locally to steer anything." };
  }
  return { result };
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
    if (steerStatic()) runSteer();
  });
  value.addEventListener("input", () => {
    const presets = steerPresets(state.steer.kind);
    state.steer.value = presets ? presets[Number(value.value)] : Number(value.value);
    paintSteer();
    if (steerStatic()) runSteer();
  });
  run.addEventListener("click", runSteer);
  show.addEventListener("change", () => {
    state.steer.show = show.checked;
    renderFrame({ seek: false });
  });
  const usable = state.run.steer.sae_usable;
  const tryIt = el("button", { type: "button", class: "steer-idea-run", "data-testid": "steer-idea-run" });
  tryIt.addEventListener("click", () => {
    const idea = state.steer.idea;
    if (!idea) return;
    state.steer.kind = idea.kind;
    state.steer.value = nearestPreset(idea.kind, idea.value);
    kind.value = idea.kind;
    runSteer();
  });
  return el("div", { class: "steer", "data-testid": "steer-panel" }, [
    el("div", { class: "steer-head" }, [
      el("h3", { class: "panel-title" }, "Playground"),
      el("span", { class: "steer-verified", "data-testid": "steer-verified", hidden: "" }, "Verified feature"),
    ]),
    el("p", { class: "steer-target", "data-testid": "steer-target" }),
    el("div", { class: "steer-idea", "data-testid": "steer-idea", hidden: "" }, [
      el("p", { class: "steer-idea-kicker" }),
      el("p", { class: "steer-idea-text" }),
      tryIt,
    ]),
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
  const presets = steerPresets(state.steer.kind);
  if (presets) {
    // Static exports only have the precomputed strengths, so the slider steps through them.
    if (!presets.includes(state.steer.value)) state.steer.value = presets.includes(spec.value) ? spec.value : presets[0];
    value.min = "0";
    value.max = String(presets.length - 1);
    value.step = "1";
    value.value = String(presets.indexOf(state.steer.value));
  } else {
    value.min = String(spec.min);
    value.max = String(spec.max);
    value.step = String(spec.step);
    value.value = String(state.steer.value);
  }
  value.disabled = state.steer.kind === "ablate";
  panel.querySelector("[data-testid=steer-readout]").textContent = state.steer.kind === "ablate" ? "" : `${state.steer.value}`;
  panel.querySelector("[data-testid=steer-help]").textContent = spec.help;
  const target = panel.querySelector("[data-testid=steer-target]");
  const feature = state.activeFeature;
  target.replaceChildren(feature ? "Steer " : "Pick a feature above to steer it.");
  if (feature) target.append(el("strong", {}, displayLabel(feature)), " and see where the prediction moves.");
  const kindSelect = panel.querySelector("[data-testid=steer-kind]");
  if (kindSelect.value !== state.steer.kind) kindSelect.value = state.steer.kind;
  paintIdea(panel, feature);
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
      `${kindLabel}${amount} on ${result.features.map((id) => displayLabel(featureRecord({ id }))).join(", ")}`,
      el("span", { class: "num" }, ` (window ${result.window})`),
      ": ",
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
      `. The feature fired on ${sum.tokens_where_feature_fired} of ${sum.tokens_total} ${state.clip?.layout === "actions" ? "tokens" : "patches"} in this window.`,
    ]),
  );
  if (!steerResultHere()) box.append(el("p", { class: "steer-warn" }, "This frame is in another window. Steer again to see it here."));
}

// The full scoreboard record for a feature on the frame, when the run has one.
function featureRecord(feature) {
  const known = (state.run.features || []).find((item) => Number(item.id) === Number(feature.id));
  return known || { id: feature.id, label: feature.label, badge: feature.badge, status: feature.status, scored: false };
}

// One edit worth trying on the selected feature, chosen from what its checks said.
// After it runs, the card moves on to the other idea, so there is always a next one.
function paintIdea(panel, feature) {
  const box = panel.querySelector("[data-testid=steer-idea]");
  const mark = panel.querySelector("[data-testid=steer-verified]");
  const g = guide();
  const real = Boolean(feature && feature.badge === "real");
  panel.dataset.real = real ? "true" : "false";
  mark.hidden = !real;
  if (!feature || !g || !g.experiments) {
    box.hidden = true;
    state.steer.idea = null;
    return;
  }
  const record = featureRecord(feature);
  const plan = g.experiments(record, guideCtx(record));
  const last = state.steer.result;
  const ranHere = (idea) => last && last.clip_id === state.clipId
    && last.features.length === 1 && Number(last.features[0]) === Number(feature.id)
    && last.kind === idea.kind && (idea.kind === "ablate" || Number(last.value) === Number(nearestPreset(idea.kind, idea.value)));
  const done = plan.ideas.findIndex(ranHere);
  const idea = plan.ideas[done === -1 ? 0 : (done + 1) % plan.ideas.length];
  state.steer.idea = idea;
  box.hidden = false;
  box.dataset.kind = plan.key;
  box.querySelector(".steer-idea-kicker").textContent = done === -1
    ? (real ? "A good one to steer" : "An experiment to try")
    : "Next, try";
  box.querySelector(".steer-idea-text").replaceChildren(el("strong", {}, `${idea.title}. `), idea.text);
  const button = box.querySelector("[data-testid=steer-idea-run]");
  button.textContent = `${ideaName(idea)} →`;
  button.disabled = state.steer.busy;
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
    if (steerStatic()) {
      const found = await staticSteer(state.clipId, Number(frame.window) || 0, feature.id, state.steer.kind, state.steer.value);
      if (token !== state.steer.token) return;
      state.steer.error = found.error || "";
      state.steer.result = found.result || null;
      if (found.result) {
        state.steer.show = true;
        const toggle = document.querySelector("[data-testid=steer-show]");
        if (toggle) toggle.checked = true;
      }
      return;
    }
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

// ---- steering scorecard (steering/scorecard.json, written by `ii steer-scorecard`) ----
// Offline measurement on held-out clips. It never sets a badge; the scoreboard does.

function steeringCard() {
  const card = state.run && state.run.steering;
  return card && Array.isArray(card.rows) ? card : null;
}

function steerTarget(card) {
  return card.target === "action_chunk" ? "planned actions" : "prediction";
}

function steerMoves(card) {
  return card.target === "action_chunk" ? "the planned actions move" : "the prediction moves";
}

function fmtPct(value) {
  return Number.isFinite(value) ? `${Math.round(100 * value)}%` : "—";
}

function fmtCi(low, high, render) {
  return Number.isFinite(low) && Number.isFinite(high) ? `[${render(low)}, ${render(high)}]` : "[no interval]";
}

function fmtRatio(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}×` : "—";
}

function steerEditName(row) {
  return row.kind === "remove" ? "Remove" : `Boost ×${Number(row.factor).toLocaleString()}`;
}

function steerVerdict(row) {
  if (!row.n_windows) return { key: "silent", text: "Never fires" };
  return row.beats_controls ? { key: "beats", text: "Beats random edits" } : { key: "within", text: "Like a random edit" };
}

// One shared axis: relative change of the output, 0 at the left.
function steerAxisMax(rows) {
  let peak = 0;
  for (const row of rows) {
    for (const block of [row.action_change, row.control_change]) {
      for (const value of [block.mean, block.ci_high]) if (Number.isFinite(value)) peak = Math.max(peak, value);
    }
  }
  return peak > 0 ? peak * 1.08 : 1;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, title = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (title) {
    const tip = document.createElementNS(SVG_NS, "title");
    tip.textContent = title;
    node.append(tip);
  }
  return node;
}

// Interval strip: the controls as a grey band (95% interval) with a tick at their mean,
// the feature edit as a cobalt dot (mean) on a whisker (95% interval).
function steerStrip(row, max, target) {
  const width = 200;
  const height = 30;
  const pad = 6;
  const x = (value) => pad + (Math.max(0, value) / max) * (width - 2 * pad);
  const svg = svgEl("svg", { class: "steer-strip", viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": row.n_windows
      ? `${steerEditName(row)} moves the ${target} by ${fmtSmall(row.action_change.mean)}; matched controls ${fmtSmall(row.control_change.mean)}`
      : "never fires on held-out windows" });
  const mid = height / 2;
  svg.append(svgEl("line", { x1: pad, x2: width - pad, y1: mid, y2: mid, class: "steer-axis" }));
  if (!row.n_windows) return svg;
  const ctrl = row.control_change;
  const ctrlTip = `Matched random controls: mean ${fmtSmall(ctrl.mean)} ${fmtCi(ctrl.ci_low, ctrl.ci_high, fmtSmall)}`;
  if (Number.isFinite(ctrl.ci_low) && Number.isFinite(ctrl.ci_high)) {
    const left = x(ctrl.ci_low);
    svg.append(svgEl("rect", { x: left, y: mid - 7, width: Math.max(2, x(ctrl.ci_high) - left), height: 14, rx: 3, class: "steer-ctrl-band" }, ctrlTip));
  }
  svg.append(svgEl("line", { x1: x(ctrl.mean), x2: x(ctrl.mean), y1: mid - 9, y2: mid + 9, class: "steer-ctrl-mean" }, ctrlTip));
  const edit = row.action_change;
  const editTip = `${steerEditName(row)}: mean ${fmtSmall(edit.mean)} ${fmtCi(edit.ci_low, edit.ci_high, fmtSmall)}`;
  if (Number.isFinite(edit.ci_low) && Number.isFinite(edit.ci_high)) {
    svg.append(svgEl("line", { x1: x(edit.ci_low), x2: x(edit.ci_high), y1: mid, y2: mid, class: "steer-edit-ci" }, editTip));
  }
  svg.append(svgEl("circle", { cx: x(edit.mean), cy: mid, r: 4.5, class: "steer-edit-dot" }, editTip));
  // Wide invisible hit target so the tooltip is easy to reach.
  svg.append(svgEl("rect", { x: 0, y: 0, width, height, fill: "transparent" }, `${editTip}\n${ctrlTip}`));
  return svg;
}

function steeringLegend(card) {
  const swatch = (cls) => el("span", { class: `steer-key ${cls}`, "aria-hidden": "true" });
  return el("p", { class: "steer-legend", "data-testid": "steering-legend" }, [
    el("span", {}, [swatch("steer-key-edit"), "The feature edit, with its likely range"]),
    el("span", {}, [swatch("steer-key-ctrl"), `${card.n_controls} random edits of the same size`]),
  ]);
}

// Plain words by default; the raw measurements carry `num` and show with the Numbers switch.
function steeringTable(card, rows, { showFeature }) {
  const max = steerAxisMax(card.rows);
  const target = steerTarget(card);
  const heads = [
    showFeature ? ["Feature", ""] : null,
    ["Edit", ""],
    [`Change in the ${target}`, ""],
    ["Compared with a random edit", ""],
    ["How often it beats every random edit", ""],
    ["Verdict", ""],
    ["Change", "num"],
    ["Excess", "num"],
    ["p", "num"],
  ].filter(Boolean);
  const body = rows.map((row) => {
    const verdict = steerVerdict(row);
    const rel = row.reliability || {};
    const ratio = row.ratio || {};
    const featureCell = showFeature
      ? el("td", {}, el("div", { class: "steer-feature" }, [
          el("a", { href: `#/features/${row.feature_id}` }, row.label ? row.label : `Feature ${row.feature_id}`),
          badge({ badge: row.real ? "real" : "unverified" }),
        ]))
      : null;
    const editCell = el("td", {}, [
      el("span", { class: "steer-edit-name" }, steerEditName(row)),
      row.n_windows ? el("span", { class: "steer-sub steer-wrap" }, `Fires in ${fmtPct(row.coverage)} of unseen moments`) : null,
      row.n_windows ? el("span", { class: "steer-sub num" }, `${row.n_windows} windows · ${row.n_episodes} episodes`) : null,
    ]);
    const verdictCell = el("td", {}, el("span", { class: `steer-verdict steer-${verdict.key}`, "data-testid": "steering-verdict" }, verdict.text));
    const cells = row.n_windows
      ? [
          el("td", {}, steerStrip(row, max, target)),
          el("td", { "data-testid": "steering-ratio" }, [
            el("span", { class: "steer-figure" }, Number.isFinite(ratio.mean) ? `${ratio.mean.toFixed(1)}× as far` : "—"),
            el("span", { class: "steer-sub" }, Number.isFinite(ratio.ci_low) && Number.isFinite(ratio.ci_high)
              ? `Likely ${ratio.ci_low.toFixed(1)}× to ${ratio.ci_high.toFixed(1)}×`
              : "Too few episodes for a range"),
          ]),
          el("td", { "data-testid": "steering-reliability" }, [
            el("span", { class: "steer-figure" }, fmtPct(rel.rate)),
            Number.isFinite(rel.ci_low) && Number.isFinite(rel.ci_high)
              ? el("span", { class: "steer-sub" }, `Likely ${fmtPct(rel.ci_low)} to ${fmtPct(rel.ci_high)}`)
              : null,
            el("span", { class: "steer-sub" }, `Chance is ${fmtPct(rel.chance)}`),
          ]),
          verdictCell,
          el("td", { class: "num" }, [fmtSmall(row.action_change.mean), el("span", { class: "steer-sub" }, `random ${fmtSmall(row.control_change.mean)}`)]),
          el("td", { class: "num" }, [fmtSmall(row.excess.mean), el("span", { class: "steer-sub" }, fmtCi(row.excess.ci_low, row.excess.ci_high, fmtSmall))]),
          el("td", { class: "num" }, Number.isFinite(row.p_value) ? row.p_value.toFixed(3) : "—"),
        ]
      : [
          el("td", { colspan: "3", class: "steer-sub" }, "It never fires on an unseen moment, so there is nothing to edit."),
          verdictCell,
          el("td", { class: "num" }, "—"),
          el("td", { class: "num" }, "—"),
          el("td", { class: "num" }, "—"),
        ];
    return el("tr", {
      "data-testid": "steering-row",
      "data-feature-id": row.feature_id,
      "data-kind": row.kind,
      "data-beats": row.beats_controls ? "true" : "false",
    }, [featureCell, editCell, ...cells]);
  });
  return el("div", { class: "table-wrap" }, el("table", { class: "top-table steer-table", "data-testid": "steering-table" }, [
    el("thead", {}, el("tr", {}, heads.map(([label, cls]) => el("th", cls ? { class: cls } : {}, label)))),
    el("tbody", {}, body),
  ]));
}

function steeringNote(card) {
  return el("p", { class: "guide-note", "data-testid": "steering-note" }, [
    `Measured on ${card.n_episodes} unseen episodes. The likely ranges come from resampling those episodes, so they are wide when there are few. `,
    "This scorecard does not change any badge or the scoreboard",
    card.target === "action_chunk" ? ", and it uses recorded clips, so it is not a closed-loop rollout finding." : ".",
  ]);
}

function renderSteering() {
  const card = steeringCard();
  if (!card) {
    app.replaceChildren(el("section", { class: "feature-index", "data-testid": "steering-missing" }, [
      el("h1", {}, "Steering"),
      el("p", { class: "lede" }, richText("No steering scorecard yet. Run `ii steer-scorecard RUN` to make one.")),
    ]));
    return;
  }
  const target = steerTarget(card);
  const beats = card.rows.filter((row) => row.n_windows && row.beats_controls).length;
  const who = card.selection === "verified" ? "verified feature" : "chosen feature";
  app.replaceChildren(el("section", { class: "feature-index steering", "data-testid": "steering-page" }, [
    el("h1", {}, card.target === "action_chunk" ? "How much do the planned actions move?" : "How much does the prediction move?"),
    el("p", { class: "lede" }, [
      el("b", {}, `${beats} of ${card.rows.length} edits move the ${target} more than a random edit.`),
      ` We removed each ${who} and boosted it ${Number(card.boost_factor).toLocaleString()}×, on clips the model never trained on, and compared each change with ${card.n_controls} random edits of the same size.`,
    ]),
    steeringLegend(card),
    steeringTable(card, card.rows, { showFeature: true }),
    el("details", { class: "guide-asks steer-howto" }, [
      el("summary", {}, "How to read this"),
      el("p", {}, richText(
        `How far is the change in the ${target} against the unedited ${target} (\`||edited − clean|| / ||clean||\`), averaged over unseen moments where the feature fires. ` +
        "Compared with a random edit divides that by how far the random edits moved it. How often is the share of moments where the feature edit moved it further than every random edit did; chance is 1 / (random edits + 1). " +
        `An edit beats random edits when its p-value is at most ${card.alpha} and its lead over them stays above zero across the likely range.`,
      )),
    ]),
    steeringNote(card),
  ]));
  document.title = `steering · ${state.run.run_id} · interpretableintelligence`;
}

function steeringSection(featureId) {
  const card = steeringCard();
  if (!card) return null;
  const rows = card.rows.filter((row) => row.feature_id === featureId);
  if (!rows.length) return null;
  return el("section", { class: "page-section steering-feature", "data-testid": "steering-feature" }, [
    el("h2", { class: "section-title" }, "Steering it"),
    el("p", { class: "section-note" }, `How much ${steerMoves(card)} when this feature is removed or boosted, next to ${card.n_controls} random edits of the same size.`),
    steeringLegend(card),
    steeringTable(card, rows, { showFeature: false }),
    el("p", { class: "guide-note" }, [el("a", { href: "#/steering" }, "See every feature"), ". This does not change the badge."]),
  ]);
}
