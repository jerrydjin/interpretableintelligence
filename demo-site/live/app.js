// ii live playground: the page. The model runs on the playground backend (remote.js, API v1)
// when config.json or ?api= names one, otherwise in the browser (engine.js).

import { Engine } from "./engine.js";
import { RemoteEngine } from "./remote.js";

const PARAMS = new URLSearchParams(location.search);
const PACK = PARAMS.get("pack") || "packs/dinov2-small-droid/";

async function apiUrl() {
  if (PARAMS.has("api")) return PARAMS.get("api") || null; // ?api= (empty) forces the browser
  try {
    const config = await (await fetch(new URL("config.json", location.href), { cache: "no-cache" })).json();
    return typeof config.api === "string" && config.api ? config.api : null;
  } catch {
    return null;
  }
}
const EMBER = [232, 88, 42];
const COBALT = [58, 91, 217];
const TOP_N = 10;
const EDITS = {
  ablate: { name: "Switch off", hint: "Sets the feature to zero on every patch." },
  scale: { name: "Turn up or down", hint: "Multiplies the feature where it already fires." },
  clamp: { name: "Force on", hint: "Writes the feature onto every patch, even where it was silent." },
};

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid.nodeType ? kid : String(kid));
  return node;
};

let engine = null;
const ui = {
  state: null, // the current picture's encoding
  source: null, // what was encoded (img or canvas)
  view: "overview", // overview | patch | feature | search
  feature: null,
  patch: null,
  hoverFeature: null,
  kind: "ablate",
  value: { scale: 2, clamp: null },
  result: null,
  show: "fires", // fires | change
  gallery: [], // [{sample, state, img}]
  cam: null,
  camLoop: false,
  busy: false,
  pending: null,
  lastEditMs: null,
  lastEncodeMs: null,
};
window.__ii = { engine, ui }; // for tests and the curious; engine is set in boot()

// ------------------------------------------------------------------ helpers

const featureById = (id) => engine.features[id];
const featureName = (f) => (f.label ? capitalise(f.label) : `Unnamed feature ${f.id}`);
function capitalise(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
const pct = (x) => `${(x * 100).toFixed(x < 0.01 ? 2 : 1)}%`;
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

function tick() {
  return el("span", { class: "tick", title: "Verified: passed all four checks" }, "✓");
}

// ------------------------------------------------------------------ loading

async function boot() {
  const fill = $("loader-fill");
  const text = $("loader-text");
  try {
    const api = await apiUrl();
    const base = new URL(PACK, location.href).href;
    engine = api ? new RemoteEngine(base, new URL(api, location.href).href) : new Engine(base);
    window.__ii.engine = engine;
    await engine.load({
      preferGpu: !PARAMS.has("cpu"), // ?cpu forces the CPU path
      onProgress: ({ phase, loaded, total, cached }) => {
        if (phase === "connect") {
          fill.style.width = "40%";
          text.textContent = "Waking the model up. The first visit in a while takes a few seconds.";
        } else if (phase === "download") {
          const frac = total ? loaded / total : 0;
          fill.style.width = `${Math.round(frac * 80)}%`;
          text.textContent = cached
            ? "Loading the model from this device."
            : `Downloading the model: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB. Only the first time.`;
        } else if (phase === "compile") {
          fill.style.width = "88%";
          text.textContent = "Starting the model on your device.";
        }
      },
    });
  } catch (err) {
    text.textContent =
      engine && engine.backend === "remote"
        ? `The model server could not be reached (${err.message}). Try again in a minute.`
        : `The model could not start in this browser (${err.message}). Try a recent Chrome, Edge, Firefox or Safari.`;
    $("loader").classList.add("failed");
    console.error(err);
    return;
  }
  fill.style.width = "100%";
  const pack = engine.pack;
  $("model-pill").textContent = `${pack.model.name.split("/").pop()} · ${WHERE[engine.backend]}`;
  if (engine.backend === "remote") {
    $("where-runs").textContent = "It runs on a small server in the cloud, which reads your picture and keeps nothing.";
  }
  $("n-real").textContent = String(pack.n_real);
  $("loader").hidden = true;
  $("work").hidden = false;
  buildThumbs();
  wireInputs();
  const first = pack.samples.find((s) => s.clip_id === "droid-028") || pack.samples[0];
  await showSample(first);
  renderPerf();
  encodeGallery();
}

// ------------------------------------------------------------------ pictures

function buildThumbs() {
  const box = $("thumbs");
  box.replaceChildren();
  for (const sample of engine.pack.samples) {
    const img = el("img", { src: engine.base + sample.file, alt: `Robot video ${sample.clip_id}`, loading: "lazy" });
    const btn = el("button", { class: "thumb", type: "button", "data-clip": sample.clip_id, title: sample.split === "heldout" ? "From a video the dictionary never saw" : "From a training video", onclick: () => showSample(sample) }, img);
    if (sample.split === "heldout") btn.append(el("span", { class: "thumb-new" }, "unseen"));
    box.append(btn);
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("that file is not a picture this browser can read"));
    img.src = src;
  });
}

async function showSample(sample) {
  stopCam();
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("on", t.dataset.clip === sample.clip_id));
  const img = await loadImage(engine.base + sample.file);
  await showPicture(img, img.src);
}

async function showPicture(img, src) {
  const photo = $("photo");
  photo.src = src;
  photo.hidden = false;
  $("cam").hidden = true;
  $("stage").style.aspectRatio = `${img.naturalWidth || img.width} / ${img.naturalHeight || img.height}`;
  ui.source = img;
  const t0 = performance.now();
  ui.state = await engine.encode(img, { readout: false });
  ui.lastEncodeMs = performance.now() - t0;
  ui.result = null;
  ui.patch = null;
  if (ui.view === "patch" || ui.view === "search") ui.view = ui.feature != null ? "feature" : "overview";
  if (ui.feature != null) await runEdit();
  render();
}

async function showFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("on"));
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    stopCam();
    await showPicture(img, url);
  } catch (err) {
    flash(err.message);
  }
}

function wireInputs() {
  $("file").addEventListener("change", (e) => showFile(e.target.files[0]));
  window.addEventListener("dragover", (e) => {
    e.preventDefault();
    document.body.classList.add("dragging");
  });
  window.addEventListener("dragleave", (e) => {
    if (e.relatedTarget == null) document.body.classList.remove("dragging");
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    document.body.classList.remove("dragging");
    showFile(e.dataTransfer.files[0]);
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) showFile(item.getAsFile());
  });
  $("cam-btn").addEventListener("click", () => (ui.cam ? snapCam() : startCam()));
  $("search").addEventListener("input", () => {
    ui.view = $("search").value.trim() ? "search" : ui.feature != null ? "feature" : "overview";
    renderPanel();
  });
  const stage = $("stage");
  stage.addEventListener("pointermove", (e) => drawGrid(patchAt(e)));
  stage.addEventListener("pointerleave", () => drawGrid(null));
  stage.addEventListener("click", (e) => {
    const p = patchAt(e);
    if (p == null || !ui.state) return;
    ui.patch = p;
    ui.view = "patch";
    ui.feature = null;
    ui.result = null;
    ui.show = "fires";
    render();
  });
  new ResizeObserver(() => drawHeat()).observe(stage);
}

// ------------------------------------------------------------------ camera

async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    const video = $("cam");
    video.srcObject = stream;
    await video.play();
    ui.cam = stream;
    video.hidden = false;
    $("photo").hidden = true;
    $("stage").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    $("cam-btn").textContent = "Take the picture";
    document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("on"));
    ui.camLoop = true;
    ui.source = video;
    camLoop();
  } catch (err) {
    flash("The camera is not available here. You can still upload a picture.");
  }
}

async function camLoop() {
  if (!ui.camLoop) return;
  const video = $("cam");
  if (!ui.busy && video.readyState >= 2) {
    ui.busy = true;
    const t0 = performance.now();
    try {
      ui.state = await engine.encode(video, { readout: ui.feature != null && ui.show === "change" });
      ui.lastEncodeMs = performance.now() - t0;
      if (ui.feature != null && ui.show === "change") ui.result = await engine.edit(ui.state, [ui.feature], ui.kind, currentValue());
      drawHeat();
      if (ui.view === "overview" && (!ui._lastPanel || performance.now() - ui._lastPanel > 600)) {
        ui._lastPanel = performance.now();
        renderPanel();
      }
      renderPerf();
    } finally {
      ui.busy = false;
    }
  }
  requestAnimationFrame(camLoop);
}

async function snapCam() {
  const video = $("cam");
  const canvas = el("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  stopCam();
  const url = canvas.toDataURL("image/jpeg", 0.9);
  const img = await loadImage(url);
  await showPicture(img, url);
}

function stopCam() {
  ui.camLoop = false;
  if (ui.cam) ui.cam.getTracks().forEach((t) => t.stop());
  ui.cam = null;
  $("cam-btn").textContent = "Use your camera";
}

// ------------------------------------------------------------------ editing

function currentValue() {
  if (ui.kind === "scale") return ui.value.scale;
  if (ui.kind === "clamp") {
    if (ui.value.clamp == null) ui.value.clamp = Number((featureById(ui.feature).max || 0.5).toFixed(2));
    return ui.value.clamp;
  }
  return 0;
}

// Only the newest slider position runs; a move during a run queues one more.
async function runEdit() {
  if (ui.feature == null || !ui.state) return;
  if (ui.busy && !ui.camLoop) {
    ui.pending = true;
    return;
  }
  ui.busy = true;
  try {
    do {
      ui.pending = false;
      ui.result = await engine.edit(ui.state, [ui.feature], ui.kind, currentValue());
      ui.lastEditMs = ui.result.timing.total;
      ui.show = "change";
      drawHeat();
      renderResult();
      renderPerf();
    } while (ui.pending);
  } finally {
    ui.busy = false;
  }
}

function openFeature(id) {
  ui.feature = id;
  ui.view = "feature";
  ui.result = null;
  ui.show = "fires";
  ui.value.clamp = null;
  $("search").value = "";
  render();
}

// ------------------------------------------------------------------ drawing

function patchAt(e) {
  const rect = $("stage").getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  if (x < 0 || y < 0 || x >= 1 || y >= 1 || !engine.gridW) return null;
  return Math.floor(y * engine.gridH) * engine.gridW + Math.floor(x * engine.gridW);
}

function sizeCanvas(canvas) {
  const rect = $("stage").getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return canvas.getContext("2d");
}

function paintMap(canvas, values, peak, rgb, floor = 0) {
  const g = sizeCanvas(canvas);
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!values) return;
  const W = engine.gridW;
  const H = engine.gridH;
  const small = new ImageData(W, H);
  for (let t = 0; t < W * H; t++) {
    const a = Math.min(1, Math.max(0, values[t] / (peak || 1)));
    const alpha = a <= floor ? 0 : Math.pow(a, 0.7) * 0.9;
    small.data[t * 4] = rgb[0];
    small.data[t * 4 + 1] = rgb[1];
    small.data[t * 4 + 2] = rgb[2];
    small.data[t * 4 + 3] = Math.round(alpha * 255);
  }
  const tmp = paintMap._tmp || (paintMap._tmp = document.createElement("canvas"));
  tmp.width = W;
  tmp.height = H;
  tmp.getContext("2d").putImageData(small, 0, 0);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

function drawHeat() {
  const canvas = $("heat");
  const key = $("stage-key");
  const stage = $("stage");
  stage.classList.remove("mapping");
  if (!ui.state) return paintMap(canvas, null);
  stage.classList.add("mapping");
  // With nothing chosen, the overview shows its strongest feature, so the picture is never bare.
  const id = ui.hoverFeature ?? ui.feature ?? (ui.view === "overview" ? spotlight() : null);
  if (ui.show === "change" && ui.result && ui.hoverFeature == null) {
    // Scale to the 97th percentile, so one outlier patch does not hide the rest.
    const sorted = Float32Array.from(ui.result.shift).sort();
    const peak = Math.max(sorted[Math.floor(sorted.length * 0.97)], 0.01);
    paintMap(canvas, ui.result.shift, peak, COBALT, 0.08);
    key.replaceChildren(el("span", { class: "swatch cobalt" }), "Where the model's view changed. Brighter means a bigger change.");
    return;
  }
  if (id != null) {
    const f = featureById(id);
    paintMap(canvas, engine.featureMap(ui.state, id), f.max || 1, EMBER);
    key.replaceChildren(el("span", { class: "swatch ember" }), `Where "${featureName(f)}" fires. Brighter means stronger.`);
    return;
  }
  stage.classList.remove("mapping");
  paintMap(canvas, null);
  key.replaceChildren("Hover a feature to see where it fires, or click a patch of the picture.");
}

function spotlight() {
  const totals = ui.state.totals;
  let best = null;
  for (let i = 0; i < totals.length; i++) if (totals[i] > 0 && (best == null || totals[i] > totals[best])) best = i;
  return best;
}

function drawGrid(hover) {
  const canvas = $("grid");
  const g = sizeCanvas(canvas);
  g.clearRect(0, 0, canvas.width, canvas.height);
  const cw = canvas.width / engine.gridW;
  const ch = canvas.height / engine.gridH;
  const box = (t, style, width) => {
    const x = (t % engine.gridW) * cw;
    const y = Math.floor(t / engine.gridW) * ch;
    g.strokeStyle = style;
    g.lineWidth = width * (window.devicePixelRatio || 1);
    g.strokeRect(x + 1, y + 1, cw - 2, ch - 2);
  };
  if (hover != null) box(hover, "rgba(255,255,255,0.9)", 1.5);
  if (ui.view === "patch" && ui.patch != null) box(ui.patch, "#141413", 2);
}

// ------------------------------------------------------------------ panel

function render() {
  drawHeat();
  drawGrid(null);
  renderPanel();
}

function renderPanel() {
  const body = $("panel-body");
  if (!ui.state) return;
  if (ui.view === "search") body.replaceChildren(searchView());
  else if (ui.view === "patch") body.replaceChildren(patchView());
  else if (ui.view === "feature" && ui.feature != null) body.replaceChildren(featureView());
  else body.replaceChildren(overview());
  if (ui.view === "feature") renderResult();
}

function featureRow(f, strength, peak) {
  const row = el(
    "button",
    {
      class: "frow",
      type: "button",
      "data-feature": f.id,
      onclick: () => openFeature(f.id),
      onmouseenter: () => {
        ui.hoverFeature = f.id;
        drawHeat();
      },
      onmouseleave: () => {
        ui.hoverFeature = null;
        drawHeat();
      },
      onfocus: () => {
        ui.hoverFeature = f.id;
        drawHeat();
      },
      onblur: () => {
        ui.hoverFeature = null;
        drawHeat();
      },
    },
    el("span", { class: f.label ? "fname" : "fname unnamed" }, featureName(f), f.real ? tick() : null),
  );
  if (strength != null) {
    const bar = el("span", { class: "fbar" }, el("span", { style: `width:${Math.max(4, Math.round((strength / (peak || 1)) * 100))}%` }));
    row.append(bar);
  }
  return row;
}

function overview() {
  const totals = ui.state.totals;
  const order = [...totals.keys()].filter((i) => totals[i] > 0).sort((a, b) => totals[b] - totals[a]);
  const top = order.slice(0, TOP_N);
  const named = order.filter((i) => featureById(i).label && !top.includes(i)).slice(0, 4);
  const verified = engine.features.filter((f) => f.real);
  const peak = totals[top[0]] || 1;
  return el(
    "div",
    { class: "view" },
    el("h2", {}, "What it notices here"),
    el("p", { class: "muted" }, `${order.length} of ${engine.nFeatures.toLocaleString("en-GB")} features fire somewhere on this picture. These are the strongest. Hover one to see where, click it to play with it.`),
    el("div", { class: "flist" }, top.map((i) => featureRow(featureById(i), totals[i], peak))),
    named.length ? el("h3", {}, "Also here, with names") : null,
    named.length ? el("div", { class: "flist" }, named.map((i) => featureRow(featureById(i), totals[i], peak))) : null,
    el("h3", {}, "Features with a green tick"),
    el("p", { class: "muted" }, "These passed every check. Open one, even if it is quiet on this picture."),
    el("div", { class: "flist" }, verified.map((f) => featureRow(f, totals[f.id] || 0, peak))),
    el("p", { class: "hint" }, "Tip: click any patch of the picture to see everything the model notices there."),
  );
}

function patchView() {
  const row = ui.state.rows[ui.patch];
  const order = [...row.idx.keys()].sort((a, b) => row.val[b] - row.val[a]);
  const peak = row.val[order[0]] || 1;
  const r = Math.floor(ui.patch / engine.gridW) + 1;
  const c = (ui.patch % engine.gridW) + 1;
  return el(
    "div",
    { class: "view" },
    backLink(),
    el("h2", {}, "What it notices in this patch"),
    el("p", { class: "muted" }, `Row ${r}, column ${c}. The dictionary describes this patch with ${order.length} features. Click one to play with it.`),
    el("div", { class: "flist" }, order.map((j) => featureRow(featureById(row.idx[j]), row.val[j], peak))),
  );
}

function searchView() {
  const q = $("search").value.trim().toLowerCase();
  const asNum = /^\d+$/.test(q) ? Number(q) : null;
  const hits = engine.features
    .filter((f) => (asNum != null && f.id === asNum) || (f.label && f.label.toLowerCase().includes(q)) || (q === "verified" && f.real))
    .sort((a, b) => (b.real ? 1 : 0) - (a.real ? 1 : 0) || (ui.state.totals[b.id] || 0) - (ui.state.totals[a.id] || 0))
    .slice(0, 40);
  const named = engine.features.filter((f) => f.label).length;
  return el(
    "div",
    { class: "view" },
    el("h2", {}, hits.length ? `${hits.length === 40 ? "First 40" : hits.length} found` : "Nothing found"),
    el("p", { class: "muted" }, hits.length ? "Bars show how strongly each one fires on this picture." : `${named} features have names so far; the rest you can open by number, 0 to ${engine.nFeatures - 1}.`),
    el("div", { class: "flist" }, hits.map((f) => featureRow(f, ui.state.totals[f.id] || 0, Math.max(...hits.map((h) => ui.state.totals[h.id] || 0)) || 1))),
  );
}

function backLink() {
  return el(
    "button",
    {
      class: "back",
      type: "button",
      onclick: () => {
        ui.view = "overview";
        ui.feature = null;
        ui.patch = null;
        ui.result = null;
        ui.show = "fires";
        render();
      },
    },
    "← Everything on this picture",
  );
}

function checksList(f) {
  const c = f.checks || {};
  const item = (ok, text, untested) =>
    el("li", { class: ok === true ? "ok" : ok === false ? "no" : "na" }, el("span", { class: "mark-dot" }, ok === true ? "✓" : ok === false ? "✗" : "–"), text, ok == null && untested ? el("span", { class: "muted" }, ` (${untested})`) : null);
  const tested = c.random_percentile != null;
  return el(
    "ul",
    { class: "checks" },
    item(c.distinct_clips_top20 == null ? null : c.distinct_clips_top20 >= 5, "Fires across many different videos", "not checked"),
    item(c.label_holds_heldout == null ? null : c.label_holds_heldout === true, f.label ? "Its name still fits held-back videos" : "Has a name that fits held-back videos", "not checked"),
    item(tested ? c.random_percentile >= 95 : null, "Switching it off beats random look-alike edits", "not tested yet"),
    item(c.absent_in_random_init == null ? null : c.absent_in_random_init === true, "Learnt, not there by chance in an untrained model", "not checked"),
  );
}

function featureView() {
  const f = featureById(ui.feature);
  const map = engine.featureMap(ui.state, f.id);
  let fired = 0;
  let best = 0;
  for (const v of map) {
    if (v > 0) fired++;
    if (v > best) best = v;
  }
  const status = f.real
    ? el("p", { class: "status real" }, tick(), " Verified: it passed all four checks.")
    : el("p", { class: "status" }, f.label ? "Not verified. The name is a guess that has not passed every check." : "Not verified and not named yet. Look at where it fires and guess what it is.");
  const kinds = el(
    "div",
    { class: "seg", role: "tablist" },
    Object.entries(EDITS).map(([kind, meta]) =>
      el(
        "button",
        {
          type: "button",
          role: "tab",
          class: kind === ui.kind ? "on" : "",
          "aria-selected": kind === ui.kind ? "true" : "false",
          "data-kind": kind,
          onclick: () => {
            ui.kind = kind;
            renderPanel();
            runEdit();
          },
        },
        meta.name,
      ),
    ),
  );
  const control = el("div", { class: "control", id: "control" });
  if (ui.kind === "scale") {
    control.append(slider("Strength", 0, 5, 0.1, ui.value.scale, (v) => `× ${v.toFixed(1)}`, (v) => (ui.value.scale = v)));
  } else if (ui.kind === "clamp") {
    const peak = Math.max(f.max || 0.5, 0.05);
    control.append(slider("Set every patch to", 0, Number((peak * 2).toFixed(2)), Number((peak / 50).toFixed(3)), currentValue(), (v) => v.toFixed(2), (v) => (ui.value.clamp = v)));
  }
  const toggle = el(
    "div",
    { class: "seg small" },
    el("button", { type: "button", class: ui.show === "fires" ? "on" : "", "data-show": "fires", onclick: () => { ui.show = "fires"; drawHeat(); renderPanel(); } }, "Where it fires"),
    el("button", { type: "button", class: ui.show === "change" ? "on" : "", "data-show": "change", onclick: () => { ui.show = "change"; if (!ui.result) runEdit(); else { drawHeat(); renderPanel(); } } }, "What changed"),
  );
  return el(
    "div",
    { class: "view" },
    backLink(),
    el("p", { class: "fid" }, `Feature ${f.id}`),
    el("h2", { class: "ftitle" }, featureName(f)),
    status,
    checksList(f),
    el("h3", {}, "On this picture"),
    el("p", { class: "muted", id: "fires-text" }, fired ? `It fires on ${fired} of ${engine.nTok} patches. Its strongest here is ${pct(best / (f.max || 1))} of the most it ever reached on the training videos.` : "It is silent on this picture. Try Force on to see what it would do, or pick one of the pictures below where it fires."),
    el("h3", {}, "Change it and watch"),
    kinds,
    el("p", { class: "muted small" }, EDITS[ui.kind].hint),
    control,
    el("div", { class: "result", id: "result" }),
    toggle,
    el("h3", {}, "Where else it fires"),
    el("div", { class: "gallery", id: "gallery" }, galleryFor(f.id)),
  );
}

function slider(label, min, max, step, value, fmt, set) {
  const out = el("output", {}, fmt(value));
  const input = el("input", { type: "range", min, max, step, value, "aria-label": label });
  input.addEventListener("input", () => {
    const v = Number(input.value);
    set(v);
    out.textContent = fmt(v);
    runEdit();
  });
  return el("label", { class: "slider" }, el("span", {}, label), input, out);
}

function renderResult() {
  const box = $("result");
  if (!box) return;
  const r = ui.result;
  if (!r) {
    box.replaceChildren(el("p", { class: "muted" }, "Pick an edit above. The rest of the model runs again on your device."));
    return;
  }
  const f = featureById(ui.feature);
  const label = ui.kind === "ablate" ? "Switching it off" : ui.kind === "scale" ? `Scaling it by ${ui.value.scale.toFixed(1)}` : `Forcing it to ${currentValue().toFixed(2)} everywhere`;
  const quiet = r.max < 0.002;
  box.replaceChildren(
    el(
      "p",
      { class: "lead-num" },
      quiet ? `${label} barely moved the model's view of this picture.` : `${label} moved the model's final view by ${pct(r.mean)} on average, and by up to ${pct(r.max)} on one patch.`,
    ),
    el("p", { class: "muted small" }, `Measured as how far each patch's final description turned (cosine distance), comparing before and after the edit. ${quiet && !engine.featureMap(ui.state, f.id).some((v) => v > 0) && ui.kind !== "clamp" ? "It is silent here, so there was nothing to remove." : ""}`),
  );
  document.querySelectorAll("[data-show]").forEach((b) => b.classList.toggle("on", b.dataset.show === ui.show));
}

// ------------------------------------------------------------------ gallery (computed on this device)

async function encodeGallery() {
  for (const sample of engine.pack.samples) {
    if (ui.gallery.some((g) => g.sample === sample)) continue;
    // Yield to the picture the user is looking at.
    while (ui.busy || ui.camLoop) await new Promise((r) => setTimeout(r, 150));
    const img = await loadImage(engine.base + sample.file);
    ui.busy = true;
    try {
      const state = await engine.encode(img, { readout: false });
      ui.gallery.push({ sample, img, state: { rows: state.rows, peaks: state.peaks } });
    } finally {
      ui.busy = false;
    }
    if (ui.pending) runEdit();
    const box = $("gallery");
    if (box && ui.feature != null) box.replaceChildren(galleryFor(ui.feature));
  }
}

function galleryFor(id) {
  if (!ui.gallery.length) return el("p", { class: "muted" }, "Looking through the sample pictures on your device…");
  const f = featureById(id);
  const ranked = ui.gallery
    .map((g) => ({ g, peak: g.state.peaks[id] }))
    .filter((x) => x.peak > 0)
    .sort((a, b) => b.peak - a.peak)
    .slice(0, 6);
  const done = ui.gallery.length === engine.pack.samples.length;
  if (!ranked.length) return el("p", { class: "muted" }, done ? `It does not fire on any of the ${ui.gallery.length} sample pictures. Try your own.` : "Not on the pictures checked so far…");
  return ranked.map(({ g, peak }) => {
    const canvas = el("canvas", { width: 160, height: 120 });
    const ctx = canvas.getContext("2d");
    ctx.drawImage(g.img, 0, 0, 160, 120);
    const W = engine.gridW;
    const H = engine.gridH;
    const small = new ImageData(W, H);
    g.state.rows.forEach((row, t) => {
      const j = row.idx.indexOf(id);
      const a = j >= 0 ? Math.min(1, row.val[j] / (f.max || 1)) : 0;
      small.data.set([EMBER[0], EMBER[1], EMBER[2], Math.round(Math.pow(a, 0.8) * 0.82 * 255)], t * 4);
    });
    const tmp = document.createElement("canvas");
    tmp.width = W;
    tmp.height = H;
    tmp.getContext("2d").putImageData(small, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, 0, 0, 160, 120);
    return el(
      "button",
      { class: "gthumb", type: "button", title: `Open ${g.sample.clip_id}`, onclick: () => showSample(g.sample) },
      canvas,
      el("span", {}, pct(peak / (f.max || 1))),
    );
  });
}

// ------------------------------------------------------------------ footer

const WHERE = { remote: "in the cloud", webgpu: "on your GPU", wasm: "on your CPU" };

function renderPerf() {
  const t = engine.timings;
  const where = {
    remote: `Running on the model server (${engine.device === "cuda" ? "GPU" : "CPU"})`,
    webgpu: "Running on WebGPU",
    wasm: `Running on WebAssembly (${self.crossOriginIsolated ? "multi-threaded" : "one thread"})`,
  };
  const parts = [
    where[engine.backend],
    `ready in ${fmtMs(t.ready)}`,
  ];
  if (ui.lastEncodeMs != null) parts.push(`reading a picture ${fmtMs(ui.lastEncodeMs)}`);
  if (ui.lastEditMs != null) parts.push(`one edit ${fmtMs(ui.lastEditMs)}`);
  $("perf").textContent = parts.join(" · ") + ".";
}

function flash(message) {
  const box = el("div", { class: "toast", role: "status" }, message);
  document.body.append(box);
  setTimeout(() => box.remove(), 4000);
}

boot();
