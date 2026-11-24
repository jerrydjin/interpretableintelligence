// ii playground, words first: what does the model see, and which features make it see that?
// Runs a version 2 pack with the words readout (ii export-live --readout words) through
// engine.js, in the browser. Exploration only: nothing here can change a feature's badge.

import { Engine } from "./engine.js";

const PARAMS = new URLSearchParams(location.search);
const PACK = PARAMS.get("pack") || "packs/clip-b16-words/";
const EMBER = [232, 88, 42];
const FIRST = "01-coco-14475"; // the picture the page opens on
const BAKE = PARAMS.has("bake");
const N_WORDS = 6; // words shown
const N_WHY = 6; // features shown for a word
const THUMB = 44; // px, the crop of the picture shown beside each feature
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
const pct = (p) => (p >= 0.995 ? "99%" : p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.01 ? `${(p * 100).toFixed(1)}%` : p >= 0.001 ? `${(p * 100).toFixed(2)}%` : "<0.1%");
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

let engine = null;
const ui = {
  state: null, // the picture's encoding, with .seen (the words)
  picture: null, // {sample} or {img}
  word: null, // index of the word being explained
  why: null, // engine.why() for that word
  off: new Set(), // features switched off
  result: null, // words after switching them off
  control: null, // words after switching off as many random features that also fire here
  feature: null, // feature open in detail
  mode: "scale", // detail edit: scale | clamp
  strength: { scale: 3, clamp: 2 },
  detail: null, // words after the detail edit
  hover: null,
  crop: null, // the square the model sees, for the feature thumbnails
  background: new Set(), // features that fire on every picture, left out of "why"
  gallery: [],
  busy: 0,
  timings: {},
  live: false, // the model is running; until then the page shows answers baked ahead of time
  baked: null, // the baked answer for the picture on show (instant.json), if any
  pendingFeature: null, // a feature to open once the model is ready
};
window.__ii = { ui };

// ------------------------------------------------------------------ loading

async function boot() {
  const fill = $("loader-fill");
  const text = $("loader-text");
  engine = new Engine(new URL(PACK, location.href).href);
  window.__ii.engine = engine;
  // A first answer for each example picture, worked out ahead of time by this same page
  // (ii bake-live), so the page answers at once while the model downloads.
  if (!BAKE) {
    try {
      await engine.loadMeta();
      if (engine.wordsMode) ui.instant = await engine.loadInstant();
    } catch {
      ui.instant = null;
    }
    if (ui.instant) {
      showMeta();
      $("work").hidden = false;
      buildThumbs();
      wireInputs();
      if (!(await restoreFromLink())) await showSample(engine.pack.samples.find((s) => s.clip_id === FIRST) || engine.pack.samples[0]);
    }
  }
  try {
    await engine.load({
      preferGpu: !PARAMS.has("cpu"),
      onProgress: ({ phase, loaded, total, cached }) => {
        if (phase === "download") {
          fill.style.width = `${Math.round((total ? loaded / total : 0) * 85)}%`;
          text.textContent = cached
            ? "Loading the model from this device."
            : `Downloading the model: ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB. Only the first time; after that it starts straight away.`;
        } else if (phase === "compile") {
          fill.style.width = "92%";
          text.textContent = "Starting the model on your device.";
        }
      },
    });
    if (!engine.wordsMode) throw new Error("this pack has no words readout; open it in the feature playground");
  } catch (err) {
    text.textContent = `The model could not start in this browser (${err.message}). Try a recent Chrome, Edge, Firefox or Safari.`;
    $("loader").classList.add("failed");
    console.error(err);
    return;
  }
  const pack = engine.pack;
  $("model-pill").textContent = `${pack.model.name.split("/").pop()} · ${engine.backend === "webgpu" ? "on your GPU" : "on your CPU"}`;
  showMeta();
  // CLIP keeps a few always-on features: very strong, on the same few patches of every picture
  // (or on nearly every patch), carrying the whole picture rather than what sits under them.
  // Switching them off moves any word, so "why" leaves them out and ranks the rest.
  ui.background = new Set(engine.features.filter((f) => (f.mean || 0) >= 3 || (f.freq || 0) >= 0.5).map((f) => f.id));
  $("loader").hidden = true;
  ui.live = true;
  if (ui.instant) await goLive();
  else {
    $("work").hidden = false;
    buildThumbs();
    wireInputs();
    if (!(await restoreFromLink())) await showSample(pack.samples.find((s) => s.clip_id === FIRST) || pack.samples[0]);
  }
  renderPerf();
  if (BAKE) return bake();
  encodeGallery();
}

function showMeta() {
  const pack = engine.pack;
  $("layer").textContent = String(pack.model.layer);
  $("n-features").textContent = pack.sae.n_features.toLocaleString("en-GB");
  $("n-words").textContent = `${engine.vocab.length} everyday things`;
}

// The model is ready: work out for real what the page has shown so far (the baked answer, or a
// picture, word or switch chosen while it loaded), keeping the same picture, word and switches.
async function goLive() {
  const feature = ui.pendingFeature;
  ui.pendingFeature = null;
  ui.baked = null;
  if (!ui.cam && ui.picture) {
    const img = ui.picture.sample ? await loadImage(engine.base + sampleOf(ui.picture.sample).file) : ui.picture.img;
    await showPicture(img, { word: ui.word, off: [...ui.off] });
  }
  if (feature != null) await openFeature(feature);
}

const sampleOf = (key) => engine.pack.samples.find((s) => s.clip_id === key);

// ------------------------------------------------------------------ baked answers (instant.json)

const unq8 = (b64, peak) => Float32Array.from(atob(b64), (c) => (c.charCodeAt(0) / 255) * peak);
const q8 = (values) => {
  let peak = 0;
  for (const v of values) if (v > peak) peak = v;
  const bytes = Array.from(values, (v) => Math.round((Math.max(0, v) / (peak || 1)) * 255));
  return [btoa(String.fromCharCode(...bytes)), +peak.toPrecision(5)];
};
const probs = (sparse) => {
  const prob = new Float32Array(engine.vocab.length);
  for (const [w, p] of Object.entries(sparse)) prob[w] = p;
  return { prob };
};
const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));

// Show the baked answer for an example picture; false when there is none for this word.
function showBaked(key, { word = null, off = [] } = {}) {
  const rec = ui.instant?.samples?.[key];
  if (!rec || (word != null && word !== rec.word)) return false;
  ui.baked = rec;
  ui.state = { baked: true, seen: probs(rec.prob), maps: new Map(Object.entries(rec.maps).map(([id, [b64, peak]]) => [Number(id), unq8(b64, peak)])) };
  ui.feature = null;
  ui.detail = null;
  ui.word = rec.word;
  ui.off = new Set(off);
  bakedWhy();
  bakedOff();
  render();
  return true;
}

function bakedWhy() {
  const rec = ui.baked;
  ui.why = rec && ui.word === rec.word ? { word: rec.word, ranked: rec.why.map(([id, score]) => ({ id, score })), patches: unq8(...rec.patches) } : null;
}

function bakedOff() {
  const rec = ui.baked;
  ui.result = ui.control = null;
  if (rec && ui.word === rec.word && ui.off.size && sameSet([...ui.off], rec.off.ids)) {
    ui.result = probs(rec.off.prob);
    const control = probs({ [rec.word]: rec.off.control.p });
    ui.control = { ids: rec.off.control.ids, seen: control };
  }
}

// ?bake: work out the first answer (and "switch them all off") for every example picture, for
// ii bake-live to save as instant.json. Runs the same code as a visitor's click.
async function bake() {
  const all = engine.pack.digests || {};
  const digests = Object.fromEntries(["front", "sae", "decoder", "words", "text", "vocab"].filter((k) => all[k]).map((k) => [k, all[k]]));
  const out = { version: 1, digests, backend: engine.backend, samples: {} };
  for (const sample of engine.pack.samples) {
    await showSample(sample);
    const word = ui.word;
    const list = reasons();
    const seen = ui.state.seen;
    const maps = {};
    for (const r of list) maps[r.id] = q8(engine.featureMap(ui.state, r.id));
    const rec = { word, why: list.map((r) => [r.id, +r.score.toPrecision(5)]), patches: q8(ui.why.patches), maps };
    const keep = new Set([word]);
    const top = (prob) => [...prob.keys()].sort((a, b) => prob[b] - prob[a]).slice(0, 12);
    top(seen.prob).forEach((w) => keep.add(w));
    if (list.length) {
      await setAll(false);
      top(ui.result.prob).forEach((w) => keep.add(w));
      const sparse = (prob) => Object.fromEntries([...keep].map((w) => [w, +prob[w].toPrecision(5)]));
      rec.off = { ids: [...ui.off], prob: sparse(ui.result.prob), control: { ids: ui.control.ids, p: +ui.control.seen.prob[word].toPrecision(5) } };
      rec.prob = sparse(seen.prob);
    } else {
      rec.off = { ids: [], prob: {}, control: { ids: [], p: 0 } };
      rec.prob = Object.fromEntries([...keep].map((w) => [w, +seen.prob[w].toPrecision(5)]));
    }
    out.samples[sample.clip_id] = rec;
  }
  window.__ii.baked = out;
}

// ------------------------------------------------------------------ pictures

function buildThumbs() {
  const box = $("thumbs");
  box.replaceChildren(
    ...engine.pack.samples.map((sample) =>
      el(
        "button",
        { class: "thumb", type: "button", "data-key": sample.clip_id, title: sample.credit || sample.clip_id, onclick: () => showSample(sample) },
        el("img", { src: engine.base + sample.file, alt: sample.clip_id, loading: "lazy" }),
      ),
    ),
  );
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

async function showSample(sample, opts = {}) {
  stopCam();
  ui.picture = { sample: sample.clip_id };
  markThumb(sample.clip_id);
  $("credit").textContent = sample.credit || "";
  const img = await loadImage(engine.base + sample.file);
  if (!ui.live && showPhoto(img) && showBaked(sample.clip_id, opts)) return;
  await showPicture(img, opts);
}

function showPhoto(img) {
  const photo = $("photo");
  photo.src = img.src;
  photo.hidden = false;
  $("cam").hidden = true;
  ui.crop = cropOf(img);
  return true;
}

function markThumb(key) {
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("on", t.dataset.key === key));
}

async function showPicture(img, { word = null, off = [] } = {}) {
  showPhoto(img);
  if (!ui.live) {
    // Read for real once the model is ready (goLive), with this word and these switches.
    Object.assign(ui, { state: null, why: null, result: null, control: null, baked: null, feature: null, word, off: new Set(off) });
    return render();
  }
  await busy(async () => {
    const t0 = performance.now();
    ui.state = await engine.encode(img);
    ui.timings.read = performance.now() - t0;
  });
  ui.feature = null;
  ui.detail = null;
  await explain(word ?? topWord(ui.state.seen), off);
}

async function busy(fn) {
  ui.busy++;
  document.body.classList.add("working");
  try {
    return await fn();
  } finally {
    ui.busy--;
    if (!ui.busy) document.body.classList.remove("working");
  }
}

function topWord(seen) {
  let best = 0;
  for (let w = 1; w < seen.prob.length; w++) if (seen.prob[w] > seen.prob[best]) best = w;
  return best;
}

// ------------------------------------------------------------------ the loop: word, why, switch off

async function explain(word, off = []) {
  ui.word = word;
  ui.off = new Set(off);
  ui.result = null;
  ui.control = null;
  if (!ui.live) {
    bakedWhy();
    bakedOff();
    return render();
  }
  ui.why = await busy(async () => {
    const t0 = performance.now();
    const why = await engine.why(ui.state, word);
    ui.timings.why = performance.now() - t0;
    return why;
  });
  if (ui.off.size) await runOff();
  render();
}

// A feature's name (the word writing it in makes the model see, ii name-live), else its number.
function fname(id) {
  return engine.features[id]?.name || `Feature ${id}`;
}

function reasons() {
  return ui.why ? ui.why.ranked.filter((r) => r.score > 0 && !ui.background.has(r.id)).slice(0, N_WHY) : [];
}

// The middle square of the picture, as the model sees it.
function cropOf(img) {
  const s = engine.size;
  const c = el("canvas", { width: s, height: s });
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const side = Math.min(w, h);
  c.getContext("2d").drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, s, s);
  return c;
}

// A small crop around the patches where a feature fires most on this picture.
function thumbOf(id) {
  const dpr = window.devicePixelRatio || 1;
  const c = el("canvas", { class: "fthumb", width: THUMB * dpr, height: THUMB * dpr, "aria-hidden": "true" });
  if (!ui.crop) return c;
  const map = engine.featureMap(ui.state, id);
  const W = engine.gridW;
  const H = engine.gridH;
  let peak = 0;
  for (const v of map) if (v > peak) peak = v;
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  map.forEach((v, t) => {
    if (peak && v >= 0.4 * peak) {
      const x = t % W, y = Math.floor(t / W);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  });
  if (x1 < 0) [x0, y0, x1, y1] = [0, 0, W - 1, H - 1];
  let side = Math.min(W, Math.max(4, x1 - x0 + 3, y1 - y0 + 3));
  const cx = (x0 + x1 + 1) / 2, cy = (y0 + y1 + 1) / 2;
  const sx = Math.min(W - side, Math.max(0, cx - side / 2));
  const sy = Math.min(H - side, Math.max(0, cy - side / 2));
  const px = engine.size / W;
  c.getContext("2d").drawImage(ui.crop, sx * px, sy * px, side * px, side * px, 0, 0, c.width, c.height);
  return c;
}

async function toggle(id) {
  if (ui.off.has(id)) ui.off.delete(id);
  else ui.off.add(id);
  await runOff();
  render();
}

async function setAll(on) {
  ui.off = on ? new Set() : new Set(reasons().map((r) => r.id));
  await runOff();
  render();
}

// Switch the chosen features off, and as many random features that also fire here (not among
// the ones ranked for this word), so the drop can be told apart from what any edit would do.
async function runOff() {
  if (!ui.live) {
    bakedOff();
    return syncAddress();
  }
  if (!ui.off.size) {
    ui.result = ui.control = null;
    return;
  }
  const ids = [...ui.off];
  await busy(async () => {
    const t0 = performance.now();
    ui.result = await engine.see(ui.state, ids.map((id) => ({ id, kind: "ablate" })));
    ui.timings.edit = performance.now() - t0;
    const ranked = new Set(ui.why.ranked.slice(0, 50).map((r) => r.id));
    const pool = ui.why.ranked.map((r) => r.id).filter((id) => !ranked.has(id) && !ui.off.has(id) && !ui.background.has(id));
    const picked = seeded(pool, ids.length, hashOf(ui.picture, ui.word, ids.length));
    ui.control = picked.length ? { ids: picked, seen: await engine.see(ui.state, picked.map((id) => ({ id, kind: "ablate" }))) } : null;
  });
  syncAddress();
}

function hashOf(...parts) {
  let h = 2166136261;
  for (const ch of JSON.stringify(parts)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

function seeded(pool, n, seed) {
  const out = pool.slice();
  let s = seed || 1;
  for (let i = out.length - 1; i > 0; i--) {
    s = Math.imul(s ^ (s >>> 15), 2246822507) >>> 0;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out.slice(0, n);
}

// ------------------------------------------------------------------ one feature

async function openFeature(id) {
  if (!ui.live) {
    ui.pendingFeature = id;
    return flash(`Feature ${id} opens as soon as the model is ready.`);
  }
  ui.feature = id;
  ui.detail = null;
  ui.hover = null;
  render();
  await runDetail();
}

async function runDetail() {
  const id = ui.feature;
  if (id == null) return;
  const f = engine.features[id];
  const edit = ui.mode === "scale" ? { id, kind: "scale", value: ui.strength.scale } : { id, kind: "clamp", value: ui.strength.clamp * (f.max || 1) };
  const seen = await busy(() => engine.see(ui.state, [edit]));
  if (ui.feature !== id) return;
  ui.detail = seen;
  renderPanel();
  syncAddress();
}

// ------------------------------------------------------------------ drawing

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

function paintMap(values, rgb) {
  const canvas = $("heat");
  const g = sizeCanvas(canvas);
  g.clearRect(0, 0, canvas.width, canvas.height);
  $("stage").classList.toggle("mapping", Boolean(values));
  if (!values) return;
  let peak = 0;
  for (const v of values) if (v > peak) peak = v;
  const W = engine.gridW;
  const H = engine.gridH;
  const small = new ImageData(W, H);
  for (let t = 0; t < W * H; t++) {
    const a = Math.min(1, Math.max(0, values[t] / (peak || 1)));
    small.data.set([rgb[0], rgb[1], rgb[2], Math.round(Math.pow(a, 0.7) * 0.9 * 255)], t * 4);
  }
  const tmp = paintMap._tmp || (paintMap._tmp = document.createElement("canvas"));
  tmp.width = W;
  tmp.height = H;
  tmp.getContext("2d").putImageData(small, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

function drawHeat() {
  const key = $("stage-key");
  if (!ui.state) return paintMap(null);
  const id = ui.hover ?? ui.feature;
  if (id != null) {
    paintMap(engine.featureMap(ui.state, id), EMBER);
    key.replaceChildren(el("span", { class: "swatch ember" }), engine.features[id]?.name ? `Where “${engine.features[id].name}” fires` : `Where feature ${id} fires`);
  } else if (ui.why) {
    paintMap(ui.why.patches, EMBER);
    key.replaceChildren(el("span", { class: "swatch ember" }), `Where it finds “${engine.vocab[ui.word]}”`);
  } else {
    paintMap(null);
    key.replaceChildren();
  }
}

function render() {
  drawHeat();
  renderPanel();
  renderPerf();
  syncAddress();
}

// ------------------------------------------------------------------ panel

function renderPanel() {
  const body = $("panel-body");
  if (!ui.state) {
    if (!ui.live) body.replaceChildren(el("div", { class: "view" }, el("p", { class: "muted" }, "It reads this picture as soon as the model is ready.")));
    return;
  }
  body.replaceChildren(ui.feature != null ? featureView() : mainView());
}

function wordList(seen, before = null, { pick = true } = {}) {
  const order = [...seen.prob.keys()].sort((a, b) => seen.prob[b] - seen.prob[a]).slice(0, N_WORDS);
  if (pick && ui.word != null && !order.includes(ui.word)) order.push(ui.word);
  return el(
    "div",
    { class: "words" },
    order.map((w) => {
      const p = seen.prob[w];
      const was = before ? before.prob[w] : null;
      const moved = was != null && Math.abs(p - was) >= 0.01;
      return el(
        pick ? "button" : "div",
        {
          class: `wrow${w === ui.word ? " on" : ""}`,
          type: pick ? "button" : null,
          "data-word": engine.vocab[w],
          onclick: pick ? () => explain(w) : null,
        },
        el("span", { class: "wname" }, engine.vocab[w]),
        el(
          "span",
          { class: "wbar" },
          moved ? el("span", { class: "was", style: `width:${Math.max(1, was * 100)}%` }) : null,
          el("span", { class: "now", style: `width:${Math.max(1, p * 100)}%` }),
        ),
        el("span", { class: "wpct" }, moved ? `${pct(was)} → ${pct(p)}` : pct(p)),
      );
    }),
  );
}

function wordPicker() {
  const list = el("datalist", { id: "vocab" }, engine.vocab.map((w) => el("option", { value: w })));
  const input = el("input", { type: "search", placeholder: "Or type a word: cat, cup, robot arm…", list: "vocab", autocomplete: "off", "aria-label": "Pick a word" });
  input.addEventListener("change", () => {
    const w = engine.vocab.indexOf(input.value.trim().toLowerCase());
    if (w >= 0) explain(w);
    else if (input.value.trim()) flash(`“${input.value.trim()}” is not on the list of ${engine.vocab.length} words yet.`);
  });
  return el("div", { class: "find" }, input, list);
}

function mainView() {
  const word = engine.vocab[ui.word];
  const seen = ui.state.seen;
  const now = ui.result || seen;
  const list = reasons();
  const total = list.reduce((a, r) => a + r.score, 0) || 1;
  const allOff = list.length && list.every((r) => ui.off.has(r.id));
  const waiting = !ui.live && ui.off.size && !ui.result;
  return el(
    "div",
    { class: "view" },
    el("h2", {}, ui.result ? "Now it sees" : "It sees"),
    resultCard(word),
    wordList(now, ui.result ? seen : null),
    wordPicker(),
    el("h2", { class: "why-head" }, `Why it sees “${word}”`),
    !ui.why
      ? el("p", { class: "muted" }, "It works this out as soon as the model is ready.")
      : list.length
      ? el(
          "p",
          { class: "muted" },
          `These features push it towards “${word}” the most. Switch them off and watch the words.`,
        )
      : el("p", { class: "muted" }, `Nothing on this picture pushes it towards “${word}”. Try another word or picture.`),
    el(
      "div",
      { class: "flist" },
      list.map((r) => {
        const off = ui.off.has(r.id);
        return el(
          "div",
          {
            class: `frow why${off ? " off" : ""}`,
            "data-feature": r.id,
            onmouseenter: () => {
              ui.hover = r.id;
              drawHeat();
            },
            onmouseleave: () => {
              ui.hover = null;
              drawHeat();
            },
          },
          el(
            "button",
            { class: "switch", type: "button", role: "switch", "aria-checked": off ? "false" : "true", "aria-label": `${fname(r.id)} ${off ? "off" : "on"}`, onclick: () => toggle(r.id) },
            el("span", {}),
          ),
          el(
            "button",
            { class: "fname-btn", type: "button", onclick: () => openFeature(r.id) },
            thumbOf(r.id),
            el("span", { class: engine.features[r.id]?.name ? "fname" : "fname unnamed" }, engine.features[r.id]?.name || String(r.id)),
          ),
          el("span", { class: "fbar" }, el("span", { style: `width:${Math.max(4, Math.round((r.score / total) * 100))}%` })),
        );
      }),
    ),
    list.length
      ? el(
          "p",
          { class: "row-actions" },
          el("button", { class: "pill-btn", type: "button", "data-testid": "all-off", onclick: () => setAll(allOff) }, allOff ? "Switch them back on" : `Switch all ${list.length} off`),
          el("button", { class: "pill-btn", type: "button", onclick: copyLink }, "Copy a link to this"),
        )
      : null,
    waiting ? el("p", { class: "muted small" }, "Switching them off as soon as the model is ready.") : null,
  );
}

function resultCard(word) {
  if (!ui.result) return null;
  const w = ui.word;
  const before = ui.state.seen.prob[w];
  const after = ui.result.prob[w];
  const n = ui.off.size;
  const control = ui.control;
  return el(
    "div",
    { class: "result", "data-testid": "result" },
    el("p", { class: "lead-num" }, `“${word}”: ${pct(before)} → ${pct(after)}`),
    el(
      "p",
      { class: "muted small" },
      `With ${n} feature${n === 1 ? "" : "s"} off. `,
      control
        ? `Switching off ${control.ids.length} random feature${control.ids.length === 1 ? "" : "s"} that also fire here instead leaves it at ${pct(control.seen.prob[w])}.`
        : "",
    ),
  );
}

function featureView() {
  const id = ui.feature;
  const f = engine.features[id];
  const map = engine.featureMap(ui.state, id);
  const fired = map.reduce((a, v) => a + (v > 0 ? 1 : 0), 0);
  // Only words it pushed up on most of the pictures it was written onto: measured, not a name.
  const sees = (f.sees || []).filter((r) => r.rise >= 0.05 && r.top_on * 2 > r.of);
  const modes = el(
    "div",
    { class: "seg", role: "tablist" },
    [
      ["scale", "Turn it up"],
      ["clamp", "Write it everywhere"],
    ].map(([mode, name]) =>
      el(
        "button",
        {
          type: "button",
          role: "tab",
          class: mode === ui.mode ? "on" : "",
          "aria-selected": mode === ui.mode ? "true" : "false",
          onclick: () => {
            ui.mode = mode;
            renderPanel();
            runDetail();
          },
        },
        name,
      ),
    ),
  );
  const control =
    ui.mode === "scale"
      ? slider("Strength", 0, 8, 0.5, ui.strength.scale, (v) => `× ${v}`, (v) => (ui.strength.scale = v))
      : slider("Strength", 0, 4, 0.25, ui.strength.clamp, (v) => `${v} × its peak`, (v) => (ui.strength.clamp = v));
  return el(
    "div",
    { class: "view" },
    el(
      "button",
      {
        class: "back",
        type: "button",
        onclick: () => {
          ui.feature = null;
          ui.detail = null;
          render();
        },
      },
      `← Why it sees “${engine.vocab[ui.word]}”`,
    ),
    el("h2", { class: "ftitle" }, fname(id)),
    el(
      "p",
      { class: "muted" },
      fired
        ? `It fires on ${fired} of ${engine.nTok} patches of this picture.`
        : "It is silent on this picture, so turning it up does nothing here. Write it everywhere instead.",
      f.name
        ? ` Written onto every patch of ${engine.pack.names?.pictures ?? "a few"} pictures, it makes the model see “${f.name}” more than other features do.`
        : sees.length
        ? ` Written onto every patch of ${sees[0].of} pictures it had not trained on, it made the model see ${sees.map((r) => `“${r.word}”`).join(", ")} more.`
        : "",
    ),
    el("h3", {}, "Change it and watch the words"),
    modes,
    control,
    ui.detail ? wordList(ui.detail, ui.state.seen, { pick: false }) : el("p", { class: "muted" }, "Running the model…"),
    el("h3", {}, "Where else it fires"),
    el("div", { class: "gallery", id: "gallery" }, galleryFor(id)),
  );
}

function slider(label, min, max, step, value, fmt, set) {
  const out = el("output", {}, fmt(value));
  const input = el("input", { type: "range", min, max, step, value, "aria-label": label });
  let queued = false;
  input.addEventListener("input", async () => {
    const v = Number(input.value);
    set(v);
    out.textContent = fmt(v);
    if (ui.busy) {
      queued = true;
      return;
    }
    do {
      queued = false;
      await runDetail();
    } while (queued);
  });
  return el("label", { class: "slider" }, el("span", {}, label), input, out);
}

// ------------------------------------------------------------------ gallery (in the background)

async function encodeGallery() {
  for (const sample of engine.pack.samples) {
    while (ui.busy || ui.cam) await new Promise((r) => setTimeout(r, 150));
    const img = await loadImage(engine.base + sample.file);
    const state = await engine.encode(img, { readout: false });
    ui.gallery.push({ sample, img, rows: state.rows, peaks: state.peaks });
    const box = $("gallery");
    if (box && ui.feature != null) box.replaceChildren(...[galleryFor(ui.feature)].flat());
  }
}

function galleryFor(id) {
  if (!ui.gallery.length) return el("p", { class: "muted" }, "Looking through the example pictures…");
  const f = engine.features[id];
  const ranked = ui.gallery.filter((g) => g.peaks[id] > 0).sort((a, b) => b.peaks[id] - a.peaks[id]).slice(0, 6);
  if (!ranked.length) {
    const done = ui.gallery.length === engine.pack.samples.length;
    return el("p", { class: "muted" }, done ? "It does not fire on any of the example pictures. Try your own." : "Not on the pictures checked so far…");
  }
  return ranked.map((g) => {
    const canvas = el("canvas", { width: 140, height: 140 });
    const ctx = canvas.getContext("2d");
    const w = g.img.naturalWidth;
    const h = g.img.naturalHeight;
    const side = Math.min(w, h);
    ctx.drawImage(g.img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 140, 140);
    ctx.fillStyle = "rgba(20,20,19,0.35)";
    ctx.fillRect(0, 0, 140, 140);
    const W = engine.gridW;
    const small = new ImageData(W, engine.gridH);
    g.rows.forEach((row, t) => {
      const j = row.idx.indexOf(id);
      const a = j >= 0 ? Math.min(1, row.val[j] / (f.max || 1)) : 0;
      small.data.set([EMBER[0], EMBER[1], EMBER[2], Math.round(Math.pow(a, 0.7) * 0.9 * 255)], t * 4);
    });
    const tmp = document.createElement("canvas");
    tmp.width = W;
    tmp.height = engine.gridH;
    tmp.getContext("2d").putImageData(small, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, 140, 140);
    return el("button", { class: "gthumb", type: "button", title: g.sample.credit || g.sample.clip_id, onclick: () => showSample(g.sample) }, canvas);
  });
}

// ------------------------------------------------------------------ inputs

function wireInputs() {
  $("file").addEventListener("change", (e) => {
    addFile(e.target.files?.[0]);
    e.target.value = "";
  });
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
    addFile(e.dataTransfer.files?.[0]);
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) addFile(item.getAsFile());
  });
  $("cam-btn").addEventListener("click", () => (ui.cam ? snapCam() : startCam()));
  new ResizeObserver(() => drawHeat()).observe($("stage"));
}

async function addFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return flash("That is not a picture this browser can read.");
  try {
    const img = await loadImage(URL.createObjectURL(file));
    stopCam();
    ui.picture = { img };
    markThumb(null);
    $("credit").textContent = "";
    await showPicture(img);
  } catch (err) {
    flash(err.message);
  }
}

async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    const video = $("cam");
    video.srcObject = stream;
    await video.play();
    ui.cam = stream;
    video.hidden = false;
    $("photo").hidden = true;
    paintMap(null);
    $("cam-btn").textContent = "Take the picture";
  } catch {
    flash("The camera is not available here. You can still use a picture.");
  }
}

async function snapCam() {
  const video = $("cam");
  const canvas = el("canvas", { width: video.videoWidth, height: video.videoHeight });
  canvas.getContext("2d").drawImage(video, 0, 0);
  stopCam();
  const img = await loadImage(canvas.toDataURL("image/jpeg", 0.9));
  ui.picture = { img };
  markThumb(null);
  await showPicture(img);
}

function stopCam() {
  if (ui.cam) ui.cam.getTracks().forEach((t) => t.stop());
  ui.cam = null;
  $("cam").hidden = true;
  $("cam-btn").textContent = "Use your camera";
}

// ------------------------------------------------------------------ links

// #p=<example>&w=<word>&off=1,2,3 (or #img=<jpeg> for your own picture), &f=<feature> for one feature.
function linkParams({ withPicture = false } = {}) {
  const q = new URLSearchParams();
  if (ui.picture?.sample) q.set("p", ui.picture.sample);
  else if (withPicture && ui.picture?.img) q.set("img", pictureCode(ui.picture.img));
  if (ui.word != null) q.set("w", engine.vocab[ui.word]);
  if (ui.off.size) q.set("off", [...ui.off].join(","));
  if (ui.feature != null) q.set("f", String(ui.feature));
  return q;
}

function syncAddress() {
  const hash = linkParams().toString();
  if ((hash ? `#${hash}` : "") !== location.hash) history.replaceState(null, "", hash ? `#${hash}` : location.pathname + location.search);
}

function pictureCode(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const side = Math.min(w, h);
  const c = el("canvas", { width: 224, height: 224 });
  c.getContext("2d").drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 224, 224);
  return c.toDataURL("image/jpeg", 0.85).split(",")[1].replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function copyLink() {
  const url = `${location.origin}${location.pathname}${location.search}#${linkParams({ withPicture: true })}`;
  try {
    await navigator.clipboard.writeText(url);
    flash("Link copied. It opens this picture with the same features off.");
  } catch {
    window.prompt("Copy this link:", url);
  }
}

async function restoreFromLink() {
  const q = new URLSearchParams(location.hash.slice(1));
  const sample = engine.pack.samples.find((s) => s.clip_id === q.get("p"));
  const word = engine.vocab.indexOf(q.get("w") || "");
  const off = (q.get("off") || "").split(",").filter(Boolean).map(Number).filter((n) => Number.isInteger(n) && n >= 0 && n < engine.nFeatures);
  try {
    if (sample) await showSample(sample, { word: word >= 0 ? word : null, off });
    else if (q.get("img")) {
      const img = await loadImage(`data:image/jpeg;base64,${q.get("img").replace(/-/g, "+").replace(/_/g, "/")}`);
      ui.picture = { img };
      await showPicture(img, { word: word >= 0 ? word : null, off });
    } else return false;
  } catch {
    flash("That link's picture could not be opened.");
    return false;
  }
  const f = Number(q.get("f"));
  if (q.has("f") && Number.isInteger(f) && f >= 0 && f < engine.nFeatures) await openFeature(f);
  return true;
}

// ------------------------------------------------------------------ footer

function renderPerf() {
  if (!ui.live) return;
  const t = ui.timings;
  const parts = [engine.backend === "webgpu" ? "Running on WebGPU" : `Running on WebAssembly (${self.crossOriginIsolated ? "multi-threaded" : "one thread"})`, `ready in ${fmtMs(engine.timings.ready)}`];
  if (t.read != null) parts.push(`reading a picture ${fmtMs(t.read)}`);
  if (t.why != null) parts.push(`finding why ${fmtMs(t.why)}`);
  if (t.edit != null) parts.push(`one edit ${fmtMs(t.edit)}`);
  $("perf").textContent = parts.join(" · ") + ".";
}

function flash(message) {
  const box = el("div", { class: "toast", role: "status" }, message);
  document.body.append(box);
  setTimeout(() => box.remove(), 4000);
}

boot();
