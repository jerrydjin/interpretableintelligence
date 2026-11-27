// ii playground: what lights up inside. Pick a picture; the page shows the switches (SAE features)
// that light up on it most, which are on at any spot you point at, where each lights up on
// thousands of other photos, and what turning one up does beside a random switch. Typing a word
// finds the switch that makes it see that word (find the switch). Runs a version 2 pack with the
// words readout (ii export-live --readout words) through engine.js, in the browser. Exploration
// only: nothing here can change a feature's badge.

import { Engine } from "./engine.js";

const PARAMS = new URLSearchParams(location.search);
const PACK = PARAMS.get("pack") || "packs/clip-b16-words/";
const EMBER = [232, 88, 42];
const FIRST = "01-coco-14475"; // the picture the page opens on (a teddy bear: its snout and eyes light up switches for snouts and eyes)
const TARGET = "umbrella"; // the word it opens on (packs without it use their last word)
const SUGGEST = ["snow", "umbrella", "cat", "pizza", "guitar"]; // one tap each, where on the list
const BAKE = PARAMS.has("bake");
const N_WORDS = 6; // words shown
const N_WHY = 6; // features shown for a word
const N_ON = 3; // features offered to turn on
const TRY = 5; // switches tried for a word, and random switches tried as the control
const THUMB = 44; // px, the crop of the picture shown beside each feature
const PUSHES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]; // the Strength slider's steps (0 is off)
const N_EXAMPLES = 4; // other pictures shown for a switch beside the answer
const N_TOP = 6; // photos shown on a switch's own card
const N_ACTIVE = 8; // switches listed for the picture (and as many more on request)
const N_SPOT = 8; // switches listed for one spot
const N_BAKED = 30; // switches baked per example picture, with where they light up
const PREVIEW = 36; // px, the crops of other photos beside each switch in the list
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
  strength: { add: 1 }, // the one switch's own strength (featureView)
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
  toward: null, // engine.toward() for the word, when it is not the top word: features to turn on
  on: new Set(), // features turned on (written onto every patch)
  push: 0.5, // how strongly, in units of each feature's training peak
  boost: null, // words with them on
  boostControl: null, // words with as many random features turned on the same way
  curve: null, // the word's probability at every strength, with the switch and with the random one
  curveBusy: false,
  view: "inside", // inside: what lights up on the picture | word: make it see a word, or why it sees one
  spot: null, // the patch pointed at (inside view)
  pinned: false, // the spot stays after the pointer leaves (a tap)
  more: false, // the longer list of switches
  firsts: null, // gallery/firsts.json: a few crops of other photos per switch
};
let curveToken = 0;
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
    loadFirsts();
    if (ui.instant) {
      showMeta();
      $("work").hidden = false;
      buildThumbs();
      wireInputs();
      if (!(await restoreFromLink())) await showFirst();
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
    if (!(await restoreFromLink())) await showFirst();
  }
  if (BAKE) return bake();
  if (!engine.pack.files.gallery) encodeGallery();
}

function showMeta() {
  $("n-features").textContent = engine.pack.sae.n_features.toLocaleString("en-GB");
}

// The word the page opens on: TARGET where the pack's list has it, else its last word.
function targetWord() {
  const w = engine.vocab.indexOf(TARGET);
  return w >= 0 ? w : engine.vocab.length - 1;
}

function showFirst() {
  const pack = engine.pack;
  return showSample(pack.samples.find((s) => s.clip_id === FIRST) || pack.samples[0], { word: null });
}

// The model is ready: work out for real what the page has shown so far (the baked answer, or a
// picture, word or switch chosen while it loaded), keeping the same picture, word and switches.
async function goLive() {
  const feature = ui.pendingFeature;
  ui.pendingFeature = null;
  ui.baked = null;
  if (!ui.cam && ui.picture) {
    const img = ui.picture.sample ? await loadImage(engine.base + sampleOf(ui.picture.sample).file) : ui.picture.img;
    await showPicture(img, { word: ui.view === "word" ? ui.word : null, off: [...ui.off], on: [...ui.on] });
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
function showBaked(key, { word = null, off = [], on = [] } = {}) {
  const rec = ui.instant?.samples?.[key];
  if (word == null) return rec?.active ? showBakedInside(rec) : false;
  if (rec?.flip && word === rec.flip.word && !off.length) return showBakedFlip(rec);
  if (!rec || (word != null && word !== rec.word)) return false;
  ui.baked = rec;
  ui.state = { baked: true, seen: probs(rec.prob), maps: new Map(Object.entries(rec.maps).map(([id, [b64, peak]]) => [Number(id), unq8(b64, peak)])) };
  ui.view = "word";
  ui.feature = null;
  ui.detail = null;
  ui.word = rec.word;
  ui.off = new Set(off);
  ui.on = new Set(on);
  bakedWhy();
  bakedOff();
  render();
  return true;
}

// The baked switch for the word the page opens on: before, with the switch, with a random one.
function showBakedFlip(rec) {
  const flip = rec.flip;
  ui.baked = null;
  ui.state = { baked: true, seen: probs(flip.prob), maps: new Map() };
  Object.assign(ui, { view: "word", feature: null, detail: null, why: null, result: null, control: null, word: flip.word, off: new Set(), on: new Set(flip.on), push: flip.push });
  ui.toward = { ranked: flip.toward.map(([id, score]) => ({ id, score })) };
  ui.tried = null;
  ui.boost = probs(flip.boost);
  ui.boostControl = { ids: flip.control.ids, seen: probs(flip.control.prob), of: flip.control.of };
  ui.state.maps = new Map(Object.entries(flip.maps || {}).map(([id, [b64, peak]]) => [Number(id), unq8(b64, peak)]));
  ui.curve = flip.curve
    ? {
        key: curveKey(),
        full: true,
        on: new Map(flip.curve.on.map(([s, p]) => [s, probs(p)])),
        ctl: new Map(flip.curve.ctl.map(([s, p]) => [s, probs(p)])),
      }
    : null;
  render();
  return true;
}

// The baked switches that light up most on an example picture, with where each lights up.
function showBakedInside(rec) {
  ui.baked = rec;
  ui.state = { baked: true, seen: probs(rec.prob), maps: new Map() };
  for (const [id, [b64, peak]] of Object.entries(rec.active.maps)) ui.state.maps.set(Number(id), unq8(b64, peak));
  for (const [id, [b64, peak]] of Object.entries(rec.maps)) ui.state.maps.set(Number(id), unq8(b64, peak));
  ui.active = rec.active.ids.map(([id, score]) => ({ id, score }));
  toInside();
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
  const target = targetWord();
  await encodeGallery();
  const shown = new Set(); // switches offered on some picture: their examples go in the file
  for (const sample of engine.pack.samples) {
    ui.word = null;
    await showSample(sample, { word: null }); // what lights up inside, first
    const active = activeList(N_BAKED);
    await explain(topWord(ui.state.seen)); // then why it sees its own top word
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
    // The page's opening move on this picture: find the switch for the target word.
    if (target !== word) {
      await explain(target);
      if (ui.boost) {
        await computeCurve();
        const keep = new Set([target]);
        const top = (prob) => [...prob.keys()].sort((a, b) => prob[b] - prob[a]).slice(0, 2);
        const steps = [...ui.curve.on.values(), ...ui.curve.ctl.values()];
        [seen, ui.boost, ui.boostControl.seen, ...steps].forEach((p) => top(p.prob).forEach((w) => keep.add(w)));
        const sparse = (prob) => Object.fromEntries([...keep].map((w) => [w, +prob[w].toPrecision(4)]));
        const offered = boosters();
        offered.forEach((r) => shown.add(r.id));
        rec.flip = {
          word: target,
          toward: offered.map((r) => [r.id, +r.score.toPrecision(5)]),
          maps: Object.fromEntries(offered.map((r) => [r.id, q8(engine.featureMap(ui.state, r.id))])),
          curve: {
            on: PUSHES.map((s) => [s, sparse(ui.curve.on.get(s).prob)]),
            ctl: PUSHES.map((s) => [s, sparse(ui.curve.ctl.get(s).prob)]),
          },
          on: [...ui.on],
          push: ui.push,
          prob: sparse(seen.prob),
          boost: sparse(ui.boost.prob),
          control: { ids: ui.boostControl.ids, of: ui.boostControl.of, prob: sparse(ui.boostControl.seen.prob) },
        };
      }
    }
    rec.active = {
      ids: active.map((r) => [r.id, +r.score.toPrecision(4)]),
      maps: Object.fromEntries(active.filter((r) => !(r.id in maps)).map((r) => [r.id, q8(engine.featureMap(ui.state, r.id))])),
    };
    out.samples[sample.clip_id] = rec;
  }
  // Where each offered switch lights up most on the example pictures, so the page shows it at once.
  out.examples = Object.fromEntries(
    [...shown].map((id) => [id, examplesFor(id).map((g) => [g.sample.clip_id, ...q8(engine.featureMap(g, id))])]),
  );
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
  opts = { ...opts, word: opts.word ?? ui.word }; // a new picture keeps the word
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

async function showPicture(img, { word = null, off = [], on = [] } = {}) {
  showPhoto(img);
  if (!ui.live) {
    // Read for real once the model is ready (goLive), with this word and these switches.
    Object.assign(ui, { state: null, why: null, result: null, control: null, baked: null, feature: null, word, off: new Set(off), on: new Set(on), view: word == null ? "inside" : "word" });
    return render();
  }
  await busy(async () => {
    const t0 = performance.now();
    ui.state = await engine.encode(img);
    ui.timings.read = performance.now() - t0;
  });
  ui.feature = null;
  ui.detail = null;
  if (ui.wantWord) {
    // A link to a word off the list: read it first.
    const typed = ui.wantWord;
    ui.wantWord = null;
    try {
      word = await busy(() => engine.addWord(typed));
      ui.state.seen = engine.wordsFrom(ui.state.seen.embed);
    } catch {
      word = null;
    }
  }
  if (word == null) {
    ui.active = activeList(N_BAKED);
    toInside();
    return render();
  }
  await explain(word, off, on);
}

// ------------------------------------------------------------------ what lights up inside

// Back to what lights up on the picture: no word, nothing turned on or off.
function toInside() {
  cancelCurve();
  Object.assign(ui, { view: "inside", word: null, why: null, toward: null, tried: null, boost: null, boostControl: null, result: null, control: null, feature: null, detail: null, hover: null, spot: null, pinned: false, more: false });
  ui.off = new Set();
  ui.on = new Set();
}

function backToInside() {
  if (!ui.live && ui.picture?.sample && ui.instant?.samples?.[ui.picture.sample]?.active) return showBakedInside(ui.instant.samples[ui.picture.sample]);
  if (ui.state && !ui.state.baked) ui.active = activeList(N_BAKED);
  toInside();
  render();
}

// The switches that light up most on the picture: how much of it they cover at their usual top
// strength (activation summed over the patches, over the feature's training peak). Always-on
// switches (ui.background) are left out: they carry the whole picture, not what is in it.
function activeList(n) {
  const s = ui.state;
  if (!s || s.baked) return ui.active || [];
  const out = [];
  for (let f = 0; f < s.totals.length; f++) {
    if (s.totals[f] <= 0 || ui.background.has(f)) continue;
    out.push({ id: f, score: s.totals[f] / (engine.features[f]?.max || 1) });
  }
  return out.sort((a, b) => b.score - a.score).slice(0, n);
}

// The switches on at one patch, strongest first, as a share of each one's usual top.
function spotList(t) {
  const s = ui.state;
  const out = [];
  if (s.baked) {
    for (const [id, map] of s.maps) if (map[t] > 0) out.push({ id, score: map[t] / (engine.features[id]?.max || 1) });
  } else {
    const row = s.rows[t];
    for (let j = 0; j < row.idx.length; j++) {
      const id = row.idx[j];
      if (row.val[j] > 0 && !ui.background.has(id)) out.push({ id, score: row.val[j] / (engine.features[id]?.max || 1) });
    }
  }
  const seen = new Set();
  return out.filter((r) => !seen.has(r.id) && seen.add(r.id)).sort((a, b) => b.score - a.score).slice(0, N_SPOT);
}

// The patch under a pointer on the picture.
function patchAt(e) {
  const r = $("stage").getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * engine.gridW);
  const y = Math.floor(((e.clientY - r.top) / r.height) * engine.gridH);
  if (x < 0 || y < 0 || x >= engine.gridW || y >= engine.gridH) return null;
  return y * engine.gridW + x;
}

function setSpot(t, pinned = ui.pinned) {
  if (ui.spot === t && ui.pinned === pinned) return;
  ui.spot = t;
  ui.pinned = t != null && pinned;
  const box = $("active");
  if (box) box.replaceWith(activeView());
  drawHeat();
}

function wireStage() {
  const stage = $("stage");
  const inside = () => ui.view === "inside" && ui.feature == null && ui.state && !ui.cam;
  stage.addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" || ui.pinned || !inside()) return;
    setSpot(patchAt(e), false);
  });
  stage.addEventListener("pointerleave", (e) => {
    if (e.pointerType === "mouse" && !ui.pinned && inside()) setSpot(null, false);
  });
  stage.addEventListener("click", (e) => {
    if (!inside()) return;
    const t = patchAt(e);
    if (ui.pinned && ui.spot === t) setSpot(null, false);
    else setSpot(t, true);
  });
}

// A few photos from the gallery where a switch lights up most, cropped to where it does.
async function loadFirsts() {
  try {
    const file = engine.pack.files?.gallery;
    if (!file) return;
    const dir = engine.base + file.replace(/[^/]*$/, "");
    const index = await fetch(engine.base + file).then((r) => r.json());
    if (!index.firsts) return;
    ui.firsts = await fetch(dir + index.firsts).then((r) => r.json());
    ui.firsts.read = index.read;
    const box = $("active");
    if (box) box.replaceWith(activeView());
  } catch {
    ui.firsts = null;
  }
}

const pictureCache = new Map();
const picture = (src) => {
  if (!pictureCache.has(src)) pictureCache.set(src, loadImage(src).catch(() => null));
  return pictureCache.get(src);
};

function previewsOf(id) {
  const list = ui.firsts?.top?.[id];
  if (!list) return null;
  const dpr = window.devicePixelRatio || 1;
  return el(
    "span",
    { class: "previews", "aria-hidden": "true" },
    list.map(([pic, x, y, side]) => {
      const c = el("canvas", { width: PREVIEW * dpr, height: PREVIEW * dpr });
      picture(engine.base + ui.firsts.pictures[pic]).then((img) => {
        if (!img) return;
        const w = img.naturalWidth, h = img.naturalHeight, sq = Math.min(w, h);
        const px = sq / engine.gridW;
        c.getContext("2d").drawImage(img, (w - sq) / 2 + x * px, (h - sq) / 2 + y * px, side * px, side * px, 0, 0, c.width, c.height);
      });
      return c;
    }),
  );
}

// What lights up on the picture, or at the spot pointed at: each switch with where it is on this
// picture, where it lights up on other photos, and how strongly.
function activeView() {
  const spot = ui.spot != null;
  const list = spot ? spotList(ui.spot) : ui.active.slice(0, ui.more ? 3 * N_ACTIVE : N_ACTIVE);
  const top = spot ? 1 : list[0]?.score || 1;
  const named = fitsHere;
  const rows = list.map((r) =>
    el(
      "button",
      {
        class: "frow lit",
        type: "button",
        "data-feature": r.id,
        onclick: () => openFeature(r.id),
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
        "span",
        { class: "lit-mid" },
        el("span", { class: named(r.id) ? "fname" : "fname unnamed" }, named(r.id) ? engine.features[r.id].name : `#${r.id}`),
        el("span", { class: "fbar" }, el("span", { style: `width:${Math.max(4, Math.min(100, Math.round((r.score / top) * 100)))}%` })),
      ),
      previewsOf(r.id),
    ),
  );
  return el(
    "div",
    { id: "active", "data-testid": "active" },
    el(
      "div",
      { class: "lit-head" },
      el("h2", { class: "lit-title" }, spot ? "Lit up at this spot" : "Lit up inside"),
      spot && ui.pinned ? el("button", { class: "back", type: "button", onclick: () => setSpot(null, false) }, "Whole picture") : null,
    ),
    el("p", { class: "muted small lit-note" }, spot ? "How strongly each is on here, against the most it ever is." : `The switches most on in this picture. ${matchMedia("(hover: hover)").matches ? "Point at" : "Tap"} any part of it to see what is on there.`),
    list.length ? el("div", { class: "flist" }, rows) : el("p", { class: "muted" }, "Nothing stands out here."),
    !spot && !ui.more && ui.active.length > N_ACTIVE
      ? el("button", { class: "back more", type: "button", onclick: () => ((ui.more = true), $("active").replaceWith(activeView())) }, "Show more")
      : null,
  );
}

// A switch's name is the word turning it up makes the model say on other pictures. Beside a
// picture it is named only when that word is among the 20 it might call this picture, so a
// "camera" switch on a teddy bear's snout does not read as a camera in the picture.
function fitsHere(id) {
  const name = engine.features[id]?.name;
  if (!name || !ui.state) return false;
  const seen = ui.state.seen;
  return [...seen.prob.keys()].sort((a, b) => seen.prob[b] - seen.prob[a]).slice(0, 20).some((w) => engine.vocab[w] === name);
}

const litName = (id) => switchName(id);

// What it calls the picture, in a line: tap a word to see which switches make it say that.
function seenLine() {
  const seen = ui.state.seen;
  const order = [...seen.prob.keys()].sort((a, b) => seen.prob[b] - seen.prob[a]).slice(0, 3);
  return el(
    "p",
    { class: "seen-line", "data-testid": "seen" },
    el("span", { class: "muted" }, "It calls this "),
    order.flatMap((w, i) => [i ? el("span", { class: "muted" }, i === order.length - 1 ? " or " : ", ") : null, el("button", { class: "seen-word", type: "button", "data-word": engine.vocab[w], onclick: () => openWord(w) }, engine.vocab[w], el("small", {}, ` ${pct(seen.prob[w])}`))]),
  );
}

function openWord(w) {
  if (!ui.live && ui.picture?.sample && showBaked(ui.picture.sample, { word: w })) return;
  explain(w);
}

function insideView() {
  return el("div", { class: "view" }, seenLine(), activeView(), el("div", { class: "see-more section" }, seeInput()));
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

async function explain(word, off = [], on = []) {
  ui.view = "word";
  ui.spot = null;
  ui.pinned = false;
  ui.feature = null;
  ui.word = word;
  ui.off = new Set(off);
  ui.on = new Set(on);
  ui.result = null;
  ui.control = null;
  ui.toward = ui.boost = ui.boostControl = ui.tried = null;
  cancelCurve();
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
  // A word it does not mostly see: also find what would make it see it.
  if (word !== topWord(ui.state.seen)) {
    ui.toward = await busy(() => engine.toward(ui.state, word, ui.background));
    // A word it barely sees needs a stronger push (measured on the example pictures).
    if (!on.length) ui.push = ui.state.seen.prob[word] < 0.01 ? 1 : 0.5;
  }
  if (ui.off.size) await runOff();
  if (ui.toward && !on.length && !off.length) await findSwitch();
  else if (ui.on.size && ui.toward) await runOn();
  render();
}

// The switches offered for the word: measured (findSwitch) where they were tried, else as ranked.
// Switches that are on come first, even when ranked lower (a link, or the baked answer gone live).
function boosters() {
  const ranked = ui.tried || (ui.toward ? ui.toward.ranked.filter((r) => r.score > 0) : []);
  if (!ranked.length) return [];
  const top = ranked[0].score;
  const on = [...ui.on].map((id) => ranked.find((r) => r.id === id) || { id, score: top });
  return [...on, ...ranked.filter((r) => !ui.on.has(r.id))].slice(0, Math.max(N_ON, on.length));
}

const pushEdits = (ids, push = ui.push) => ids.map((id) => ({ id, kind: "add", value: push * (engine.features[id].max || 0) }));

// Each of `ids` turned on alone; the one that makes it see the word most, and the rest by that.
async function tryEach(ids) {
  const out = [];
  for (const id of ids) {
    const seen = await engine.see(ui.state, pushEdits([id]));
    out.push({ id, score: seen.prob[ui.word], seen });
  }
  return out.sort((a, b) => b.score - a.score);
}

// Random switches for the control: as many sets as tried switches, each the size of `n`, drawn
// from the features ranked far down for this word.
function randomSets(n) {
  const pool = ui.toward.ranked.slice(200).map((r) => r.id).filter((id) => !ui.on.has(id));
  return seeded(pool, n * TRY, hashOf(ui.picture, ui.word, "on", n)).reduce((sets, id, i) => (i % n ? sets[sets.length - 1].push(id) : sets.push([id]), sets), []);
}

// The best of the random sets: the control a picked switch has to beat.
async function bestRandom(n) {
  let best = null;
  for (const ids of randomSets(n)) {
    const seen = await engine.see(ui.state, pushEdits(ids));
    if (!ui.state) return null;
    if (!best || seen.prob[ui.word] > best.seen.prob[ui.word]) best = { ids, seen };
  }
  return best && { ...best, of: TRY };
}

// Find the switch: try the TRY features ranked first for the word one by one, turn on the one
// that makes it see the word most, and set it beside the best of TRY random switches.
async function findSwitch() {
  const cands = ui.toward.ranked.filter((r) => r.score > 0).slice(0, TRY).map((r) => r.id);
  if (!cands.length) return;
  await busy(async () => {
    ui.tried = await tryEach(cands);
    ui.on = new Set([ui.tried[0].id]);
    ui.boost = ui.tried[0].seen;
    ui.boostControl = await bestRandom(1);
  });
  syncAddress();
  computeCurve();
}

// Use another switch for the word instead (one switch at a time).
async function pickSwitch(id) {
  if (ui.on.size === 1 && ui.on.has(id)) return;
  ui.off = new Set();
  ui.result = ui.control = null;
  ui.on = new Set([id]);
  cancelCurve();
  if (!ui.live) {
    ui.boost = ui.boostControl = null;
    return render();
  }
  await runOn();
  render();
}

const curveKey = () => JSON.stringify([ui.picture?.sample || ui.picture?.gallery || "own", ui.word, [...ui.on], ui.boostControl?.ids || []]);

function cancelCurve() {
  curveToken++;
  ui.curve = null;
  ui.curveBusy = false;
}

// The word's probability at every step of the Strength slider, with the switch and with the same
// random switch, so moving the slider is instant and the chart shows the whole climb.
async function computeCurve() {
  if (!ui.live || !ui.on.size || !ui.boostControl?.ids.length) return;
  const key = curveKey();
  if (ui.curve?.key === key) return ui.curve.done;
  const token = ++curveToken;
  const ids = [...ui.on];
  const ctl = ui.boostControl.ids;
  const state = ui.state;
  const curve = { key, on: new Map(ui.push ? [[ui.push, ui.boost]] : []), ctl: new Map(ui.push ? [[ui.push, ui.boostControl.seen]] : []) };
  ui.curve = curve;
  ui.curveBusy = true;
  curve.done = (async () => {
    for (const s of PUSHES) {
      for (const [which, set] of [[curve.on, ids], [curve.ctl, ctl]]) {
        if (which.has(s)) continue;
        const seen = await engine.see(state, pushEdits(set, s));
        if (token !== curveToken) return;
        which.set(s, seen);
      }
      drawCurve();
    }
    curve.full = true;
    ui.curveBusy = false;
    drawCurve();
  })();
  return curve.done;
}

// The slider moved: read the answer off the curve where it has it, else work it out.
async function setPush(v) {
  ui.push = v;
  const c = ui.curve;
  if (v === 0) {
    ui.boost = ui.state.seen;
    ui.boostControl = ui.boostControl && { ...ui.boostControl, seen: ui.state.seen };
  } else if (c && c.on.has(v) && c.ctl.has(v)) {
    ui.boost = c.on.get(v);
    ui.boostControl = { ...ui.boostControl, seen: c.ctl.get(v) };
  } else if (ui.live && ui.on.size) await runOn({ keepControl: true });
  // Only the parts that change, so the slider stays under the finger.
  $("answer")?.replaceWith(answerCard());
  drawCurve();
  drawHeat();
  syncAddress();
}

// Write the chosen features onto every patch at `push` times their training peak, and as many
// random features the same way (from those ranked far down for this word), as the control.
async function runOn({ keepControl = false } = {}) {
  if (!ui.on.size) {
    ui.boost = ui.boostControl = null;
    return syncAddress();
  }
  const ids = [...ui.on];
  await busy(async () => {
    ui.boost = await engine.see(ui.state, pushEdits(ids));
    const ctl = keepControl && ui.boostControl?.ids.length === ids.length ? ui.boostControl : null;
    ui.boostControl = ctl ? { ...ctl, seen: await engine.see(ui.state, pushEdits(ctl.ids)) } : await bestRandom(ids.length);
  });
  syncAddress();
  computeCurve();
}

// A switch's title, everywhere on the page: what it fires on in the gallery photos (`looks`, ii
// look-live), else the word turning it up makes the model say when that word fits the
// picture (`name`, see fitsHere), else its number.
function title(id) {
  const f = engine.features[id] || {};
  const text = f.looks || (fitsHere(id) ? f.name : null);
  return text ? text[0].toUpperCase() + text.slice(1) : `Switch #${id}`;
}
const fname = title;
// The same inside a sentence: “dog face”, or #2290.
function switchName(id) {
  const t = title(id);
  return t.startsWith("Switch #") ? `#${id}` : `“${t.toLowerCase()}”`;
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
  if (ui.on.size) {
    ui.on = new Set();
    ui.boost = ui.boostControl = null;
    cancelCurve();
  }
  if (ui.off.has(id)) ui.off.delete(id);
  else ui.off.add(id);
  await runOff();
  render();
}

async function setAll(on) {
  ui.on = new Set();
  ui.boost = ui.boostControl = null;
  cancelCurve();
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
  ui.detail = ui.detailControl = null;
  ui.hover = null;
  ui.spot = null;
  ui.pinned = false;
  render();
  await runDetail();
}

async function runDetail() {
  const id = ui.feature;
  if (id == null) return;
  const add = ui.strength.add;
  const ctlId = randomSwitch(id);
  const [seen, ctl] = await busy(async () => [await engine.see(ui.state, pushEdits([id], add)), add && ctlId != null ? await engine.see(ui.state, pushEdits([ctlId], add)) : null]);
  if (ui.feature !== id) return;
  ui.detail = seen;
  ui.detailControl = ctl;
  const box = $("detail-words");
  if (box) box.replaceChildren(...[detailView()].flat());
  else renderPanel();
  syncAddress();
}

// A random switch for the control: any that is not always on, fixed for this switch and picture.
function randomSwitch(id) {
  const pool = engine.features.filter((f) => f.id !== id && (f.max || 0) > 0 && !ui.background.has(f.id)).map((f) => f.id);
  return seeded(pool, 1, hashOf(ui.picture?.sample || ui.picture?.gallery || "own", id, "detail"))[0] ?? null;
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

// A glow over the patches: `values` scaled by `scale` (default: their own peak), plus an even
// `wash` (0..1) over the whole picture for a switch turned up everywhere.
function paintMap(values, rgb, { scale = 0, wash = 0, canvas = $("heat") } = {}) {
  const g = canvas.id === "heat" ? sizeCanvas(canvas) : canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (canvas.id === "heat") $("stage").classList.toggle("mapping", Boolean(values));
  if (!values) return;
  let peak = scale;
  if (!peak) for (const v of values) if (v > peak) peak = v;
  const W = engine.gridW;
  const H = engine.gridH;
  const small = new ImageData(W, H);
  for (let t = 0; t < W * H; t++) {
    const a = Math.pow(Math.min(1, Math.max(0, values[t] / (peak || 1))), 0.7) * 0.9;
    small.data.set([rgb[0], rgb[1], rgb[2], Math.round((1 - (1 - a) * (1 - wash)) * 255)], t * 4);
  }
  const tmp = paintMap._tmp || (paintMap._tmp = document.createElement("canvas"));
  tmp.width = W;
  tmp.height = H;
  tmp.getContext("2d").putImageData(small, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

// A switch on the picture as a grid of lights, one per patch: how strongly it is on there.
// `values` are its activations, `scale` the value that lights a patch fully.
function paintLights(values, scale, canvas = $("heat")) {
  const g = canvas.id === "heat" ? sizeCanvas(canvas) : canvas.getContext("2d");
  if (canvas.id === "heat") $("stage").classList.add("mapping");
  g.clearRect(0, 0, canvas.width, canvas.height);
  const W = engine.gridW;
  const H = engine.gridH;
  const cw = canvas.width / W;
  const ch = canvas.height / H;
  const r0 = Math.min(cw, ch) * 0.5;
  g.fillStyle = "rgba(20,20,19,0.18)";
  g.fillRect(0, 0, canvas.width, canvas.height);
  for (let t = 0; t < W * H; t++) {
    const a = Math.min(1, Math.max(0, values[t] / (scale || 1)));
    if (a < 0.02) continue;
    const cx = (t % W + 0.5) * cw;
    const cy = (Math.floor(t / W) + 0.5) * ch;
    const r = r0 * (0.25 + 0.95 * Math.sqrt(a));
    const glow = g.createRadialGradient(cx, cy, 0, cx, cy, r);
    glow.addColorStop(0, `rgba(255,214,170,${0.35 + 0.6 * a})`);
    glow.addColorStop(0.35, `rgba(${EMBER.join(",")},${0.3 + 0.6 * a})`);
    glow.addColorStop(1, `rgba(${EMBER.join(",")},0)`);
    g.fillStyle = glow;
    g.beginPath();
    g.arc(cx, cy, r, 0, Math.PI * 2);
    g.fill();
  }
}

function drawHeat() {
  const key = $("stage-key");
  if (!ui.state) {
    key.replaceChildren();
    return paintMap(null);
  }
  const id = ui.hover ?? ui.feature;
  const on = ui.toward && ui.on.size === 1 ? [...ui.on][0] : null;
  if (id != null && (ui.feature != null || ui.view !== "inside")) {
    // One switch: where it lights up on this picture by itself, against its usual top.
    paintLights(engine.featureMap(ui.state, id), engine.features[id]?.max || 1);
    key.replaceChildren(el("span", { class: "swatch ember" }), `Where ${ui.toward && ui.feature == null ? switchName(id) : litName(id)} lights up on this picture`);
  } else if (on != null) {
    // The switch on this picture: where it already lights up, plus the even glow of turning it
    // up everywhere, brighter as the strength goes up.
    const peak = engine.features[on]?.max || 1;
    const lit = engine.featureMap(ui.state, on).map((v) => v + ui.push * peak);
    paintLights(lit, 3 * peak);
    key.replaceChildren(el("span", { class: "swatch ember" }), ui.push ? `${switchName(on)}, turned up ×${ui.push} everywhere` : `Where ${switchName(on)} lights up by itself`);
  } else if (ui.view === "inside") {
    // The switch pointed at in the list, else (with no spot) the first one, so the picture is lit.
    const lit = ui.hover ?? (ui.spot == null ? ui.active?.[0]?.id : null);
    document.querySelectorAll(".frow.lit").forEach((r) => r.classList.toggle("on", Number(r.dataset.feature) === lit));
    if (ui.spot != null) paintSpot(ui.spot, ui.hover);
    else if (lit != null) paintLights(engine.featureMap(ui.state, lit), engine.features[lit]?.max || 1);
    else paintMap(null);
    const what = lit != null ? `Where ${litName(lit)} lights up on this picture` : ui.spot != null ? (ui.pinned ? "The spot you picked" : "The spot you are pointing at") : null;
    key.replaceChildren(...(what ? [el("span", { class: "swatch ember" }), what] : []));
  } else if (ui.why && !ui.toward) {
    // The same lights as everywhere else: where on the picture pushes it towards the word.
    let top = 0;
    for (const v of ui.why.patches) if (v > top) top = v;
    paintLights(ui.why.patches, top || 1);
    key.replaceChildren(el("span", { class: "swatch ember" }), `Where it finds “${engine.vocab[ui.word]}”`);
  } else {
    paintMap(null);
    key.replaceChildren();
  }
}

// The spot pointed at: a ring round its patch, the rest dimmed; with a switch hovered, its lights.
function paintSpot(t, id) {
  const canvas = $("heat");
  if (id != null) paintLights(engine.featureMap(ui.state, id), engine.features[id]?.max || 1);
  const g = id != null ? canvas.getContext("2d") : sizeCanvas(canvas);
  if (id == null) {
    g.clearRect(0, 0, canvas.width, canvas.height);
    $("stage").classList.add("mapping");
  }
  const cw = canvas.width / engine.gridW;
  const ch = canvas.height / engine.gridH;
  const x = (t % engine.gridW) * cw;
  const y = Math.floor(t / engine.gridW) * ch;
  if (id == null) {
    g.fillStyle = "rgba(20,20,19,0.35)";
    g.fillRect(0, 0, canvas.width, canvas.height);
    g.clearRect(x, y, cw, ch);
  }
  g.lineWidth = Math.max(2, cw * 0.08);
  g.strokeStyle = "rgba(255,255,255,0.95)";
  g.strokeRect(x, y, cw, ch);
}

function render() {
  renderPanel();
  drawHeat();
  syncAddress();
}

// ------------------------------------------------------------------ panel

function renderPanel() {
  const body = $("panel-body");
  if (!ui.state) {
    if (!ui.live) body.replaceChildren(el("div", { class: "view" }, el("p", { class: "muted" }, "It reads this picture as soon as the model is ready.")));
    return;
  }
  body.replaceChildren(ui.feature != null ? featureView() : ui.view === "inside" ? insideView() : mainView());
}

function wordList(seen, before = null, { pick = true, skip = null } = {}) {
  const order = [...seen.prob.keys()].filter((w) => w !== skip).sort((a, b) => seen.prob[b] - seen.prob[a]).slice(0, N_WORDS);
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
        // Without an edit, a bar. With one, where it was (hollow) and where it is now (filled),
        // joined by the jump: ember when the edit raised it, grey when it lowered it.
        was == null
          ? el("span", { class: "wbar" }, el("span", { class: "now", style: `width:${Math.max(1, p * 100)}%` }))
          : el(
              "span",
              { class: `wtrack${moved ? (p > was ? " up" : " down") : ""}` },
              moved ? el("span", { class: "jump", style: `left:${Math.min(was, p) * 100}%;width:${Math.abs(p - was) * 100}%` }) : null,
              moved ? el("span", { class: "dot was", style: `left:${was * 100}%` }) : null,
              el("span", { class: "dot now", style: `left:${p * 100}%` }),
            ),
        el(
          "span",
          { class: "wpct" },
          pct(p),
          moved ? el("small", { class: p > was ? "up" : "down" }, ` ${p > was ? "+" : "−"}${Math.round(Math.abs(p - was) * 100)}`) : null,
        ),
      );
    }),
  );
}

// The one verb: make it see a word. Type one (any word, where the pack reads any) or tap one.
function seeInput() {
  const list = el("datalist", { id: "vocab" }, engine.vocab.map((w) => el("option", { value: w })));
  const input = el("input", {
    type: "search",
    class: "see-input",
    value: ui.word != null ? engine.vocab[ui.word] : "",
    placeholder: ui.view === "inside" ? "any word" : "a word",
    list: "vocab",
    autocomplete: "off",
    "aria-label": "Make it see",
    "data-testid": "see-input",
  });
  input.addEventListener("change", () => seeWord(input.value));
  const chips = SUGGEST.filter((w) => engine.vocab.includes(w) && engine.vocab.indexOf(w) !== ui.word).slice(0, 4);
  return el(
    "div",
    { class: "see-head" },
    el("label", { class: "see-label" }, el("span", {}, "Make it see"), input, list),
    chips.length ? el("p", { class: "chips" }, chips.map((w) => el("button", { class: "chip", type: "button", "data-word": w, onclick: () => seeWord(w) }, w))) : null,
  );
}

async function seeWord(text) {
  const typed = text.trim().toLowerCase().replace(/\s+/g, " ").slice(0, 60);
  if (!typed) return;
  let w = engine.vocab.indexOf(typed);
  if (w < 0 && engine.readsAnyWord) {
    if (!ui.live || !ui.state?.seen?.embed) return flash("Words off the list work as soon as the model is ready.");
    if (!engine.tokenizer) flash("Getting the part of the model that reads words. Only the first time.");
    try {
      w = await busy(() => engine.addWord(typed));
    } catch (err) {
      console.error(err);
      return flash("That word could not be read here.");
    }
    // The new word joins the list: read the same picture's probabilities over the longer list.
    ui.state.seen = engine.wordsFrom(ui.state.seen.embed);
  }
  if (w < 0) return flash(`“${typed}” is not on the list of ${engine.vocab.length} words.`);
  if (!ui.live && ui.picture?.sample && showBaked(ui.picture.sample, { word: w })) return;
  explain(w);
}

function mainView() {
  const word = engine.vocab[ui.word];
  const seen = ui.state.seen;
  const top = ui.why && ui.word === topWord(seen) && !ui.toward;
  const parts = [backButton(), top ? el("h2", { class: "sw-name why-title" }, `Why it calls this “${word}”`) : seeInput()];
  if (ui.toward) {
    // A word it does not mostly see: the switch, before and after, beside a random switch.
    parts.push(answerCard(), switchView(word));
  } else if (ui.why && ui.word === topWord(seen)) {
    parts.push(...whyView(word));
  } else {
    parts.push(wordList(seen, null, { pick: false }), el("p", { class: "muted" }, "It works this out as soon as the model is ready."));
  }
  return el("div", { class: "view" }, parts);
}

// Where a switch's card goes back to, named the way that screen names itself.
function backTo() {
  if (ui.view === "inside") return "All switches";
  const word = engine.vocab[ui.word];
  return ui.toward ? `Make it see “${word}”` : `Why it calls this “${word}”`;
}

function backButton() {
  return el("button", { class: "back", type: "button", "data-testid": "back", onclick: backToInside }, "← All switches");
}

// One way to show every experiment's answer: the chance it calls the picture a word, before and
// after the edit, with the random control beside it. Used by make it see, switch off and a
// switch's own card, so the three read the same.
function beforeAfter({ word, before, after, wasLabel, nowLabel, control = null, controlLabel = "A random switch instead", testid = null, extra = [], cap = null }) {
  return el(
    "div",
    { class: "answer", id: testid === "boost" ? "answer" : null, "data-testid": testid },
    el("p", { class: "cap" }, cap || `Chance it calls this “${word}”`),
    el(
      "div",
      { class: "anum" },
      el("span", { class: "was", "data-testid": testid === "boost" ? "flip-before" : null }, pct(before), el("small", {}, wasLabel)),
      el("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      el("span", { class: "now", "data-testid": testid === "boost" ? "flip-after" : null }, pct(after), el("small", {}, nowLabel)),
    ),
    control != null ? el("p", { class: "actl", "data-testid": testid === "boost" ? "flip-control" : `${testid}-control` }, `${controlLabel}: `, el("b", {}, pct(control))) : null,
    ...extra,
  );
}

// The answer: how likely it says the word before and with the switch on, the random switch
// beside it, and what it says most.
function answerCard() {
  const w = ui.word;
  const word = engine.vocab[w];
  const before = ui.state.seen.prob[w];
  const control = ui.boostControl;
  if (!ui.boost) return el("div", { class: "answer", id: "answer", "data-testid": "boost" }, el("p", { class: "cap" }, `Chance it calls this “${word}”`), el("div", { class: "anum" }, el("span", { class: "was" }, pct(before))), el("p", { class: "actl" }, ui.live ? "Turning the switch on…" : "Turning the switch on as soon as the model is ready."));
  const after = ui.boost.prob[w];
  const top = topWord(ui.boost);
  const wins = top === w;
  return beforeAfter({
    word,
    before,
    after,
    wasLabel: "as it is",
    nowLabel: ui.push ? `turned up ×${ui.push}` : "as it is",
    control: control?.ids.length ? control.seen.prob[w] : null,
    testid: "boost",
    cap: "Chance it says so", // the word is in the title right above
    extra: [
      wins ? null : el("p", { class: "atop" }, `It still calls it “${engine.vocab[top]}”.`),
      !wins && after < 0.01 && ui.push === 2 ? el("p", { class: "atop" }, "Not enough to change its mind here. Try another switch or picture.") : null,
    ],
  });
}

// The climb: the word's chance at every strength, for the switch and for the random switch.
function curveChart() {
  const box = el("div", { class: "curve", id: "curve", "aria-hidden": "true" });
  fillCurve(box);
  return box;
}

function drawCurve() {
  const box = $("curve");
  if (box) fillCurve(box);
}

function fillCurve(box) {
  const c = ui.curve;
  const w = ui.word;
  if (!c || !ui.state || w == null) return box.replaceChildren();
  const W = 320, H = 116, L = 9, R = 9, T = 14, B = 4;
  const before = ui.state.seen.prob[w];
  const pts = (m) => [[0, before], ...PUSHES.filter((s) => m.has(s)).map((s) => [s, m.get(s).prob[w]])];
  // The top of the chart: the smallest of 10%, 25%, 50%, 100% that fits both lines.
  const high = Math.max(...pts(c.on).map(([, p]) => p), ...pts(c.ctl).map(([, p]) => p));
  const top = [0.1, 0.25, 0.5, 1].find((v) => high <= v) || 1;
  const x = (s) => L + (s / 2) * (W - L - R);
  const y = (p) => T + (1 - Math.min(1, p / top)) * (H - T - B);
  const path = (list) => list.map(([s, p], i) => `${i ? "L" : "M"}${x(s).toFixed(1)},${y(p).toFixed(1)}`).join("");
  const ns = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "curve-svg", role: "img" });
  root.append(
    svg("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), class: "axis" }),
    svg("line", { x1: L, x2: W - R, y1: y(top), y2: y(top), class: "grid" }),
    svg("text", { x: 0, y: y(top) - 4, class: "tick" }, `${top * 100}%`),
    svg("line", { x1: x(ui.push), x2: x(ui.push), y1: y(top), y2: y(0), class: "now-line" }),
  );
  const series = [
    [c.ctl, "ctl", "random"],
    [c.on, "on", "this switch"],
  ];
  const ends = [];
  for (const [m, cls, label] of series) {
    const list = pts(m);
    root.append(svg("path", { d: path(list), class: `line ${cls}` }));
    const at = ui.push ? m.get(ui.push)?.prob[w] : before;
    if (at != null) root.append(svg("circle", { cx: x(ui.push), cy: y(at), r: 4, class: `dot ${cls}` }));
    const [ls, lp] = list[list.length - 1];
    ends.push({ x: x(ls), y: y(lp), cls, text: label.length > 14 ? `${label.slice(0, 13)}…` : label });
  }
  // The names sit at the right end, the switch's above its line, the random one's below unless
  // that would leave the chart.
  const [ctl, on] = ends;
  // Near the top there is no room above the line: the name goes under it.
  on.y = on.y - 7 >= T + 4 ? on.y - 7 : on.y + 15;
  ctl.y = ctl.y + 15 <= H ? ctl.y + 15 : ctl.y - 7;
  if (Math.abs(on.y - ctl.y) < 13) on.y = Math.max(T + 2, ctl.y - 14);
  for (const e of ends) root.append(svg("text", { x: e.x, y: e.y, class: `lab ${e.cls}`, "text-anchor": "end" }, e.text));
  box.replaceChildren(root);
}

// The switch that is on, with where it lights up on other pictures, the slider, and other
// switches for the same word to try instead.
function switchView(word) {
  const list = boosters();
  if (!list.length) return el("p", { class: "muted" }, `Nothing inside it pushes it towards “${word}” on this picture. Try another word.`);
  const id = [...ui.on][0] ?? list[0].id;
  const others = list.filter((r) => r.id !== id);
  return el(
    "div",
    { class: "toward", "data-testid": "toward" },
    curveChart(),
    strengthSlider(),
    el(
      "div",
      { class: "sw section", "data-feature": id },
      el("h3", { class: "sw-name" }, title(id)),
      el("div", { class: "examples", id: "examples", "data-feature": id }, examplesView(id)),
      el("p", { class: "row-actions" }, el("button", { class: "back", type: "button", onclick: () => openFeature(id) }, "More about this switch →")),
    ),
    others.length
      ? el(
          "p",
          { class: "chips others" },
          el("span", { class: "cap" }, "Other switches"),
          others.map((r) => el("button", { class: "chip", type: "button", "data-feature": r.id, onclick: () => pickSwitch(r.id) }, fname(r.id))),
        )
      : null,
    el("p", { class: "row-actions" }, el("button", { class: "pill-btn", type: "button", onclick: copyLink }, "Copy a link to this")),
  );
}

// Strength, under the chart and as wide as it: from off to twice the switch's usual top.
// Used for every strength in the playground (make it see, a switch's own card), so they look and
// move the same. `apply(v)` runs after each move, one at a time; the last move always runs.
function strengthSlider(value = ui.push, set = (v) => (ui.push = v), apply = setPush, testid = "strength") {
  const out = el("output", {}, value ? `×${value}` : "off");
  const input = el("input", { type: "range", min: 0, max: 2, step: 0.25, value, "aria-label": "Strength", "data-testid": testid });
  let queued = false;
  let running = false;
  input.addEventListener("input", async () => {
    const v = Number(input.value);
    set(v);
    out.textContent = v ? `×${v}` : "off";
    if (running) {
      queued = true;
      return;
    }
    running = true;
    do {
      queued = false;
      await apply(v);
    } while (queued);
    running = false;
  });
  return el("label", { class: "strength" }, input, el("span", { class: "strength-row" }, el("span", {}, "off"), el("span", {}, "Strength ", out), el("span", {}, "×2")));
}

// What turning it up did on other pictures (ii name-live): the words it made the model say.
function saysView(id) {
  const f = engine.features[id] || {};
  const said = (f.sees || []).filter((r) => r.top_on >= 2).slice(0, 3); // once in 8 is noise
  if (!said.length) return null;
  const n = said[0].of;
  const times = (k) => (k === 1 ? "once" : k === 2 ? "twice" : `${k} times`);
  const parts = said.map((r) => [el("b", {}, `“${r.word}”`), ` ${times(r.top_on)}`]);
  return el(
    "p",
    { class: "says" },
    `Turned up on ${n} other pictures, it made the model say `,
    parts.flatMap((p, i) => (i ? [i === parts.length - 1 ? " and " : ", ", ...p] : p)),
    ".",
  );
}

// Where a switch lights up most by itself (at least a quarter of its strongest in training, so a
// faint flicker does not pass for what it stands for): from the pack's gallery of a few thousand
// photos (ii gallery-live) when it has one, else from the example pictures.
function examplesView(id, n = N_EXAMPLES) {
  const peak = engine.features[id]?.max || 1;
  let list;
  if (engine.pack.files?.gallery) {
    list = gallery.tops.get(id);
    if (!list) {
      galleryTops(id).then(() => {
        const box = document.querySelector(`.examples[data-feature="${id}"]`);
        if (box) box.replaceChildren(...[examplesView(id, Number(box.dataset.n) || N_EXAMPLES)].flat().filter(Boolean));
      });
      return null;
    }
    const here = ui.picture?.sample ?? ui.picture?.gallery;
    list = list.filter((g) => (g.pic.sample ?? g.pic.file) !== here);
  } else {
    const here = ui.picture?.sample;
    const live = ui.gallery.length === engine.pack.samples.length;
    if (live) list = examplesFor(id).filter((g) => g.sample.clip_id !== here).map((g) => ({ pic: g.sample, map: engine.featureMap(g, id) }));
    else {
      const baked = ui.instant?.examples?.[id];
      if (!baked) return null;
      list = baked.filter(([key]) => key !== here).map(([key, b64, p]) => ({ pic: sampleOf(key), map: unq8(b64, p) }));
    }
    list = list.filter(({ map }) => Math.max(...map) >= 0.25 * peak);
  }
  list = list.slice(0, n);
  if (!list.length) return null;
  const read = gallery.read;
  return [
    el("p", { class: "cap" }, read ? `Where it lights up most, out of ${read.toLocaleString("en-GB")} photos` : "Where it lights up by itself"),
    el(
      "div",
      { class: `ex-row${n > 4 ? " three" : ""}` },
      list.map(({ pic, map }) => {
        const canvas = el("canvas", { width: engine.gridW * 10, height: engine.gridH * 10 });
        paintLights(map, Math.max(...map), canvas); // where, not how much: its own strongest patch is full
        return el(
          "button",
          { class: "ex", type: "button", title: pic.credit || "", onclick: () => openExample(pic, id) },
          el("img", { src: engine.base + pic.file, alt: "", loading: "lazy" }),
          canvas,
        );
      }),
    ),
  ];
}

// The pack's gallery (ii gallery-live): each switch's strongest pictures sit in one of a few
// small files, fetched when that switch is first shown.
const gallery = { index: null, shards: new Map(), tops: new Map() };

async function galleryTops(id) {
  const file = engine.pack.files.gallery;
  gallery.index ||= fetch(engine.base + file).then((r) => r.json()).catch(() => null);
  const index = await gallery.index;
  if (!index) return gallery.tops.set(id, []).get(id);
  gallery.read = index.read;
  const s = id % index.shards;
  const url = engine.base + file.replace(/[^/]*$/, "") + `top-${String(s).padStart(3, "0")}.json`;
  if (!gallery.shards.has(s)) gallery.shards.set(s, fetch(url).then((r) => r.json()).catch(() => ({})));
  const shard = await gallery.shards.get(s);
  const list = (shard[id] || []).map((pic) => ({ pic, map: unq8(pic.map, pic.peak) }));
  gallery.tops.set(id, list);
  return list;
}

// Open an example picture with the same switch: still turned up for the word, or still open.
async function openExample(pic, id) {
  const opened = ui.feature === id;
  const opts = { word: ui.word, on: opened || !ui.on.has(id) ? [] : [id] };
  try {
    if (pic.sample) await showSample(sampleOf(pic.sample), opts);
    else {
      const img = await loadImage(engine.base + pic.file);
      stopCam();
      ui.picture = { img, gallery: pic.file };
      markThumb(null);
      $("credit").textContent = pic.credit || "";
      await showPicture(img, opts);
    }
  } catch (err) {
    return flash(err.message);
  }
  $("stage").scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (opened) await openFeature(id);
}

// A word it already sees most: the features that make it see that, to switch off.
function whyView(word) {
  const seen = ui.state.seen;
  // A switch's name is the word turning it up makes the model say on other pictures. Beside a
  // picture it is named only when that word is among what it might call this picture, so a
  // "camera" switch on a teddy bear's eye does not read as a camera in the picture.
  const near = new Set([...seen.prob.keys()].sort((a, b) => seen.prob[b] - seen.prob[a]).slice(0, 20).map((w) => engine.vocab[w]));
  const fits = (id) => engine.features[id]?.name && near.has(engine.features[id].name);
  const list = reasons();
  const top = list[0]?.score || 1; // bars against the first, as in the list of what lights up
  const allOff = list.length && list.every((r) => ui.off.has(r.id));
  const waiting = !ui.live && ui.off.size && !ui.result;
  return [
    el("p", { class: "muted" }, list.length ? `These switches push it most towards “${word}”. Turn them off and watch it change its mind.` : `No switch on this picture stands out as the reason.`),
    resultCard(word),
    wordList(ui.result || seen, ui.result ? seen : null, { pick: false }),
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
            { class: "fname-btn lit-mid", type: "button", onclick: () => openFeature(r.id) },
            el("span", { class: fits(r.id) ? "fname" : "fname unnamed" }, fits(r.id) ? engine.features[r.id].name : `#${r.id}`),
            el("span", { class: "fbar" }, el("span", { style: `width:${Math.max(4, Math.round((r.score / top) * 100))}%` })),
          ),
          previewsOf(r.id),
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
  ];
}

function resultCard(word) {
  if (!ui.result) return null;
  const w = ui.word;
  const before = ui.state.seen.prob[w];
  const after = ui.result.prob[w];
  const n = ui.off.size;
  const control = ui.control;
  const sw = (k) => `${k} switch${k === 1 ? "" : "es"}`;
  return beforeAfter({
    word,
    before,
    after,
    wasLabel: "as it is",
    nowLabel: `${sw(n)} off`,
    control: control ? control.seen.prob[w] : null,
    controlLabel: control ? `${control.ids.length} random switch${control.ids.length === 1 ? "" : "es"} off instead` : "",
    testid: "result",
  });
}

// One switch on its own: where it lights up on other photos, what turning it up did on other
// pictures, and turning it up on this one beside a random switch turned up the same way.
function featureView() {
  const id = ui.feature;
  return el(
    "div",
    { class: "view", "data-testid": "feature", "data-feature": id },
    el(
      "button",
      {
        class: "back",
        type: "button",
        "data-testid": "feature-back",
        onclick: () => {
          ui.feature = null;
          ui.detail = ui.detailControl = null;
          render();
        },
      },
      `← ${backTo()}`,
    ),
    el("h2", { class: "sw-name" }, title(id)),
    el("p", { class: "lit-note" }, `One of the ${engine.pack.sae.n_features.toLocaleString("en-GB")} patterns the model picks up on${title(id).startsWith("Switch #") ? "" : `, #${id}`}`),
    el("div", { class: "examples", id: "examples", "data-feature": id, "data-n": N_TOP }, examplesView(id, N_TOP)),
    saysView(id),
    el(
      "div",
      { class: "section" },
      el("p", { class: "cap" }, "Turn this switch up on your picture"),
      strengthSlider(ui.strength.add, (v) => (ui.strength.add = v), () => runDetail(), "detail-strength"),
      el("div", { id: "detail-words" }, detailView()),
    ),
  );
}

// The word turning the switch up moved most, before and after, the random switch beside it, and
// every word it might call the picture.
function detailView() {
  if (!ui.detail) return el("p", { class: "muted" }, "Running the model…");
  const before = ui.state.seen;
  const after = ui.detail;
  if (!ui.strength.add) return wordList(before, null, { pick: false });
  let w = 0;
  for (let i = 1; i < after.prob.length; i++) if (after.prob[i] - before.prob[i] > after.prob[w] - before.prob[w]) w = i;
  const ctl = ui.detailControl;
  const noMore = ctl && Math.abs(after.prob[w] - before.prob[w]) <= Math.abs(ctl.prob[w] - before.prob[w]);
  return [
    beforeAfter({
      word: engine.vocab[w],
      before: before.prob[w],
      after: after.prob[w],
      wasLabel: "as it is",
      nowLabel: `turned up ×${ui.strength.add}`,
      control: ctl ? ctl.prob[w] : null,
      testid: "detail",
      extra: [noMore ? el("p", { class: "atop" }, "This switch does no more than a random one here.") : null],
    }),
    wordList(after, before, { pick: false, skip: w }),
  ];
}

// ------------------------------------------------------------------ gallery (in the background)

function encodeGallery() {
  if (!encodeGallery.run) {
    encodeGallery.run = (async () => {
      for (const sample of engine.pack.samples) {
        while ((ui.busy || ui.cam || ui.curveBusy) && !BAKE) await new Promise((r) => setTimeout(r, 150));
        const img = await loadImage(engine.base + sample.file);
        const state = await engine.encode(img, { readout: false });
        ui.gallery.push({ sample, img, rows: state.rows, peaks: state.peaks });
        const box = $("gallery");
        if (box) box.replaceChildren(...[galleryFor(Number(box.dataset.feature))].flat());
      }
      const ex = $("examples");
      if (ex) ex.replaceChildren(...[examplesView(Number(ex.dataset.feature))].flat().filter(Boolean));
    })();
  }
  return encodeGallery.run;
}

// The example pictures a switch lights up on most, strongest first (the gallery read so far).
function examplesFor(id, n = N_EXAMPLES + 1) {
  return ui.gallery.filter((g) => g.peaks[id] > 0).sort((a, b) => b.peaks[id] - a.peaks[id]).slice(0, n);
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
  wireStage();
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
    await showPicture(img, { word: ui.view === "word" ? ui.word : null });
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
  await showPicture(img, { word: ui.view === "word" ? ui.word : null });
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
  if (ui.on.size) {
    q.set("on", [...ui.on].join(","));
    q.set("s", String(ui.push));
  }
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
    flash("Link copied. It opens this picture with the same switches.");
  } catch {
    window.prompt("Copy this link:", url);
  }
}

async function restoreFromLink() {
  const q = new URLSearchParams(location.hash.slice(1));
  const sample = engine.pack.samples.find((s) => s.clip_id === q.get("p"));
  const word = engine.vocab.indexOf(q.get("w") || "");
  if (word < 0 && q.get("w") && engine.readsAnyWord) ui.wantWord = q.get("w").slice(0, 60);
  const ids = (key) => (q.get(key) || "").split(",").filter(Boolean).map(Number).filter((n) => Number.isInteger(n) && n >= 0 && n < engine.nFeatures);
  const off = ids("off");
  const on = ids("on");
  const push = Number(q.get("s"));
  if (q.has("s") && push >= 0 && push <= 2) ui.push = push;
  try {
    if (sample) await showSample(sample, { word: word >= 0 ? word : null, off, on });
    else if (q.get("img")) {
      const img = await loadImage(`data:image/jpeg;base64,${q.get("img").replace(/-/g, "+").replace(/_/g, "/")}`);
      ui.picture = { img };
      await showPicture(img, { word: word >= 0 ? word : null, off, on });
    } else return false;
  } catch {
    flash("That link's picture could not be opened.");
    return false;
  }
  const f = Number(q.get("f"));
  if (q.has("f") && Number.isInteger(f) && f >= 0 && f < engine.nFeatures) await openFeature(f);
  return true;
}

// ------------------------------------------------------------------ messages

function flash(message) {
  const box = el("div", { class: "toast", role: "status" }, message);
  document.body.append(box);
  setTimeout(() => box.remove(), 4000);
}

boot();
