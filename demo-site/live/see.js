// ii playground: find the switch. Pick a picture and a word; the page finds the feature inside
// the model that most makes it see that word, turns it on, and shows the answer before, with the
// switch, and with a random switch instead. Runs a version 2 pack with the words readout
// (ii export-live --readout words) through engine.js, in the browser. Exploration only: nothing
// here can change a feature's badge.

import { Engine } from "./engine.js";

const PARAMS = new URLSearchParams(location.search);
const PACK = PARAMS.get("pack") || "packs/clip-b16-words/";
const EMBER = [232, 88, 42];
const FIRST = "06-droid-083"; // the picture the page opens on
const TARGET = "umbrella"; // the word it opens on (packs without it use their last word)
const SUGGEST = ["umbrella", "cat", "pizza", "snow", "guitar"]; // one tap each, where on the list
const BAKE = PARAMS.has("bake");
const N_WORDS = 6; // words shown
const N_WHY = 6; // features shown for a word
const N_ON = 3; // features offered to turn on
const TRY = 5; // switches tried for a word, and random switches tried as the control
const THUMB = 44; // px, the crop of the picture shown beside each feature
const PUSHES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]; // the Strength slider's steps (0 is off)
const N_EXAMPLES = 4; // other pictures shown for a switch
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
  encodeGallery();
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
  return showSample(pack.samples.find((s) => s.clip_id === FIRST) || pack.samples[0], { word: targetWord() });
}

// The model is ready: work out for real what the page has shown so far (the baked answer, or a
// picture, word or switch chosen while it loaded), keeping the same picture, word and switches.
async function goLive() {
  const feature = ui.pendingFeature;
  ui.pendingFeature = null;
  ui.baked = null;
  if (!ui.cam && ui.picture) {
    const img = ui.picture.sample ? await loadImage(engine.base + sampleOf(ui.picture.sample).file) : ui.picture.img;
    await showPicture(img, { word: ui.word, off: [...ui.off], on: [...ui.on] });
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
  if (rec?.flip && word === rec.flip.word && !off.length) return showBakedFlip(rec);
  if (!rec || (word != null && word !== rec.word)) return false;
  ui.baked = rec;
  ui.state = { baked: true, seen: probs(rec.prob), maps: new Map(Object.entries(rec.maps).map(([id, [b64, peak]]) => [Number(id), unq8(b64, peak)])) };
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
  Object.assign(ui, { feature: null, detail: null, why: null, result: null, control: null, word: flip.word, off: new Set(), on: new Set(flip.on), push: flip.push });
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
    ui.word = null; // each picture's own top word first
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
    Object.assign(ui, { state: null, why: null, result: null, control: null, baked: null, feature: null, word, off: new Set(off), on: new Set(on) });
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
  await explain(word ?? ui.word ?? topWord(ui.state.seen), off, on);
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

const curveKey = () => JSON.stringify([ui.picture?.sample || "own", ui.word, [...ui.on], ui.boostControl?.ids || []]);

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

// A feature's name (the word writing it in makes the model see, ii name-live), else its number.
function fname(id) {
  return engine.features[id]?.name || `Switch ${id}`;
}

// “kitten”, or #3880 for a switch with no name.
const switchName = (id) => (engine.features[id]?.name ? `“${engine.features[id].name}”` : `#${id}`);

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
  ui.detail = null;
  ui.hover = null;
  render();
  await runDetail();
}

async function runDetail() {
  const id = ui.feature;
  if (id == null) return;
  const seen = await busy(() => engine.see(ui.state, pushEdits([id], ui.strength.add)));
  if (ui.feature !== id) return;
  ui.detail = seen;
  const box = $("detail-words");
  if (box) box.replaceChildren(wordList(seen, ui.state.seen, { pick: false }));
  else renderPanel();
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
  if (id != null) {
    const peak = engine.features[id]?.max || 1;
    const add = ui.hover == null ? ui.strength.add : 0;
    paintLights(engine.featureMap(ui.state, id).map((v) => v + add * peak), 3 * peak);
    key.replaceChildren(el("span", { class: "swatch ember" }), add ? `The ${switchName(id)} switch, turned up ×${add} everywhere` : `Where the ${switchName(id)} switch lights up by itself`);
  } else if (on != null) {
    // The switch on this picture: where it already lights up, plus the even glow of turning it
    // up everywhere, brighter as the strength goes up.
    const peak = engine.features[on]?.max || 1;
    const lit = engine.featureMap(ui.state, on).map((v) => v + ui.push * peak);
    paintLights(lit, 3 * peak);
    key.replaceChildren(el("span", { class: "swatch ember" }), ui.push ? `The ${switchName(on)} switch, turned up ×${ui.push} everywhere` : `Where the ${switchName(on)} switch lights up by itself`);
  } else if (ui.why && !ui.toward) {
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

// The one verb: make it see a word. Type one (any word, where the pack reads any) or tap one.
function seeInput() {
  const list = el("datalist", { id: "vocab" }, engine.vocab.map((w) => el("option", { value: w })));
  const input = el("input", {
    type: "search",
    class: "see-input",
    value: ui.word != null ? engine.vocab[ui.word] : "",
    placeholder: "a word",
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
  const parts = [seeInput()];
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

// The answer: how likely it says the word before and with the switch on, the random switch
// beside it, and what it says most.
function answerCard() {
  const w = ui.word;
  const word = engine.vocab[w];
  const before = ui.state.seen.prob[w];
  const control = ui.boostControl;
  if (!ui.boost) return el("div", { class: "answer", id: "answer", "data-testid": "boost" }, el("p", { class: "alab" }, `Chance it calls this “${word}”`), el("p", { class: "anum" }, pct(before)), el("p", { class: "muted small" }, ui.live ? "Turning the switch on…" : "Turning the switch on as soon as the model is ready."));
  const after = ui.boost.prob[w];
  const top = topWord(ui.boost);
  const wins = top === w;
  return el(
    "div",
    { class: "answer", id: "answer", "data-testid": "boost" },
    el("p", { class: "alab" }, `Chance it calls this “${word}”`),
    el(
      "div",
      { class: "anum" },
      el("span", { class: "was", "data-testid": "flip-before" }, pct(before), el("small", {}, "switch off")),
      el("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      el("span", { class: "now", "data-testid": "flip-after" }, pct(after), el("small", {}, ui.push ? `switch on ×${ui.push}` : "switch off")),
    ),
    control?.ids.length
      ? el("p", { class: "actl", "data-testid": "flip-control" }, "A random switch instead: ", el("b", {}, pct(control.seen.prob[w])))
      : null,
    el("p", { class: `atop${wins ? " wins" : ""}` }, wins ? `Now it calls it “${word}”.` : `It still calls it “${engine.vocab[top]}”.`),
    !wins && after < 0.01 && ui.push === 2 ? el("p", { class: "atop" }, "Not enough to change its mind here. Try another switch or picture.") : null,
  );
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
    [c.on, "on", engine.features[[...ui.on][0]]?.name || `#${[...ui.on][0]}`],
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
  on.y = Math.max(T + 2, on.y - 7);
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
      { class: "sw", "data-feature": id },
      el("h3", { class: "sw-name" }, `The ${switchName(id)} switch`),
      saysView(id),
      el("div", { class: "examples", id: "examples", "data-feature": id }, examplesView(id)),
    ),
    others.length
      ? el(
          "p",
          { class: "chips others" },
          el("span", { class: "muted" }, "Other switches:"),
          others.map((r) => el("button", { class: "chip", type: "button", "data-feature": r.id, onclick: () => pickSwitch(r.id) }, fname(r.id))),
        )
      : null,
    el("p", { class: "row-actions" }, el("button", { class: "pill-btn", type: "button", onclick: copyLink }, "Copy a link to this")),
  );
}

// Strength, under the chart and as wide as it: from off to twice the switch's usual top.
function strengthSlider() {
  const out = el("output", {}, ui.push ? `×${ui.push}` : "off");
  const input = el("input", { type: "range", min: 0, max: 2, step: 0.25, value: ui.push, "aria-label": "Strength", "data-testid": "strength" });
  let queued = false;
  let running = false;
  input.addEventListener("input", async () => {
    ui.push = Number(input.value);
    out.textContent = ui.push ? `×${ui.push}` : "off";
    if (running) {
      queued = true;
      return;
    }
    running = true;
    do {
      queued = false;
      await setPush(ui.push);
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

// Where a switch lights up most on the other example pictures, by itself (at least a quarter of
// its strongest in training, so a faint flicker does not pass for what it stands for).
function examplesView(id) {
  const here = ui.picture?.sample;
  const live = ui.gallery.length === engine.pack.samples.length;
  let list;
  if (live) list = examplesFor(id).filter((g) => g.sample.clip_id !== here).map((g) => ({ key: g.sample.clip_id, map: engine.featureMap(g, id) }));
  else {
    const baked = ui.instant?.examples?.[id];
    if (!baked) return null;
    list = baked.filter(([key]) => key !== here).map(([key, b64, peak]) => ({ key, map: unq8(b64, peak) }));
  }
  const peak = engine.features[id]?.max || 1;
  list = list.filter(({ map }) => Math.max(...map) >= 0.25 * peak).slice(0, N_EXAMPLES);
  if (!list.length) return null;
  return [
    el("p", { class: "muted small ex-cap" }, "Where it lights up by itself"),
    el(
      "div",
      { class: "ex-row" },
      list.map(({ key, map }) => {
        const sample = sampleOf(key);
        const canvas = el("canvas", { width: engine.gridW * 10, height: engine.gridH * 10 });
        paintLights(map, peak, canvas);
        return el(
          "button",
          { class: "ex", type: "button", title: sample.credit || key, onclick: () => showSample(sample) },
          el("img", { src: engine.base + sample.file, alt: "", loading: "lazy" }),
          canvas,
        );
      }),
    ),
  ];
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
  const total = list.reduce((a, r) => a + r.score, 0) || 1;
  const allOff = list.length && list.every((r) => ui.off.has(r.id));
  const waiting = !ui.live && ui.off.size && !ui.result;
  return [
    el("p", { class: "muted" }, list.length ? `It already calls this “${word}”. These switches are why. Turn them off and watch it change its mind.` : `It already calls this “${word}”, and no switch on this picture stands out as the reason.`),
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
            { class: "fname-btn", type: "button", onclick: () => openFeature(r.id) },
            thumbOf(r.id),
            el("span", { class: fits(r.id) ? "fname" : "fname unnamed" }, fits(r.id) ? engine.features[r.id].name : `#${r.id}`),
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
  ];
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
      `With ${n} switch${n === 1 ? "" : "es"} off. `,
      control
        ? `Turning off ${control.ids.length} random switch${control.ids.length === 1 ? "" : "es"} that are also on here leaves it at ${pct(control.seen.prob[w])}.`
        : "",
    ),
  );
}

// One switch on its own (from the list of why it sees a word): what it is, where it lights up,
// and what turning it up on this picture does to every word.
function featureView() {
  const id = ui.feature;
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
      "← Back",
    ),
    el("h2", { class: "sw-name" }, `The ${switchName(id)} switch`),
    saysView(id),
    el("div", { class: "examples", id: "examples", "data-feature": id }, examplesView(id)),
    el("h3", { class: "detail-h" }, "Turn it up on this picture"),
    slider("Strength", 0, 2, 0.25, ui.strength.add, (v) => (v ? `×${v}` : "off"), (v) => (ui.strength.add = v), async () => {
      await runDetail();
      drawHeat();
    }),
    el("div", { id: "detail-words" }, ui.detail ? wordList(ui.detail, ui.state.seen, { pick: false }) : el("p", { class: "muted" }, "Running the model…")),
  );
}

function slider(label, min, max, step, value, fmt, set, run = runDetail) {
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
      await run();
    } while (queued);
  });
  return el("label", { class: "slider" }, el("span", {}, label), input, out);
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
