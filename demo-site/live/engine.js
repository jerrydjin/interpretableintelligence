// ii live engine: runs a pack (model split at the SAE's layer + the SAE) in the browser.
//
// A pack is what `ii export-live` writes (pack.json says which files). The model's first
// half and the SAE load first, so a picture can be read; the second half follows in the
// background and is only needed for an edit. The engine:
//   encode(image)   pixels -> hidden at the SAE's layer -> every feature on every patch,
//                   plus the model's clean final tokens (the playground's readout).
//   edit(ids, ...)  the playground edit, x' = x + sum_i (f'_i - f_i) W_dec[:, i], on the
//                   patch tokens, then the rest of the network, then the per-patch
//                   cosine distance between clean and edited final tokens.
// Same maths as `steer.steer_edit` and the local playground. Exploration only: nothing
// here can change a feature's badge.
//
// A version 2 pack with the words readout (CLIP, `ii export-live --readout words`) has no
// "back" half. Its "words" graph takes the hidden state and a vector u and returns the image
// embedding e and d(e.u)/d(hidden), and text.bin holds every word's text embedding, so:
//   encode(image)        also gives what the model sees: a softmax over the words.
//   why(state, word)     the features that make it see that word: activation x gradient of
//                        the word's log-probability along each feature's direction, summed
//                        over patches (a first-order estimate; switching off measures it).
//   see(state, edits)    the words after an edit, e.g. [{id, kind: "ablate"}, ...].
//   clearPatches(state, patches)  the words with the features taken off some patches.

const ORT_VERSION = "1.31.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// v1 keyed files by model sha only, so a re-exported pack could load a stale graph; v2 keys
// each file by its digest in pack.json.
const CACHE_NAME = "ii-live-v2";
const OLD_CACHES = ["ii-live-v1"];

export const KINDS = ["ablate", "scale", "clamp"];

let ortPromise = null;

async function webgpuAvailable() {
  if (!("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function loadOrt(preferGpu) {
  if (!ortPromise) {
    ortPromise = (async () => {
      const gpu = preferGpu && (await webgpuAvailable());
      const ort = await import(/* webpackIgnore: true */ ORT_BASE + (gpu ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
      ort.env.wasm.wasmPaths = ORT_BASE;
      const cores = navigator.hardwareConcurrency || 4;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(cores, 8) : 1;
      ort.env.logLevel = "error";
      return { ort, backend: gpu ? "webgpu" : "wasm" };
    })();
  }
  return ortPromise;
}

// Fetch with progress, cached in the Cache API so a second visit skips the download.
async function fetchBytes(url, key, onBytes) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(key);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onBytes(buf.byteLength, buf.byteLength, true);
      return buf;
    }
  } catch {
    cache = null;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const parts = [];
  let seen = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    seen += value.byteLength;
    onBytes(seen, total, false);
  }
  const buf = new Uint8Array(seen);
  let at = 0;
  for (const part of parts) {
    buf.set(part, at);
    at += part.byteLength;
  }
  if (cache) {
    try {
      await cache.put(key, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
    } catch {
      /* quota or private mode: fine, just slower next time */
    }
  }
  return buf;
}

// A pack file, or a big one shipped as parts (pack.json lists them), joined back together.
async function fetchFile(base, entry, version, onBytes) {
  const parts = Array.isArray(entry) ? entry : [entry];
  const seen = new Array(parts.length).fill(0);
  const totals = new Array(parts.length).fill(0);
  let cachedAll = true;
  const bufs = await Promise.all(
    parts.map((name, i) =>
      fetchBytes(base + name, `${base}${name}?v=${version}`, (got, total, cached) => {
        seen[i] = got;
        totals[i] = total || got;
        if (!cached) cachedAll = false;
        const sum = (a) => a.reduce((x, y) => x + y, 0);
        onBytes(sum(seen), sum(totals), cachedAll);
      }),
    ),
  );
  if (bufs.length === 1) return bufs[0];
  const out = new Uint8Array(bufs.reduce((a, b) => a + b.byteLength, 0));
  let at = 0;
  for (const b of bufs) {
    out.set(b, at);
    at += b.byteLength;
  }
  return out;
}

function halfToFloat(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) out[i] = s * 2 ** -14 * (f / 1024);
    else if (e === 31) out[i] = f ? NaN : s * Infinity;
    else out[i] = s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

const now = () => performance.now();

export class Engine {
  constructor(baseUrl) {
    this.base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
    this.timings = {};
    this._queue = Promise.resolve();
  }

  // ONNX Runtime sessions run one call at a time; queue everything that touches them.
  _serial(fn) {
    const run = this._queue.then(fn, fn);
    this._queue = run.catch(() => {});
    return run;
  }

  // The pack's description (and, for a words pack, its word list): enough to draw a baked answer
  // (`instant`) before any model file arrives. load() calls it too.
  async loadMeta() {
    if (this._meta) return this._meta;
    this._meta = (async () => {
      const pack = await (await fetch(this.base + "pack.json")).json();
      if (pack.format !== "ii-live-pack" || ![1, 2].includes(pack.version)) throw new Error("this pack is not an ii live pack");
      this.pack = pack;
      this.wordsMode = pack.model.readout === "words";
      if (this.wordsMode) {
        // The words and the feature names, so a baked answer can show names before the model.
        [this.vocab, this.features] = await Promise.all([pack.files.vocab, pack.files.features].map((f) => fetch(this.base + f).then((r) => r.json())));
      }
      this.nFeatures = pack.sae.n_features;
      this.d = pack.model.d_model;
      this.gridH = pack.model.grid.h;
      this.gridW = pack.model.grid.w;
      this.nTok = this.gridH * this.gridW;
      this.prefix = pack.model.prefix_tokens;
      this.size = pack.model.image_size;
      return pack;
    })();
    return this._meta;
  }

  // The first answer for each sample picture, worked out by this page ahead of time (ii bake-live),
  // or null. Ignored when the model files changed since.
  async loadInstant() {
    const pack = await this.loadMeta();
    if (!pack.files.instant) return null;
    try {
      const baked = await (await fetch(this.base + pack.files.instant)).json();
      // Only the files that change the numbers count (new feature names do not).
      const digests = Object.entries(baked.digests || {});
      const same = digests.length && digests.every(([key, digest]) => (pack.digests || {})[key] === digest);
      return same && baked.version === 1 ? baked : null;
    } catch {
      return null;
    }
  }

  // onProgress({ phase, loaded, total, cached })
  async load({ onProgress = () => {}, preferGpu = true } = {}) {
    const t0 = now();
    const pack = await this.loadMeta();
    const { ort, backend } = await loadOrt(preferGpu);
    this.ort = ort;
    this.backend = backend;
    const files = pack.files;
    const sha = (pack.model.sha || "").slice(0, 12);
    const digests = pack.digests || {};
    for (const old of OLD_CACHES) globalThis.caches?.delete(old)?.catch(() => {});
    // The rest of the network ("back") is only needed to make an edit, so it downloads and
    // starts after the first picture is read: the first answer arrives on about half the bytes.
    const want = this.wordsMode ? ["front", "sae", "decoder", "words", "text"] : ["front", "sae", "decoder"];
    const sizes = {};
    const loaded = {};
    let cachedAll = true;
    const report = () => {
      const got = Object.values(loaded).reduce((a, b) => a + b, 0);
      const total = Object.values(sizes).reduce((a, b) => a + b, 0);
      onProgress({ phase: "download", loaded: got, total, cached: cachedAll });
    };
    let backSeen = 0;
    if (!this.wordsMode) {
      const backKey = `${this.base}${files.back}?v=${digests.back || sha}`;
      this._backBytes = fetchBytes(this.base + files.back, backKey, (seen) => (backSeen = seen));
      this._backBytes.catch(() => {}); // surfaced by the first edit, not here
    }
    const [bytes, features] = await Promise.all([
      Promise.all(
        want.map((name) =>
          fetchFile(this.base, files[name], digests[name] || sha, (seen, total, cached) => {
            loaded[name] = seen;
            sizes[name] = total || seen;
            if (!cached) cachedAll = false;
            report();
          }),
        ),
      ),
      this.features || fetch(this.base + files.features).then((r) => r.json()),
    ]);
    this.timings.download = now() - t0;
    onProgress({ phase: "compile" });
    const t1 = now();
    this.decoder = halfToFloat(new Uint16Array(bytes[2].buffer, bytes[2].byteOffset, bytes[2].byteLength / 2)); // [n_features, d]
    this._bytes = { front: bytes[0], sae: bytes[1], words: this.wordsMode ? bytes[3] : null };
    this.features = features;
    if (this.wordsMode) {
      this.textDim = pack.words.dim;
      this.text = halfToFloat(new Uint16Array(bytes[4].buffer, bytes[4].byteOffset, bytes[4].byteLength / 2)); // [n_words, dim]
      this.logitScale = pack.words.logit_scale;
    }
    try {
      await this.startSessions(backend);
    } catch (err) {
      if (backend !== "webgpu") throw err;
      // Some GPUs and browsers cannot run a graph on WebGPU; the CPU path always can.
      console.warn("WebGPU could not run the model, using the CPU instead:", err);
      this.gpuError = String(err && err.message ? err.message : err);
      await this.startSessions("wasm");
    }
    this.timings.compile = now() - t1;
    const t2 = now();
    try {
      await this.encode(this.blankCanvas(), { readout: false });
    } catch (err) {
      if (this.backend !== "webgpu") throw err;
      console.warn("WebGPU failed on the first pass, using the CPU instead:", err);
      this.gpuError = String(err && err.message ? err.message : err);
      await this.startSessions("wasm");
      await this.encode(this.blankCanvas(), { readout: false });
    }
    this.timings.warmup = now() - t2;
    this.timings.ready = now() - t0;
    // Queued now, so it runs before any edit asks for it.
    this.backReady = this.wordsMode ? Promise.resolve() : this._serial(() => this._startBack());
    this.backReady.catch(() => {});
    this.backProgress = () => backSeen;
    onProgress({ phase: "ready" });
    return this;
  }

  _options(backend) {
    return {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
    };
  }

  async _startBack() {
    const t0 = now();
    const bytes = await this._backBytes;
    try {
      await this.back?.release();
    } catch {
      /* already gone */
    }
    try {
      this.back = await this.ort.InferenceSession.create(bytes, this._options(this.backend));
    } catch (err) {
      if (this.backend !== "webgpu") throw err;
      console.warn("WebGPU could not run the rest of the model, using the CPU instead:", err);
      this.gpuError = String(err && err.message ? err.message : err);
      await this.startSessions("wasm");
      this.back = await this.ort.InferenceSession.create(bytes, this._options("wasm"));
    }
    this.timings.back = now() - t0;
  }

  // One session at a time: ONNX Runtime's WebGPU backend refuses to create two at once.
  // The back half starts on its own (_startBack), after the first picture.
  async startSessions(backend) {
    const options = this._options(backend);
    for (const s of [this.front, this.sae]) {
      try {
        await s?.release();
      } catch {
        /* already gone */
      }
    }
    this.front = await this.ort.InferenceSession.create(this._bytes.front, options);
    this.sae = await this.ort.InferenceSession.create(this._bytes.sae, options);
    if (this.wordsMode) {
      try {
        await this.words?.release();
      } catch {
        /* already gone */
      }
      this.words = await this.ort.InferenceSession.create(this._bytes.words, options);
    }
    this.backend = backend;
  }

  blankCanvas() {
    const c = document.createElement("canvas");
    c.width = c.height = this.size;
    const g = c.getContext("2d");
    g.fillStyle = "#808080";
    g.fillRect(0, 0, c.width, c.height);
    return c;
  }

  // The hf adapter resizes the whole frame to the model's square input, no crop.
  pixels(source) {
    const s = this.size;
    if (!this._canvas) {
      this._canvas = document.createElement("canvas");
      this._canvas.width = this._canvas.height = s;
      this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    }
    const g = this._ctx;
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    if (this.pack.model.resize === "crop") {
      // The middle square of the picture, as CLIP was trained (and as the page shows it).
      const w = source.naturalWidth || source.videoWidth || source.width;
      const h = source.naturalHeight || source.videoHeight || source.height;
      const side = Math.min(w, h);
      g.drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, s, s);
    } else g.drawImage(source, 0, 0, s, s);
    const rgba = g.getImageData(0, 0, s, s).data;
    const [m0, m1, m2] = this.pack.model.mean;
    const [s0, s1, s2] = this.pack.model.std;
    const plane = s * s;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; p < plane; p++, i += 4) {
      out[p] = (rgba[i] / 255 - m0) / s0;
      out[plane + p] = (rgba[i + 1] / 255 - m1) / s1;
      out[2 * plane + p] = (rgba[i + 2] / 255 - m2) / s2;
    }
    return out;
  }

  // Features on every patch: sparse rows (idx, val) of the SAE's activation.
  activate(pre) {
    const { nTok, nFeatures } = this;
    const sae = this.pack.sae;
    const k = sae.activation === "topk" ? sae.k : 0;
    const thr = sae.activation === "jumprelu" ? sae.threshold : 0;
    const rows = new Array(nTok);
    for (let t = 0; t < nTok; t++) {
      const off = t * nFeatures;
      let idx = [];
      let val = [];
      if (k) {
        // Keep the k largest (a small sorted buffer; k is about 32).
        const bi = new Int32Array(k).fill(-1);
        const bv = new Float32Array(k).fill(-1);
        let min = -1;
        for (let f = 0; f < nFeatures; f++) {
          const v = pre[off + f];
          if (v <= min) continue;
          let j = k - 1;
          while (j > 0 && bv[j - 1] < v) {
            bv[j] = bv[j - 1];
            bi[j] = bi[j - 1];
            j--;
          }
          bv[j] = v;
          bi[j] = f;
          min = bv[k - 1];
        }
        for (let j = 0; j < k; j++) if (bv[j] > 0) { idx.push(bi[j]); val.push(bv[j]); }
      } else {
        for (let f = 0; f < nFeatures; f++) {
          const v = pre[off + f];
          if (v > thr) { idx.push(f); val.push(v); }
        }
      }
      rows[t] = { idx: Int32Array.from(idx), val: Float32Array.from(val) };
    }
    return rows;
  }

  encode(source, options) {
    return this._serial(() => this._encode(source, options));
  }

  // `pixels` is the model's input, [3, S, S] float32, already normalised (tests pass it directly).
  encodePixels(pixels, options) {
    return this._serial(() => this._encode(null, options, pixels));
  }

  async _encode(source, { readout = true } = {}, given = null) {
    const ort = this.ort;
    const t0 = now();
    const pixels = given || this.pixels(source);
    const t1 = now();
    const { hidden } = await this.front.run({ pixel_values: new ort.Tensor("float32", pixels, [1, 3, this.size, this.size]) });
    const hid = hidden.data instanceof Float32Array ? hidden.data : await hidden.getData();
    const patches = hid.subarray(this.prefix * this.d);
    const { pre } = await this.sae.run({ x: new ort.Tensor("float32", patches.slice(), [this.nTok, this.d]) });
    const preData = pre.data instanceof Float32Array ? pre.data : await pre.getData();
    const t2 = now();
    const rows = this.activate(preData);
    const totals = new Float32Array(this.nFeatures);
    const peaks = new Float32Array(this.nFeatures);
    for (const row of rows) {
      for (let j = 0; j < row.idx.length; j++) {
        const f = row.idx[j];
        totals[f] += row.val[j];
        if (row.val[j] > peaks[f]) peaks[f] = row.val[j];
      }
    }
    const t3 = now();
    let base = null;
    let seen = null;
    if (this.wordsMode && readout) seen = await this._see(hid);
    else if (readout) base = await this.runBack(hid);
    const t4 = now();
    const state = { hidden: hid.slice(), rows, totals, peaks, base, seen };
    state.timing = { preprocess: t1 - t0, model: t2 - t1, features: t3 - t2, readout: t4 - t3, total: t4 - t0 };
    return state;
  }

  // Always called inside _serial, after load queued _startBack, so the session is there.
  async runBack(hidden) {
    const ort = this.ort;
    if (!this.back) await this.backReady;
    const tokens = this.prefix + this.nTok;
    const { final } = await this.back.run({ hidden: new ort.Tensor("float32", hidden, [1, tokens, this.d]) });
    const data = final.data instanceof Float32Array ? final.data : await final.getData();
    return data.slice(this.prefix * this.d);
  }

  // Map of one feature over the patches.
  featureMap(state, id) {
    if (state.maps) return state.maps.get(id) || new Float32Array(this.nTok); // a baked answer
    const out = new Float32Array(this.nTok);
    state.rows.forEach((row, t) => {
      const j = row.idx.indexOf(id);
      if (j >= 0) out[t] = row.val[j];
    });
    return out;
  }

  // ---------------------------------------------------------------- words readout

  // Hidden state (and u) through the words graph: the embedding, and d(e.u)/d(hidden) when u is given.
  async _wordsRun(hidden, u = null) {
    const ort = this.ort;
    const tokens = this.prefix + this.nTok;
    const feeds = {
      hidden: new ort.Tensor("float32", hidden, [1, tokens, this.d]),
      u: new ort.Tensor("float32", u || new Float32Array(this.textDim), [1, this.textDim]),
    };
    const out = await this.words.run(feeds, u ? ["embed", "grad"] : ["embed"]);
    const embed = out.embed.data instanceof Float32Array ? out.embed.data : await out.embed.getData();
    let grad = null;
    if (u) grad = out.grad.data instanceof Float32Array ? out.grad.data : await out.grad.getData();
    return { embed: embed.slice(), grad };
  }

  // The words: log-probabilities over the vocabulary at the model's logit scale.
  wordsFrom(embed) {
    const { textDim: D, text } = this;
    const n = this.vocab.length;
    let norm = 0;
    for (let i = 0; i < D; i++) norm += embed[i] * embed[i];
    norm = Math.sqrt(norm) || 1;
    const logits = new Float32Array(n);
    let max = -Infinity;
    for (let w = 0; w < n; w++) {
      let dot = 0;
      const off = w * D;
      for (let i = 0; i < D; i++) dot += text[off + i] * embed[i];
      logits[w] = (this.logitScale * dot) / norm;
      if (logits[w] > max) max = logits[w];
    }
    let sum = 0;
    for (let w = 0; w < n; w++) sum += Math.exp(logits[w] - max);
    const lse = max + Math.log(sum);
    const logp = new Float32Array(n);
    const prob = new Float32Array(n);
    for (let w = 0; w < n; w++) {
      logp[w] = logits[w] - lse;
      prob[w] = Math.exp(logp[w]);
    }
    return { embed, norm, logp, prob };
  }

  async _see(hidden) {
    const { embed } = await this._wordsRun(hidden);
    return this.wordsFrom(embed);
  }

  // d log p(word) / d e, for e before normalisation: s (t_w - sum_v p_v t_v), projected off e, over |e|.
  _uFor(seen, word) {
    const { textDim: D, text } = this;
    const n = this.vocab.length;
    const a = new Float32Array(D);
    for (let i = 0; i < D; i++) a[i] = text[word * D + i];
    for (let v = 0; v < n; v++) {
      const p = seen.prob[v];
      if (p < 1e-7) continue;
      const off = v * D;
      for (let i = 0; i < D; i++) a[i] -= p * text[off + i];
    }
    const nrm = seen.norm;
    let along = 0;
    for (let i = 0; i < D; i++) along += (a[i] * seen.embed[i]) / nrm;
    const u = new Float32Array(D);
    for (let i = 0; i < D; i++) u[i] = (this.logitScale * (a[i] - (along * seen.embed[i]) / nrm)) / nrm;
    return u;
  }

  why(state, word) {
    return this._serial(() => this._why(state, word));
  }

  // Every feature firing on the picture, scored by how much it pushes `word` up (positive) or down.
  async _why(state, word) {
    const t0 = now();
    const { d, nTok, prefix } = this;
    const { grad } = await this._wordsRun(state.hidden, this._uFor(state.seen, word));
    const score = new Map();
    const patches = new Float32Array(nTok); // where on the picture the features push the word up
    state.rows.forEach((row, t) => {
      const g = (prefix + t) * d;
      for (let j = 0; j < row.idx.length; j++) {
        const f = row.idx[j];
        const dir = f * d;
        let dot = 0;
        for (let i = 0; i < d; i++) dot += grad[g + i] * this.decoder[dir + i];
        const s = row.val[j] * dot;
        score.set(f, (score.get(f) || 0) + s);
        patches[t] += s;
      }
    });
    const ranked = [...score.entries()].map(([id, s]) => ({ id, score: s })).sort((a, b) => b.score - a.score);
    return { word, ranked, patches, ms: now() - t0 };
  }

  // ---------------------------------------------------------------- any word (ii export-text)

  get readsAnyWord() {
    return Boolean(this.wordsMode && this.pack.files.text_encoder && this.pack.files.tokenizer);
  }

  // The text half and its tokenizer, fetched the first time a word off the list is asked for.
  loadText(onProgress = () => {}) {
    if (!this._text) {
      this._text = (async () => {
        const { Tokenizer } = await import("./text.js");
        const files = this.pack.files;
        const digests = this.pack.digests || {};
        const [bytes, spec] = await Promise.all([
          fetchFile(this.base, files.text_encoder, digests.text_encoder || "", (seen, total) => onProgress(seen, total)),
          fetch(this.base + files.tokenizer).then((r) => r.json()),
        ]);
        this.tokenizer = new Tokenizer(spec);
        this.textSession = await this._serial(() => this.ort.InferenceSession.create(bytes, this._options("wasm")));
      })();
      this._text.catch(() => (this._text = null));
    }
    return this._text;
  }

  // Add a word (or a short phrase) to the list; returns its index. Probabilities are then over
  // the longer list, so callers read them again with wordsFrom(state.seen.embed).
  async addWord(word, onProgress) {
    const known = this.vocab.indexOf(word);
    if (known >= 0) return known;
    await this.loadText(onProgress);
    const { promptFor } = await import("./text.js");
    const ids = this.tokenizer.encode(promptFor(this.pack.words.prompt, word));
    const vec = await this._serial(async () => {
      const out = await this.textSession.run({ input_ids: new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]) });
      return out.embed.data instanceof Float32Array ? out.embed.data : await out.embed.getData();
    });
    const text = new Float32Array(this.text.length + this.textDim);
    text.set(this.text);
    text.set(vec.subarray(0, this.textDim), this.text.length);
    this.text = text;
    this.vocab.push(word);
    return this.vocab.length - 1;
  }

  toward(state, word, skip = new Set()) {
    return this._serial(() => this._toward(state, word, skip));
  }

  // Every feature, scored by how much writing it onto every patch at its training peak would
  // raise `word` (first order: the patch-summed gradient along its direction, times its peak).
  // Features that never fired in training (no peak) and `skip` are left out.
  async _toward(state, word, skip) {
    const { d, nTok, prefix, nFeatures } = this;
    const { grad } = await this._wordsRun(state.hidden, this._uFor(state.seen, word));
    const g = new Float32Array(d);
    for (let t = 0; t < nTok; t++) {
      const off = (prefix + t) * d;
      for (let i = 0; i < d; i++) g[i] += grad[off + i];
    }
    const ranked = [];
    for (let f = 0; f < nFeatures; f++) {
      const peak = this.features[f].max || 0;
      if (!peak || skip.has(f)) continue;
      let dot = 0;
      const dir = f * d;
      for (let i = 0; i < d; i++) dot += g[i] * this.decoder[dir + i];
      ranked.push({ id: f, score: dot * peak });
    }
    ranked.sort((a, b) => b.score - a.score);
    return { word, ranked };
  }

  see(state, edits) {
    return this._serial(async () => {
      const t0 = now();
      const seen = await this._see(this.applyEdits(state, edits));
      seen.ms = now() - t0;
      return seen;
    });
  }

  // The words with every feature (but `skip`) taken off the given patches: what the model loses
  // when that part of the picture loses its features.
  clearPatches(state, patches, skip = new Set()) {
    return this._serial(async () => {
      const { d, prefix } = this;
      const hidden = state.hidden.slice();
      for (const t of patches) {
        const row = state.rows[t];
        const off = (prefix + t) * d;
        for (let j = 0; j < row.idx.length; j++) {
          const f = row.idx[j];
          if (skip.has(f)) continue;
          const v = row.val[j];
          const dir = f * d;
          for (let i = 0; i < d; i++) hidden[off + i] -= v * this.decoder[dir + i];
        }
      }
      return this._see(hidden);
    });
  }

  // edits: [{id, kind, value}] applied together on the patch tokens (x' = x + (f' - f) W_dec).
  applyEdits(state, edits) {
    const { d, nTok, prefix } = this;
    const hidden = state.hidden.slice();
    const maps = edits.map((e) => this.featureMap(state, e.id));
    for (let t = 0; t < nTok; t++) {
      const off = (prefix + t) * d;
      edits.forEach((e, n) => {
        const current = maps[n][t];
        let target;
        if (e.kind === "ablate") target = 0;
        else if (e.kind === "scale") target = current * e.value;
        else if (e.kind === "clamp") target = e.value;
        else target = current + e.value;
        const delta = target - current;
        if (delta === 0) return;
        const dir = e.id * d;
        for (let i = 0; i < d; i++) hidden[off + i] += delta * this.decoder[dir + i];
      });
    }
    return hidden;
  }

  // The playground edit on the patch tokens, then the rest of the network.
  edit(state, ids, kind, value) {
    return this._serial(() => this._edit(state, ids, kind, value));
  }

  async _edit(state, ids, kind, value) {
    const t0 = now();
    const { d, nTok, prefix } = this;
    if (!state.base) state.base = await this.runBack(state.hidden);
    const hidden = state.hidden.slice();
    const maps = ids.map((id) => this.featureMap(state, id));
    for (let t = 0; t < nTok; t++) {
      const off = (prefix + t) * d;
      for (let n = 0; n < ids.length; n++) {
        const current = maps[n][t];
        let target;
        if (kind === "ablate") target = 0;
        else if (kind === "scale") target = current * value;
        else if (kind === "clamp") target = value;
        else target = current + value;
        const delta = target - current;
        if (delta === 0) continue;
        const dir = ids[n] * d;
        for (let i = 0; i < d; i++) hidden[off + i] += delta * this.decoder[dir + i];
      }
    }
    const t1 = now();
    const edited = await this.runBack(hidden);
    const t2 = now();
    const shift = new Float32Array(nTok);
    let sum = 0;
    let max = 0;
    let sq = 0;
    for (let t = 0; t < nTok; t++) {
      let dot = 0, na = 0, nb = 0;
      const off = t * d;
      for (let i = 0; i < d; i++) {
        const a = state.base[off + i];
        const b = edited[off + i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        sq += (a - b) * (a - b);
      }
      const v = 1 - dot / (Math.sqrt(na * nb) || 1);
      shift[t] = v;
      sum += v;
      if (v > max) max = v;
    }
    return {
      shift,
      mean: sum / nTok,
      max,
      lossChange: sq / (nTok * d),
      timing: { edit: t1 - t0, model: t2 - t1, total: now() - t0 },
    };
  }
}
