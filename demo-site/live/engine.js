// ii live engine: runs a pack (model split at the SAE's layer + the SAE) in the browser.
//
// A pack is what `ii export-live` writes (pack.json says which files). The engine:
//   encode(image)   pixels -> hidden at the SAE's layer -> every feature on every patch,
//                   plus the model's clean final tokens (the playground's readout).
//   edit(ids, ...)  the playground edit, x' = x + sum_i (f'_i - f_i) W_dec[:, i], on the
//                   patch tokens, then the rest of the network, then the per-patch
//                   cosine distance between clean and edited final tokens.
// Same maths as `steer.steer_edit` and the local playground. Exploration only: nothing
// here can change a feature's badge.

const ORT_VERSION = "1.31.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE_NAME = "ii-live-v1";

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

  // onProgress({ phase, loaded, total, cached })
  async load({ onProgress = () => {}, preferGpu = true } = {}) {
    const t0 = now();
    const pack = await (await fetch(this.base + "pack.json")).json();
    if (pack.format !== "ii-live-pack" || pack.version !== 1) throw new Error("this pack is not an ii live pack v1");
    this.pack = pack;
    const { ort, backend } = await loadOrt(preferGpu);
    this.ort = ort;
    this.backend = backend;
    const files = pack.files;
    const sha = (pack.model.sha || "").slice(0, 12);
    const want = ["front", "back", "sae", "decoder"];
    const sizes = {};
    const loaded = {};
    let cachedAll = true;
    const report = () => {
      const got = Object.values(loaded).reduce((a, b) => a + b, 0);
      const total = Object.values(sizes).reduce((a, b) => a + b, 0);
      onProgress({ phase: "download", loaded: got, total, cached: cachedAll });
    };
    const [bytes, features] = await Promise.all([
      Promise.all(
        want.map((name) =>
          fetchBytes(this.base + files[name], `${this.base}${files[name]}?v=${sha}`, (seen, total, cached) => {
            loaded[name] = seen;
            sizes[name] = total || seen;
            if (!cached) cachedAll = false;
            report();
          }),
        ),
      ),
      fetch(this.base + files.features).then((r) => r.json()),
    ]);
    this.timings.download = now() - t0;
    onProgress({ phase: "compile" });
    const t1 = now();
    const options = {
      executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
    };
    const [front, back, sae] = await Promise.all([
      ort.InferenceSession.create(bytes[0], options),
      ort.InferenceSession.create(bytes[1], options),
      ort.InferenceSession.create(bytes[2], options),
    ]);
    this.front = front;
    this.back = back;
    this.sae = sae;
    const decBits = new Uint16Array(bytes[3].buffer, bytes[3].byteOffset, bytes[3].byteLength / 2);
    this.decoder = halfToFloat(decBits); // [n_features, d]
    this.features = features;
    this.nFeatures = pack.sae.n_features;
    this.d = pack.model.d_model;
    this.gridH = pack.model.grid.h;
    this.gridW = pack.model.grid.w;
    this.nTok = this.gridH * this.gridW;
    this.prefix = pack.model.prefix_tokens;
    this.size = pack.model.image_size;
    this.timings.compile = now() - t1;
    // One warm-up pass, so the first real image is not paying for shader compilation.
    const t2 = now();
    await this.encode(this.blankCanvas());
    this.timings.warmup = now() - t2;
    this.timings.ready = now() - t0;
    onProgress({ phase: "ready" });
    return this;
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
    g.drawImage(source, 0, 0, s, s);
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
    if (readout) base = await this.runBack(hid);
    const t4 = now();
    const state = { hidden: hid.slice(), rows, totals, peaks, base };
    state.timing = { preprocess: t1 - t0, model: t2 - t1, features: t3 - t2, readout: t4 - t3, total: t4 - t0 };
    return state;
  }

  async runBack(hidden) {
    const ort = this.ort;
    const tokens = this.prefix + this.nTok;
    const { final } = await this.back.run({ hidden: new ort.Tensor("float32", hidden, [1, tokens, this.d]) });
    const data = final.data instanceof Float32Array ? final.data : await final.getData();
    return data.slice(this.prefix * this.d);
  }

  // Map of one feature over the patches.
  featureMap(state, id) {
    const out = new Float32Array(this.nTok);
    state.rows.forEach((row, t) => {
      const j = row.idx.indexOf(id);
      if (j >= 0) out[t] = row.val[j];
    });
    return out;
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
