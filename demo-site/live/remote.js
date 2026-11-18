// ii live engine, hosted: the same interface as engine.js, but the model runs on the
// playground backend (ii serve-live locally, ii deploy-live on Modal) through API v1
// (docs/PLAYGROUND-API.md). The page sends the picture already resized to the model's
// input as raw RGB, gets every patch's features back, and asks the server for edits.
// Labels, badges and sample pictures still come from the static pack.

export const KINDS = ["ablate", "scale", "clamp"];

const now = () => performance.now();

class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export class RemoteEngine {
  constructor(baseUrl, apiUrl) {
    this.base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
    this.api = apiUrl.replace(/\/+$/, "");
    this.backend = "remote";
    this.timings = {};
    this._queue = Promise.resolve();
    this._waiting = null;
  }

  _serial(fn) {
    const run = this._queue.then(fn, fn);
    this._queue = run.catch(() => {});
    return run;
  }

  async _call(path, init = {}) {
    let response;
    try {
      response = await fetch(this.api + path, { mode: "cors", ...init });
    } catch (err) {
      throw new ApiError(0, "unreachable", `the model server at ${this.api} did not answer (${err.message})`);
    }
    if (response.ok) return response.json();
    let code = "error";
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      code = body.error?.code || code;
      message = body.error?.message || message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, code, message);
  }

  // onProgress({ phase }): "connect" while the server wakes up (a cold start takes a few seconds).
  async load({ onProgress = () => {} } = {}) {
    const t0 = now();
    const packId = this.base.replace(/\/+$/, "").split("/").pop();
    onProgress({ phase: "connect" });
    const waking = this._call("/v1/health"); // wakes the server while the pack loads
    waking.catch(() => {}); // reported below
    const pack = await (await fetch(this.base + "pack.json")).json();
    const [features, health] = await Promise.all([fetch(this.base + pack.files.features).then((r) => r.json()), waking]);
    if (pack.format !== "ii-live-pack" || pack.version !== 1) throw new Error("this pack is not an ii live pack v1");
    if (health.api !== "ii-playground" || health.version !== 1) throw new Error("the model server does not speak playground API v1");
    if (!health.packs.includes(packId)) throw new Error(`the model server does not have the ${packId} pack`);
    this.packId = packId;
    this.pack = pack;
    this.features = features;
    const info = await this._call(`/v1/packs/${encodeURIComponent(packId)}`);
    this.device = info.device;
    this.nFeatures = info.n_features;
    this.gridH = info.grid.h;
    this.gridW = info.grid.w;
    this.nTok = this.gridH * this.gridW;
    this.size = info.image_size;
    this.timings.connect = now() - t0;
    this.timings.ready = now() - t0;
    onProgress({ phase: "ready" });
    return this;
  }

  // The picture squashed to the model's square input, as raw RGB bytes (the hf adapter's resize).
  rgb(source) {
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
    const out = new Uint8Array(s * s * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      out[j] = rgba[i];
      out[j + 1] = rgba[i + 1];
      out[j + 2] = rgba[i + 2];
    }
    return out;
  }

  encode(source, options) {
    return this._serial(() => this._encode(this.rgb(source)));
  }

  encodeRgb(rgb) {
    return this._serial(() => this._encode(rgb));
  }

  async _read(rgb) {
    return this._call(`/v1/packs/${encodeURIComponent(this.packId)}/read`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: rgb,
    });
  }

  async _encode(rgb) {
    const t0 = now();
    const body = await this._read(rgb);
    const t1 = now();
    const { offsets, ids, values } = body.patches;
    const rows = new Array(this.nTok);
    const totals = new Float32Array(this.nFeatures);
    const peaks = new Float32Array(this.nFeatures);
    for (let t = 0; t < this.nTok; t++) {
      const a = offsets[t];
      const b = offsets[t + 1];
      const idx = Int32Array.from(ids.slice(a, b));
      const val = Float32Array.from(values.slice(a, b));
      rows[t] = { idx, val };
      for (let j = 0; j < idx.length; j++) {
        totals[idx[j]] += val[j];
        if (val[j] > peaks[idx[j]]) peaks[idx[j]] = val[j];
      }
    }
    const state = { imageId: body.image_id, rgb, rows, totals, peaks };
    state.timing = { network: t1 - t0 - body.timing_ms.total, model: body.timing_ms.total, total: now() - t0 };
    return state;
  }

  featureMap(state, id) {
    const out = new Float32Array(this.nTok);
    state.rows.forEach((row, t) => {
      const j = row.idx.indexOf(id);
      if (j >= 0) out[t] = row.val[j];
    });
    return out;
  }

  // Edits that pile up while one is in flight (a dragged slider) collapse into the latest.
  edit(state, ids, kind, value) {
    const args = { state, ids, kind, value };
    if (this._waiting) {
      this._waiting.args = args;
      return this._waiting.promise;
    }
    const slot = { args };
    this._waiting = slot;
    slot.promise = this._serial(() => {
      this._waiting = null;
      const a = slot.args;
      return this._edit(a.state, a.ids, a.kind, a.value);
    });
    return slot.promise;
  }

  async _edit(state, ids, kind, value) {
    const t0 = now();
    const request = () =>
      this._call(`/v1/packs/${encodeURIComponent(this.packId)}/steer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ image_id: state.imageId, features: ids.map(Number), kind, value: kind === "ablate" ? 0 : Number(value) }),
      });
    let body;
    try {
      body = await request();
    } catch (err) {
      // Another server, or this one forgot the picture: read it again, then retry once.
      if (err.code !== "unknown_image") throw err;
      const again = await this._read(state.rgb);
      state.imageId = again.image_id;
      body = await request();
    }
    return {
      shift: Float32Array.from(body.shift),
      mean: body.mean,
      max: body.max,
      lossChange: body.loss_change,
      timing: { model: body.timing_ms.total, total: now() - t0 },
    };
  }
}
