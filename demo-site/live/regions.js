// Concept regions: the features firing on one picture, grouped into a few concepts that each
// cover part of it. Non-negative matrix factorisation (Lee and Seung's multiplicative updates)
// of the patch × feature activations, as in Goodfire's Paint With Ember. Plain JavaScript on
// the activations the page already has; no model call. Exploration only.

// A small seeded generator, so the same picture always splits the same way.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// X (T × C, row-major) ≈ W (T × k) H (k × C), all non-negative.
export function nmf(X, T, C, k, { iters = 200, seed = 1 } = {}) {
  const r = rng(seed);
  let mean = 0;
  for (let i = 0; i < X.length; i++) mean += X[i];
  mean = Math.sqrt(mean / X.length / k) || 1;
  const W = Float32Array.from({ length: T * k }, () => r() * mean + 1e-4);
  const H = Float32Array.from({ length: k * C }, () => r() * mean + 1e-4);
  const eps = 1e-9;
  const XHt = new Float32Array(T * k);
  const HHt = new Float32Array(k * k);
  const WtX = new Float32Array(k * C);
  const WtW = new Float32Array(k * k);
  for (let it = 0; it < iters; it++) {
    // H ← H ∘ (WᵀX) / (WᵀW H)
    WtX.fill(0);
    WtW.fill(0);
    for (let t = 0; t < T; t++) {
      const w = t * k;
      const x = t * C;
      for (let a = 0; a < k; a++) {
        const wa = W[w + a];
        if (!wa) continue;
        const o = a * C;
        for (let c = 0; c < C; c++) WtX[o + c] += wa * X[x + c];
        for (let b = 0; b < k; b++) WtW[a * k + b] += wa * W[w + b];
      }
    }
    for (let a = 0; a < k; a++) {
      for (let c = 0; c < C; c++) {
        let den = 0;
        for (let b = 0; b < k; b++) den += WtW[a * k + b] * H[b * C + c];
        H[a * C + c] *= WtX[a * C + c] / (den + eps);
      }
    }
    // W ← W ∘ (X Hᵀ) / (W H Hᵀ)
    XHt.fill(0);
    HHt.fill(0);
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += H[a * C + c] * H[b * C + c];
        HHt[a * k + b] = s;
      }
    }
    for (let t = 0; t < T; t++) {
      const x = t * C;
      for (let a = 0; a < k; a++) {
        let s = 0;
        const o = a * C;
        for (let c = 0; c < C; c++) s += X[x + c] * H[o + c];
        XHt[t * k + a] = s;
      }
    }
    for (let t = 0; t < T; t++) {
      for (let a = 0; a < k; a++) {
        let den = 0;
        for (let b = 0; b < k; b++) den += W[t * k + b] * HHt[b * k + a];
        W[t * k + a] *= XHt[t * k + a] / (den + eps);
      }
    }
  }
  return { W, H };
}

// rows: the page's sparse activations per patch ({idx, val}); peaks: each feature's training
// peak, so a feature counts by how strongly it fires for itself, not by its raw scale.
// skip: features left out (the always-on ones). Returns up to k regions, largest first:
// {patches: Int32Array of patch indices, features: [{id, weight}] (weight sums to 1)}, and
// `owner` (T entries, region index or -1 for patches no concept covers). gridW: patches per row,
// to smooth each concept's map over its neighbours (0 leaves it as is).
export function conceptRegions(rows, peaks, { k = 8, skip = new Set(), minPatches = 3, iters = 200, gridW = 0 } = {}) {
  const T = rows.length;
  const col = new Map();
  for (const row of rows) for (const f of row.idx) if (!skip.has(f) && !col.has(f)) col.set(f, col.size);
  const ids = [...col.keys()];
  const C = ids.length;
  if (!C) return { regions: [], owner: new Int32Array(T).fill(-1) };
  const X = new Float32Array(T * C);
  rows.forEach((row, t) => {
    for (let j = 0; j < row.idx.length; j++) {
      const c = col.get(row.idx[j]);
      if (c != null) X[t * C + c] = Math.min(1, row.val[j] / (peaks[row.idx[j]] || row.val[j] || 1));
    }
  });
  k = Math.min(k, T, C);
  const { W, H } = nmf(X, T, C, k, { iters });
  // How much of each patch each concept explains: W[t,a] times the size of concept a.
  const size = new Float32Array(k);
  for (let a = 0; a < k; a++) {
    let s = 0;
    for (let c = 0; c < C; c++) s += H[a * C + c] * H[a * C + c];
    size[a] = Math.sqrt(s);
  }
  // Each concept's map, smoothed over neighbouring patches so regions come out whole, not speckled.
  const raw = new Float32Array(T * k);
  for (let t = 0; t < T; t++) for (let a = 0; a < k; a++) raw[t * k + a] = W[t * k + a] * size[a];
  const map = gridW ? smooth(raw, T, k, gridW) : raw;
  const strength = new Float32Array(T);
  const best = new Int32Array(T).fill(-1);
  let top = 0;
  for (let t = 0; t < T; t++) {
    let bv = 0;
    for (let a = 0; a < k; a++) {
      const v = map[t * k + a];
      if (v > bv) {
        bv = v;
        best[t] = a;
      }
    }
    strength[t] = bv;
    if (bv > top) top = bv;
  }
  // Patches where no concept is strong (the always-on background, plain sky) stay clear.
  for (let t = 0; t < T; t++) if (strength[t] < 0.2 * top) best[t] = -1;
  const groups = Array.from({ length: k }, () => []);
  for (let t = 0; t < T; t++) if (best[t] >= 0) groups[best[t]].push(t);
  const kept = groups
    .map((patches, a) => ({ a, patches }))
    .filter((g) => g.patches.length >= minPatches)
    .sort((x, y) => y.patches.length - x.patches.length);
  const owner = new Int32Array(T).fill(-1);
  const regions = kept.map((g, n) => {
    for (const t of g.patches) owner[t] = n;
    let sum = 0;
    for (let c = 0; c < C; c++) sum += H[g.a * C + c];
    const features = [];
    for (let c = 0; c < C; c++) if (H[g.a * C + c] > 0) features.push({ id: ids[c], weight: H[g.a * C + c] / (sum || 1) });
    features.sort((x, y) => y.weight - x.weight);
    return { patches: Int32Array.from(g.patches), features };
  });
  return { regions, owner };
}

// 3 × 3 binomial blur of k maps laid out on a grid gridW wide.
function smooth(maps, T, k, gridW) {
  const H = T / gridW;
  const out = new Float32Array(T * k);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < gridW; x++) {
      const t = y * gridW + x;
      for (let a = 0; a < k; a++) {
        let s = 0;
        let wsum = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy;
            const xx = x + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= gridW) continue;
            const w = (dx ? 1 : 2) * (dy ? 1 : 2);
            s += w * maps[(yy * gridW + xx) * k + a];
            wsum += w;
          }
        }
        out[t * k + a] = s / wsum;
      }
    }
  }
  return out;
}
