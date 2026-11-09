const state = {
  run: null,
  clip: null,
  clipId: null,
  frame: 0,
  featureId: null,
  playing: false,
  timer: null,
  token: 0,
};

const app = document.getElementById("app");

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
  const kind = feature.badge === "real" ? "real" : "unverified";
  return el("span", { class: `badge badge-${kind}`, "data-testid": "badge" }, kind);
}

async function boot() {
  try {
    const response = await fetch("/api/run", { cache: "no-store" });
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
  const sae = run.sae || {};
  const saeEl = document.getElementById("sae-summary");
  const usable = sae.usable === true;
  saeEl.dataset.usable = usable ? "true" : "false";
  saeEl.textContent = `${usable ? "usable" : "not usable"} · ev_heldout ${fmtUnit(sae.ev_heldout)} · loss_recovered ${fmtUnit(sae.loss_recovered)}`;
  document.title = `${run.run_id} · physint`;
}

function route() {
  stop();
  const raw = (location.hash || "#/").replace(/^#/, "");
  const [path, query = ""] = raw.split("?");
  const params = new URLSearchParams(query);
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

function renderError(message) {
  app.replaceChildren(el("p", { class: "error" }, message));
}

function renderIndex() {
  const features = state.run.features;
  const real = features.filter((feature) => feature.badge === "real").length;
  const list = el("ul", { class: "index-list" });
  for (const feature of features) {
    const link = el(
      "a",
      {
        href: `#/features/${feature.id}`,
        "data-testid": "feature-index-link",
        "data-feature-id": feature.id,
        "data-label": feature.label || "",
      },
      [
        el("span", {}, [el("span", { class: "index-id" }, String(feature.id).padStart(2, "0")), displayLabel(feature)]),
        badge(feature),
      ],
    );
    list.append(el("li", {}, link));
  }
  app.replaceChildren(
    el("section", { class: "feature-index", "data-testid": "feature-index" }, [
      el("h1", {}, "Features"),
      el("p", { class: "kicker", "data-testid": "real-count" }, `${real} real · ${features.length - real} unverified`),
      list,
    ]),
  );
  document.title = `features · ${state.run.run_id} · physint`;
}

function showFeature(id) {
  const feature = state.run.features.find((row) => row.id === id);
  if (!feature) {
    renderError(`Feature ${id} is not in the scoreboard.`);
    return;
  }
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
  app.replaceChildren(
    el("article", {
      class: "feature-page",
      "data-testid": "feature-page",
      "data-feature-id": feature.id,
      "data-label": feature.label || "",
      "data-badge": feature.badge,
    }, [
      el("p", { class: "kicker" }, `Feature ${feature.id}`),
      el("div", { class: "feature-title" }, [el("h1", {}, displayLabel(feature)), badge(feature)]),
      feature.label_model ? el("p", { class: "label-model" }, feature.label_model) : null,
      el("dl", { class: "score", "data-testid": "scoreboard" }, [
        scoreItem("distinct_clips_top20", String(feature.distinct_clips_top20), feature.distinct_clips_top20),
        scoreItem("label_holds_heldout", fmtFlag(feature.label_holds_heldout), feature.label_holds_heldout),
        scoreItem("ablation_effect", fmtEffect(feature.ablation_effect), feature.ablation_effect),
        scoreVersus(feature),
        scoreItem("absent_in_random_init", fmtFlag(feature.absent_in_random_init), feature.absent_in_random_init),
      ]),
      el("h2", { class: "panel-title" }, "Top clips"),
      el("table", { class: "top-table" }, [
        el("thead", {}, el("tr", {}, ["clip", "split", "frame", "window / t", "activation"].map((label) => el("th", {}, label)))),
        el("tbody", {}, rows),
      ]),
    ]),
  );
  document.title = `${displayLabel(feature)} · physint`;
}

function scoreItem(name, text, raw) {
  const shown = raw === null || raw === undefined ? "null" : String(raw);
  return el("div", {}, [
    el("dt", {}, name),
    el("dd", { "data-testid": name, "data-value": shown }, text),
  ]);
}

function scoreVersus(feature) {
  return el("div", {}, [
    el("dt", {}, "random_percentile"),
    el("dd", {}, [
      el("span", { "data-testid": "random_percentile", "data-value": String(feature.random_percentile) }, fmtPercentile(feature.random_percentile)),
      " ",
      el("span", { class: "of" }, "of "),
      el("span", { "data-testid": "n_random", "data-value": String(feature.n_random) }, String(feature.n_random)),
    ]),
  ]);
}

async function showClip(clipId, frame, featureParam) {
  const token = ++state.token;
  if (!state.clip || state.clipId !== clipId) {
    const response = await fetch(`/api/clips/${encodeURIComponent(clipId)}`, { cache: "no-store" });
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
  const heatmap = el("div", { class: "heatmap", "data-testid": "heatmap" });
  const stage = el("div", { class: "stage", "data-testid": "stage" }, [image, heatmap]);
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
          el("span", { class: "split-tag" }, clip.split || ""),
        ]),
        stage,
        el("div", { class: "transport" }, [
          play,
          scrubber,
          el("p", { class: "frame-readout", "data-testid": "frame-readout" }),
        ]),
        el("p", { class: "heat-caption", "data-testid": "heatmap-label" }),
      ]),
      el("section", {}, [
        el("h2", { class: "panel-title" }, "At this frame"),
        el("ol", { class: "feature-list", "data-testid": "frame-features" }),
      ]),
    ]),
  );
  renderFrame();
  document.querySelector(".clip-list a[aria-current='page']")?.scrollIntoView({ block: "nearest" });
}

function setFrame(frame) {
  state.frame = clampFrame(frame);
  renderFrame();
  const feature = state.featureId === null || Number.isNaN(state.featureId) ? "" : `&feature=${state.featureId}`;
  const hash = `#/clips/${encodeURIComponent(state.clipId)}?frame=${state.frame}${feature}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
}

function renderFrame() {
  const clip = state.clip;
  const frame = clip.frames[state.frame];
  const stage = document.querySelector("[data-testid=stage]");
  const image = document.querySelector("[data-testid=frame-image]");
  const message = document.querySelector("[data-testid=pixels-message]");
  const readout = document.querySelector("[data-testid=frame-readout]");
  const scrubber = document.getElementById("scrubber");
  const list = document.querySelector("[data-testid=frame-features]");
  if (!frame || !list) return;

  scrubber.value = String(state.frame);
  readout.textContent = `frame ${state.frame} · t ${frame.t} · window ${frame.window}`;
  list.dataset.frame = String(state.frame);
  image.alt = `${clip.clip_id} frame ${state.frame}`;

  if (clip.width && clip.height) stage.style.aspectRatio = `${clip.width} / ${clip.height}`;
  if (clip.pixels === "ok") {
    stage.classList.remove("is-empty");
    message.hidden = true;
    const src = `/api/clips/${encodeURIComponent(clip.clip_id)}/frames/${state.frame}.png`;
    if (image.getAttribute("src") !== src) image.src = src;
    image.hidden = false;
  } else {
    stage.classList.add("is-empty");
    image.hidden = true;
    image.removeAttribute("src");
    message.hidden = false;
    message.textContent = clip.message || "Clip pixels are not available.";
  }

  const active = frame.features.find((feature) => feature.id === state.featureId) || frame.features[0] || null;
  list.replaceChildren(...(frame.features.length ? frame.features.map((feature) => renderFeatureRow(feature, active)) : [
    el("li", { class: "empty" }, "No feature fired on this frame."),
  ]));
  renderHeat(active);
  const caption = document.querySelector("[data-testid=heatmap-label]");
  caption.replaceChildren();
  if (active) {
    caption.append("heatmap · ", el("strong", {}, displayLabel(active)));
  } else {
    caption.textContent = "Nothing fired on this frame.";
  }
  document.title = `${clip.clip_id} · frame ${state.frame} · physint`;
}

function renderFeatureRow(feature, active) {
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
    el("div", { class: "activation" }, ["activation ", el("b", { "data-testid": "activation" }, fmtActivation(feature.activation))]),
    el("div", { class: "effects" }, feature.effects.map(renderEffect)),
  );
  row.addEventListener("click", () => {
    state.featureId = feature.id;
    renderFrame();
    const hash = `#/clips/${encodeURIComponent(state.clipId)}?frame=${state.frame}&feature=${feature.id}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
  });
  return row;
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

function renderHeat(feature) {
  const grid = document.querySelector("[data-testid=heatmap]");
  grid.replaceChildren();
  if (!feature || !feature.heatmap || !feature.heatmap.length) return;
  const rows = feature.heatmap;
  const height = rows.length;
  const width = rows[0].length;
  let max = 0;
  for (const row of rows) for (const value of row) if (value > max) max = value;
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
      cell.style.background = alpha > 0 ? `rgba(226, 162, 58, ${alpha.toFixed(3)})` : "transparent";
      grid.append(cell);
    }
  }
}

function togglePlay() {
  state.playing = !state.playing;
  const button = document.getElementById("play");
  button.textContent = state.playing ? "Pause" : "Play";
  button.dataset.state = state.playing ? "playing" : "paused";
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
  const button = document.getElementById("play");
  if (button) {
    button.textContent = "Play";
    button.dataset.state = "paused";
  }
}

boot();
