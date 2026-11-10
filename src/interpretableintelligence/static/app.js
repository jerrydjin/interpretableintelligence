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
  const real = feature.badge === "real";
  const status = !real && typeof feature.status === "string" ? feature.status.trim() : "";
  const shown = real ? "real" : status && status.toLowerCase() !== "real" ? status : "";
  const kind = real ? "real" : shown ? "status" : "unverified";
  return el("span", { class: `badge badge-${kind}`, "data-testid": "badge", "data-kind": kind }, shown || "unverified");
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
  document.title = `${run.run_id} · interpretableintelligence`;
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
  const moments = renderMoments();
  const list = el("ul", { class: "index-list" });
  for (const feature of features) {
    const link = el(
      "a",
      {
        href: `#/features/${feature.id}`,
        "data-testid": "feature-index-link",
        "data-feature-id": feature.id,
        "data-label": feature.label || "",
        "data-candidate": feature.candidate ? "true" : "false",
      },
      [
        el("span", {}, [el("span", { class: "index-id" }, String(feature.id).padStart(2, "0")), displayLabel(feature)]),
        badge(feature),
      ],
    );
    list.append(el("li", {}, link));
  }
  const page = [
    moments,
    el("section", { class: "feature-index", "data-testid": "feature-index" }, [
      el("h1", {}, "Features"),
      el("p", { class: "kicker", "data-testid": "real-count" }, `${real} real · ${features.length - real} unverified`),
      list,
    ]),
  ].filter(Boolean);
  app.replaceChildren(...page);
  document.title = `features · ${state.run.run_id} · interpretableintelligence`;
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
    el("ol", { class: "hardest-list" }, rows),
  ]);
}

async function showFeature(id) {
  const token = ++state.token;
  const response = await fetch(`/api/features/${id}`, { cache: "no-store" });
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
  for (const clip of feature.top_clips) {
    const image = el("img", {
      alt: `${clip.clip_id} frame ${clip.frame}`,
      src: `/api/clips/${encodeURIComponent(clip.clip_id)}/frames/${clip.frame}.png`,
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
      el("p", { class: "kicker" }, `Feature ${feature.id}`),
      el("div", { class: "feature-title" }, [el("h1", {}, displayLabel(feature)), badge(feature)]),
      feature.label_model ? el("p", { class: "label-model" }, feature.label_model) : null,
      feature.viz
        ? el("img", { class: "viz", alt: "", src: `/api/features/${feature.id}/viz`, "data-testid": "feature-viz" })
        : null,
      feature.scored === false
        ? el("p", { class: "kicker", "data-testid": "scoreboard-missing" }, "No scoreboard record. Shown as unverified.")
        : null,
      el("dl", { class: "score", "data-testid": "scoreboard" }, [
        scoreItem("distinct_clips_top20", fmtMaybe(feature.distinct_clips_top20, String), feature.distinct_clips_top20),
        scoreItem("label_holds_heldout", fmtFlag(feature.label_holds_heldout), feature.label_holds_heldout),
        scoreItem("ablation_effect", fmtEffect(feature.ablation_effect), feature.ablation_effect),
        scoreVersus(feature),
        scoreItem("absent_in_random_init", fmtFlag(feature.absent_in_random_init), feature.absent_in_random_init),
      ]),
      el("h2", { class: "panel-title" }, "Top clips"),
      feature.top_clips.length ? strip : null,
      el("table", { class: "top-table" }, [
        el("thead", {}, el("tr", {}, ["clip", "split", "frame", "window / t", "activation"].map((label) => el("th", {}, label)))),
        el("tbody", {}, rows),
      ]),
    ]),
  );
  document.title = `${displayLabel(feature)} · interpretableintelligence`;
}

function fmtMaybe(value, render) {
  if (value === null || value === undefined) return "—";
  return render(value);
}

function scoreItem(name, text, raw) {
  const shown = raw === null || raw === undefined ? "null" : String(raw);
  return el("div", {}, [
    el("dt", {}, name),
    el("dd", { "data-testid": name, "data-value": shown }, text),
  ]);
}

function scoreVersus(feature) {
  const percentile = feature.random_percentile === null || feature.random_percentile === undefined
    ? "—"
    : fmtPercentile(feature.random_percentile);
  const randomCount = feature.n_random === null || feature.n_random === undefined ? "—" : String(feature.n_random);
  return el("div", {}, [
    el("dt", {}, "random_percentile"),
    el("dd", {}, [
      el("span", { "data-testid": "random_percentile", "data-value": feature.random_percentile === null || feature.random_percentile === undefined ? "null" : String(feature.random_percentile) }, percentile),
      " ",
      el("span", { class: "of" }, "of "),
      el("span", { "data-testid": "n_random", "data-value": feature.n_random === null || feature.n_random === undefined ? "null" : String(feature.n_random) }, randomCount),
    ]),
  ]);
}

async function showClip(clipId, frame, featureParam) {
  const token = ++state.token;
  const same = Boolean(state.clip && state.clipId === clipId);
  if (!same) {
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
  if (same && document.querySelector("[data-testid=stage]")) {
    renderFrame();
    return;
  }
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
          el("p", { class: "clip-task", "data-testid": "clip-task", hidden: "" }),
        ]),
        stage,
        el("div", { class: "transport" }, [
          play,
          scrubberWrap,
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
  readout.dataset.frame = String(state.frame);
  readout.dataset.t = tubelet;
  readout.dataset.window = String(frame.window);
  list.dataset.frame = String(state.frame);
  image.alt = `${clip.clip_id} frame ${state.frame}`;
  markMoments(clip);
  showTask(clip, state.frame);

  const useVideo = clip.media === "video" && clip.pixels === "ok";
  stage.dataset.media = useVideo ? "video" : clip.pixels === "ok" ? "frames" : "none";
  if (clip.hw && clip.hw[0] > 0 && clip.hw[1] > 0) stage.style.aspectRatio = `${clip.hw[1]} / ${clip.hw[0]}`;
  else if (clip.width && clip.height) stage.style.aspectRatio = `${clip.width} / ${clip.height}`;
  placeCrop(clip);
  if (useVideo) {
    stage.classList.remove("is-empty");
    message.hidden = true;
    image.hidden = true;
    image.removeAttribute("src");
    video.hidden = false;
    const src = `/api/clips/${encodeURIComponent(clip.clip_id)}/media`;
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
    const src = `/api/clips/${encodeURIComponent(clip.clip_id)}/frames/${state.frame}.png`;
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
    active = frame.features[0] || null;
  }
  list.dataset.feature = active ? String(active.id) : "";
  list.replaceChildren(...(frame.features.length ? frame.features.map((feature) => renderFeatureRow(feature, active)) : [
    el("li", { class: "empty" }, "No feature fired on this frame."),
  ]));
  paintCaption(active);
  renderHeat(active);
  document.title = `${clip.clip_id} · frame ${state.frame} · interpretableintelligence`;
}

function paintCaption(feature) {
  const caption = document.querySelector("[data-testid=heatmap-label]");
  if (!caption) return;
  caption.replaceChildren();
  if (!feature) {
    caption.textContent = "Nothing fired on this frame.";
    return;
  }
  if (feature.inactive) {
    caption.textContent = `feature ${feature.id} not active at this frame`;
    return;
  }
  caption.append("heatmap · ", el("strong", {}, displayLabel(feature)));
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

function placeCrop(clip) {
  const heat = document.querySelector("[data-testid=heatmap]");
  const shade = document.querySelector("[data-testid=crop-shade]");
  if (!heat) return;
  const hw = clip.hw;
  const crop = clip.crop;
  const known = hw && hw[0] > 0 && hw[1] > 0 && crop && crop.width > 0 && crop.height > 0;
  if (!known) {
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

async function renderHeat(feature) {
  const grid = document.querySelector("[data-testid=heatmap]");
  if (!grid) return;
  const token = ++state.heatToken;
  const frame = state.frame;
  grid.replaceChildren();
  if (feature) grid.dataset.featureId = String(feature.id);
  else delete grid.dataset.featureId;
  if (!feature) return;
  if (feature.inactive) drawHeat(emptyHeat(state.clip));
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

function drawHeat(rows) {
  const grid = document.querySelector("[data-testid=heatmap]");
  if (!grid || !rows.length) return;
  grid.replaceChildren();
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
