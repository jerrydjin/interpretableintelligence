// Bring your own model: name a Hugging Face vision model, the backend trains a dictionary for it
// on robot videos (POST /v1/models, docs/PLAYGROUND-API.md), and the page reopens on it.
// Only shown when the model server says it builds models (health.builds).

const POLL_MS = 2500;

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

const minutes = (s) => (s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);

// Where the page goes once a pack exists: the same page, on that pack, through the same server.
export function packUrl(packId) {
  const url = new URL(location.href);
  url.searchParams.set("model", packId);
  url.searchParams.delete("pack");
  return url.href;
}

export function mountModelPicker(box, engine) {
  const limits = engine.health && engine.health.builds;
  if (!box || !limits) return;
  const input = el("input", {
    type: "text",
    id: "byom-model",
    placeholder: "facebook/dinov2-base",
    autocomplete: "off",
    spellcheck: "false",
    "aria-label": "Hugging Face model id",
  });
  const go = el("button", { class: "pill-btn", type: "submit" }, "Build");
  const status = el("p", { class: "byom-status", "aria-live": "polite" });
  const track = el("div", { class: "loader-track", hidden: true }, el("div", { class: "loader-fill" }));
  const fill = track.firstChild;
  const form = el("form", { class: "byom-form" }, input, go);
  const params = Math.round(limits.max_params / 1e6);
  box.replaceChildren(
    el(
      "details",
      { class: "byom" },
      el("summary", {}, "Try your own model"),
      el(
        "p",
        { class: "fine" },
        `Name a vision transformer on Hugging Face (ViT, DINOv2, DeiT; up to ${params}M parameters). `,
        `A dictionary is trained for it on ${limits.n_clips} robot videos, which takes a few minutes. `,
        "Every model is built once and kept, so the next person gets it straight away. ",
        "Its features have no names or ticks yet: they are for exploring."
      ),
      form,
      track,
      status
    )
  );

  let timer = null;
  const say = (text, failed = false) => {
    status.textContent = text;
    status.classList.toggle("failed", failed);
  };
  const show = (job) => {
    track.hidden = false;
    fill.style.width = `${Math.max(3, Math.round(job.progress * 100))}%`;
    if (job.state === "done") {
      say("Ready. Opening it now.");
      try {
        sessionStorage.removeItem("ii-byom-job");
      } catch {
        /* nothing to clear */
      }
      location.href = packUrl(job.pack);
      return true;
    }
    if (job.state === "failed") {
      track.hidden = true;
      say(job.message || "The build failed.", true);
      go.disabled = input.disabled = false;
      return true;
    }
    say(`${job.stage_text}. ${minutes(job.elapsed_s)} so far. You can leave this page open or come back to it.`);
    return false;
  };
  const poll = (id) => {
    timer = setTimeout(async () => {
      try {
        const { job } = await engine.call(`/v1/jobs/${encodeURIComponent(id)}`);
        if (!show(job)) poll(id);
      } catch (err) {
        say(`Lost touch with the server (${err.message}). Retrying.`);
        poll(id);
      }
    }, POLL_MS);
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const model = input.value.trim();
    if (!model) return input.focus();
    clearTimeout(timer);
    go.disabled = input.disabled = true;
    say("Checking the model on Hugging Face.");
    try {
      const { job } = await engine.call("/v1/models", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model }),
      });
      try {
        sessionStorage.setItem("ii-byom-job", job.id);
      } catch {
        /* storage off: polling still works while the page is open */
      }
      if (!show(job)) poll(job.id);
    } catch (err) {
      track.hidden = true;
      say(err.message, true);
      go.disabled = input.disabled = false;
    }
  });

  // A build started in this tab before a reload picks up where it was.
  let pending = null;
  try {
    pending = sessionStorage.getItem("ii-byom-job");
  } catch {
    pending = null;
  }
  if (pending && !new URLSearchParams(location.search).has("model")) {
    engine
      .call(`/v1/jobs/${encodeURIComponent(pending)}`)
      .then(({ job }) => {
        if (job.state === "queued" || job.state === "running") {
          box.querySelector("details").open = true;
          input.value = job.model;
          go.disabled = input.disabled = true;
          show(job);
          poll(job.id);
        }
      })
      .catch(() => {});
  }
}
