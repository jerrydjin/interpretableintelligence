// Demo site only (ii export-site). The "Try it" button on /droid/ opens a clip on a
// verified feature, switches the feature off in the playground, turns on the blue
// overlay and says in one line what the blue means. Without this script the button
// is still a plain link to the same clip and feature.
(function () {
  const button = document.querySelector("[data-story-try]");
  if (!button) return;

  function waitFor(test, done, tries) {
    const found = test();
    if (found) return done(found);
    if (tries <= 0) return undefined;
    setTimeout(() => waitFor(test, done, tries - 1), 100);
  }

  function tip(name) {
    document.querySelector(".story-tip")?.remove();
    const box = document.createElement("div");
    box.className = "story-tip";
    box.setAttribute("role", "status");
    box.dataset.testid = "story-tip";
    const text = document.createElement("p");
    const blue = document.createElement("b");
    blue.textContent = "Blue";
    text.append(blue, ` shows where the model’s view of the scene changed when ${name} was switched off. Brighter means a bigger change. Now try Scale in the playground to turn it up, or pick another feature from the list.`);
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Got it";
    close.addEventListener("click", () => box.remove());
    box.append(text, close);
    document.body.append(box);
  }

  button.addEventListener("click", (event) => {
    event.preventDefault();
    const { clip, frame, feature, name } = button.dataset;
    location.hash = `#/clips/${encodeURIComponent(clip)}?frame=${frame}&feature=${feature}`;
    // Wait for the viewer to open the clip with this feature selected.
    waitFor(() => {
      const list = document.querySelector("[data-testid=frame-features]");
      const panel = document.querySelector("[data-testid=steer-panel]");
      const run = document.querySelector("[data-testid=steer-run]");
      return list && panel && run && !run.disabled && list.dataset.feature === String(feature) ? panel : null;
    }, (panel) => {
      const kind = panel.querySelector("[data-testid=steer-kind]");
      if (kind.value !== "ablate") {
        kind.value = "ablate";
        kind.dispatchEvent(new Event("change"));
      }
      panel.querySelector("[data-testid=steer-run]").click();
      waitFor(() => panel.querySelector("[data-testid=steer-mean]"), () => {
        const show = panel.querySelector("[data-testid=steer-show]");
        if (!show.checked) {
          show.checked = true;
          show.dispatchEvent(new Event("change"));
        }
        document.getElementById("app").scrollIntoView({ behavior: "smooth", block: "start" });
        panel.classList.remove("story-glow");
        void panel.offsetWidth;
        panel.classList.add("story-glow");
        tip(name);
      }, 80);
    }, 80);
  });
})();
