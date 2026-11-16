// Plain-English guide to the scoreboard gates.
//
// Reads the same fields the scoreboard writes and says, for each gate, what it
// asks, what this feature's numbers mean, and what to try next when it fails.
// It never decides `real`: the badge still comes from the scorer. The numbers
// below mirror src/interpretableintelligence/thresholds.py, and
// tests/test_guide.py fails if they drift.
(function (root) {
  "use strict";

  const THRESHOLDS = {
    USABLE_LOSS_RECOVERED: 0.7,
    DISTINCT_CLIPS_TOP20: 5,
    TOP_K_EXAMPLES: 20,
    RANDOM_PERCENTILE_MIN: 95.0,
    LABEL_ASSIGN_MIN_CORR: 0.5,
    LABEL_HOLD_MIN_CORR: 0.6,
    PATTERN_ABSENT_MAX_CORR: 0.45,
  };

  const T = THRESHOLDS;

  function pct(value) {
    if (!Number.isFinite(value)) return "?";
    const scaled = Math.round(value * 1000) / 10;
    return Number.isInteger(scaled) ? String(scaled) : scaled.toFixed(1);
  }

  function num(value) {
    if (!Number.isFinite(value)) return "?";
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 0.01) return value.toFixed(3);
    return value.toExponential(1);
  }

  function percentile(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }

  function missing(value) {
    return value === null || value === undefined;
  }

  // ---- SAE gate -------------------------------------------------------------

  function sae(block) {
    const base = {
      key: "usable",
      title: "Is the dictionary good enough to trust?",
      asks:
        "The SAE (sparse autoencoder) rewrites the model's layer as a list of features. " +
        "We swap that rewrite back into the model and check how much of its prediction quality survives.",
    };
    if (!block) {
      return {
        ...base,
        passed: null,
        result: "This run has no scoreboard yet, so the dictionary has not been checked.",
        next: { text: "Score the run (`ii run` without `--no-score`) to get every check below." },
      };
    }
    const lr = block.loss_recovered;
    if (block.usable === true) {
      return {
        ...base,
        passed: true,
        result: Number.isFinite(lr)
          ? `With the SAE swapped in, the model keeps ${pct(lr)}% of its prediction quality (the bar is ${pct(T.USABLE_LOSS_RECOVERED)}%), and it still beats a guess-the-average baseline.`
          : "With the SAE swapped in, the model keeps enough of its prediction quality to pass.",
        next: null,
      };
    }
    const tooLeaky = Number.isFinite(lr) && lr < T.USABLE_LOSS_RECOVERED;
    return {
      ...base,
      passed: false,
      result: tooLeaky
        ? `With the SAE swapped in, the model keeps only ${pct(lr)}% of its prediction quality; the bar is ${pct(T.USABLE_LOSS_RECOVERED)}%. ` +
          "Features from a leaky dictionary may describe the SAE's mistakes rather than the model, so no feature here can be verified."
        : "The swapped-in model does no better than predicting the average of the visible patches, so the dictionary is not usable and no feature here can be verified.",
      next: {
        text:
          "Train a wider SAE or train it for longer, or try another layer with `--layer`. " +
          "You can still explore features in the playground in the meantime.",
      },
    };
  }

  // ---- feature gates --------------------------------------------------------

  function spread(feature) {
    const value = feature.distinct_clips_top20;
    const base = {
      key: "distinct_clips_top20",
      title: "Does it show up across many clips?",
      asks: `We take its ${T.TOP_K_EXAMPLES} strongest activations and count how many different clips they come from.`,
    };
    if (missing(value)) {
      return { ...base, passed: null, result: "Not measured for this feature.", next: { text: "Rerun scoring to fill this in." } };
    }
    if (value >= T.DISTINCT_CLIPS_TOP20) {
      return {
        ...base,
        passed: true,
        result: `Its strongest moments come from ${value} different clips (needs at least ${T.DISTINCT_CLIPS_TOP20}), so it is a general pattern, not one clip's quirk.`,
        next: null,
      };
    }
    return {
      ...base,
      passed: false,
      result:
        value === 0
          ? "It never fires strongly on any clip, so there is nothing to generalise from. It may be a dead feature."
          : `Its strongest moments come from only ${value} ${value === 1 ? "clip" : "clips"} (needs at least ${T.DISTINCT_CLIPS_TOP20}). It may be memorising ${value === 1 ? "that clip" : "those clips"} rather than tracking something general.`,
      next: {
        text: "Open its top clips below and look at what they share. Adding more varied clips and rerunning tells you whether it spreads out.",
      },
    };
  }

  function label(feature, ctx) {
    const value = feature.label_holds_heldout;
    const name = feature.label ? `"${feature.label}"` : "its label";
    const base = {
      key: "label_holds_heldout",
      title: "Does its name still fit on unseen clips?",
      asks:
        `The labeller names a feature from training clips only when the match is clear (correlation above ${T.LABEL_ASSIGN_MIN_CORR.toFixed(2)}). ` +
        `Then we check the name on held-out clips the SAE never saw (needs ${T.LABEL_HOLD_MIN_CORR.toFixed(2)}).`,
    };
    if (value === true) {
      return { ...base, passed: true, result: `${cap(name)} still fits on held-out clips, so the name is not a training-set coincidence.`, next: null };
    }
    const look = {
      text: "Look at its top clips and name it yourself, then steer it in the playground to see what it changes.",
      href: ctx && ctx.steerHref ? ctx.steerHref : null,
      action: ctx && ctx.steerHref ? "Steer it" : null,
    };
    if (value === false) {
      return {
        ...base,
        passed: false,
        result: feature.label
          ? `${cap(name)} matched in training but not on held-out clips, so the name is probably a coincidence.`
          : "No known cue matched it clearly enough to name it, so there is no name to check.",
        next: look,
      };
    }
    return { ...base, passed: null, result: "The labeller did not run this check.", next: look };
  }

  function ablation(feature, ctx) {
    const p = feature.random_percentile;
    const n = feature.n_random;
    const effect = feature.ablation_effect;
    const base = {
      key: "random_percentile",
      title: "Does removing it matter more than removing a random feature?",
      asks:
        "We switch the feature off and measure how far the model's prediction moves. " +
        "Then we do the same for random look-alike features of the same size and firing rate.",
    };
    if (missing(p) || missing(effect)) {
      return {
        ...base,
        passed: null,
        result: "It was not switched off in this run, usually because it fell outside the `--prefilter` shortlist.",
        next: { text: "Rerun `ii run` without `--prefilter`, or with a larger K, to test it." },
      };
    }
    const controls = Number.isFinite(n) && n > 0 ? `${n} random look-alikes` : "the random look-alikes";
    if (p >= T.RANDOM_PERCENTILE_MIN) {
      return {
        ...base,
        passed: true,
        result: `Switching it off moves the prediction more than ${percentile(p)}% of ${controls} (needs ${percentile(T.RANDOM_PERCENTILE_MIN)}%).`,
        next: null,
      };
    }
    const steer = ctx && ctx.steerHref;
    return {
      ...base,
      passed: false,
      result:
        `Switching it off moved the prediction by ${num(effect)}, which beats only ${percentile(p)}% of ${controls} (needs ${percentile(T.RANDOM_PERCENTILE_MIN)}%). ` +
        "On its own, its effect looks like noise.",
      next: {
        text:
          "Features often work in teams. Rerun with `--extra-effects group` to remove it together with the features that fire alongside it, " +
          "or steer it harder in the playground (scale or clamp) to see whether a bigger push shows an effect.",
        href: steer || null,
        action: steer ? "Steer it" : null,
      },
    };
  }

  function randomInit(feature) {
    const value = feature.absent_in_random_init;
    const base = {
      key: "absent_in_random_init",
      title: "Did training create it?",
      asks:
        "We train the same SAE on a copy of the model with random, untrained weights. " +
        `If a feature there matches this one (correlation ${T.PATTERN_ABSENT_MAX_CORR.toFixed(2)} or more), the pattern comes from the input or the architecture, not from learning.`,
    };
    if (value === true) {
      return { ...base, passed: true, result: "Nothing in the untrained model's SAE matches it, so it comes from what the model learned.", next: null };
    }
    if (value === false) {
      return {
        ...base,
        passed: false,
        result:
          "The untrained model's SAE finds the same pattern (or this one barely varies). " +
          "It probably reflects the pixels or the patch position rather than anything the model learned.",
        next: { text: "Check whether it tracks where in the frame it fires rather than what is there: open the heatmap on a few clips." },
      };
    }
    return {
      ...base,
      passed: null,
      result: "The untrained-model comparison was not run.",
      next: { text: "Run full scoring (`ii run` without `--prefilter`) to include it." },
    };
  }

  function cap(text) {
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
  }

  function feature(record, ctx) {
    const gates = [spread(record), label(record, ctx), ablation(record, ctx), randomInit(record)];
    const dictionary = ctx && ctx.sae !== undefined ? sae(ctx.sae) : null;
    const all = dictionary ? [dictionary, ...gates] : gates;
    const failed = all.filter((gate) => gate.passed === false);
    const skipped = all.filter((gate) => gate.passed === null);
    const real = record.badge === "real";
    let verdict;
    if (real) {
      verdict = "Verified. It passes every check, so its effect on the model is real, not noise or a coincidence.";
    } else if (record.scored === false) {
      verdict = "Not scored yet, so treat it as exploratory. You can still look at it and steer it.";
    } else if (failed.length === 0 && skipped.length > 0) {
      verdict = `Exploratory. ${count(skipped.length, "check")} did not run, and a check that did not run never counts as a pass.`;
    } else {
      verdict = `Exploratory. It fails ${count(failed.length, "check")}${skipped.length ? ` and ${count(skipped.length, "check")} did not run` : ""}. Most features fail at least one; each failure below comes with an experiment to try next.`;
    }
    return { verdict, real, gates: all, status: statusNote(record.status) };
  }

  function count(n, word) {
    const words = ["no", "one", "two", "three", "four", "five"];
    return `${words[n] || n} ${word}${n === 1 ? "" : "s"}`;
  }

  // Scorer status strings that are not just a list of failed gates.
  function statusNote(status) {
    if (typeof status !== "string" || !status.trim()) return null;
    const text = status.trim().toLowerCase();
    if (text.startsWith("failed:")) return null;
    if (text.includes("pending")) {
      return "Some checks are still running. The badge updates when the scorer finishes; it never changes by hand.";
    }
    if (text.includes("estimate")) {
      return "The effect shown is an attribution estimate, not a measured removal. Only switching the feature off can verify it.";
    }
    return `Scorer note: ${status.trim()}.`;
  }

  // Which gate stops the most features, for the features list.
  function summary(features) {
    const keys = ["distinct_clips_top20", "label_holds_heldout", "random_percentile", "absent_in_random_init"];
    const short = {
      distinct_clips_top20: "fire in too few clips",
      label_holds_heldout: "have a name that does not hold on unseen clips",
      random_percentile: "do not beat random features when switched off",
      absent_in_random_init: "also appear in an untrained model",
    };
    const scored = features.filter((f) => f.scored !== false);
    const counts = {};
    for (const key of keys) counts[key] = 0;
    for (const record of scored) {
      if (record.badge === "real") continue;
      for (const gate of feature(record, {}).gates) {
        if (gate.passed === false) counts[gate.key] += 1;
      }
    }
    const lines = keys
      .filter((key) => counts[key] > 0)
      .sort((a, b) => counts[b] - counts[a])
      .map((key) => ({ key, count: counts[key], text: `${counts[key]} ${short[key]}` }));
    return { scored: scored.length, lines };
  }

  // ---- what to try in the playground ----------------------------------------
  //
  // Two playground edits per situation, the most telling first. The viewer shows
  // the first one the reader has not just run, so there is always a next one.
  // Steering is exploration: none of these can change the badge.

  const SHORT = {
    usable: "Dictionary",
    distinct_clips_top20: "Many clips",
    label_holds_heldout: "Name holds",
    random_percentile: "Beats random",
    absent_in_random_init: "Learned",
  };

  const IDEAS = {
    verified: [
      {
        title: "Turn it up",
        text: "Scale it to 3× and see where the prediction moves. Removing it already beat every random look-alike; this shows what it adds.",
        kind: "scale",
        value: 3,
      },
      {
        title: "Switch it off",
        text: "Remove it and compare the glow with where it fires. A verified feature should move the prediction where it is active.",
        kind: "ablate",
        value: 0,
      },
    ],
    random_percentile: [
      {
        title: "Push harder",
        text: "Removing it alone looked like noise. Scale it to 3× and see whether a bigger push moves the prediction.",
        kind: "scale",
        value: 3,
      },
      {
        title: "Force it on",
        text: "Clamp it on every patch, even where it is silent. If the prediction still barely moves, it may only matter alongside other features.",
        kind: "clamp",
        value: 1,
      },
    ],
    label_holds_heldout: [
      {
        title: "Name it yourself",
        text: "Remove it and watch where the prediction moves. The place that changes is a clue to what it stands for.",
        kind: "ablate",
        value: 0,
      },
      {
        title: "Exaggerate it",
        text: "Scale it to 3×. Whatever grows in the glow is what the model reads from it.",
        kind: "scale",
        value: 3,
      },
    ],
    distinct_clips_top20: [
      {
        title: "Try it somewhere new",
        text: "Clamp it on across this frame, even where it is silent, and see what it would add to a clip it rarely fires on.",
        kind: "clamp",
        value: 1,
      },
      {
        title: "Switch it off",
        text: "Remove it here. If nothing moves, it may only matter on its few clips.",
        kind: "ablate",
        value: 0,
      },
    ],
    absent_in_random_init: [
      {
        title: "Place or thing?",
        text: "Remove it here, then on another clip. If the same patches change both times, it tracks a position in the frame, not an object.",
        kind: "ablate",
        value: 0,
      },
      {
        title: "Force it on",
        text: "Clamp it on every patch and see whether the change still sits in one corner of the frame.",
        kind: "clamp",
        value: 1,
      },
    ],
    unmeasured: [
      {
        title: "Do the check by hand",
        text: "Scoring never switched it off. Remove it here and see how far the prediction moves.",
        kind: "ablate",
        value: 0,
      },
      {
        title: "Turn it up",
        text: "Scale it to 3× and see what it adds.",
        kind: "scale",
        value: 3,
      },
    ],
  };

  // The gate that decides the next experiment: the first failed one, else the first one that did not run.
  function blocking(read) {
    const gates = read.gates.filter((gate) => gate.key !== "usable");
    return gates.find((gate) => gate.passed === false) || gates.find((gate) => gate.passed === null) || null;
  }

  function experiments(record, ctx) {
    const read = feature(record, ctx || {});
    if (read.real) return { key: "verified", ideas: IDEAS.verified };
    const gate = blocking(read);
    const key = gate && IDEAS[gate.key] && gate.passed === false ? gate.key : "unmeasured";
    return { key, gate: gate ? gate.key : null, ideas: IDEAS[key] };
  }

  // Checks passed, in gate order, for the feature page's check trail.
  function trail(record, ctx) {
    const read = feature(record, ctx || {});
    const marks = read.gates.map((gate) => ({
      key: gate.key,
      name: SHORT[gate.key] || gate.key,
      mark: gate.passed === true ? "pass" : gate.passed === false ? "fail" : "skip",
    }));
    return { passed: marks.filter((m) => m.mark === "pass").length, total: marks.length, marks };
  }

  const api = { THRESHOLDS, sae, feature, statusNote, summary, experiments, trail, blocking };
  root.IIGuide = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
