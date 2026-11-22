"""The Research post on V-JEPA 2-AC's gripper blind spot. Every number is read from ``demo-data/vjepa2ac/``.

The files are the summaries the ``ii gripper-probe``, ``gripper-trace``, ``gripper-sae`` and
``gripper-circuit`` runs wrote on a Modal T4 (per-event rows dropped). Blocks are counted from 1:
a ``gripper-trace`` layer ``L`` is block ``L``'s output, and ``gripper-circuit``'s ``b9.attn`` is
block 10's attention.
"""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import numpy as np

SLUG = "vjepa2ac-gripper"
TITLE = "A robot world model ignores its own gripper. We looked inside to see why."
DATE = "November 2026"
FILES = (
    "gripper-probe.json",
    "gripper-probe-random-predictor.json",
    "gripper-trace.json",
    "gripper-trace-random-predictor.json",
    "gripper-sae-L9.json",
    "gripper-circuit.json",
)


def load(data_dir: Path) -> dict:
    data = {name: json.loads((Path(data_dir) / name).read_text(encoding="utf-8")) for name in FILES}
    probe = data["gripper-probe.json"]
    if probe["verdict"] != "gripper_blind":
        raise ValueError(f"the post is written for a gripper_blind verdict, got {probe['verdict']}")
    circuit = data["gripper-circuit.json"]["verdict"]
    if circuit != {"components": "none", "heads": "none", "steer": "no_fix"}:
        raise ValueError(f"the post is written for null circuit verdicts, got {circuit}")
    return data


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _range(lo: float, hi: float) -> str:
    return f"{100 * lo:.0f}-{100 * hi:.0f}%"


def _interval(r: dict) -> str:
    return f"{100 * r['lo']:.0f}-{100 * r['hi']:.0f}%"


def facts(data: dict) -> dict:
    """Every number the post states, derived from the result files."""
    probe, probe_rand = data["gripper-probe.json"], data["gripper-probe-random-predictor.json"]
    trace, trace_rand = data["gripper-trace.json"], data["gripper-trace-random-predictor.json"]
    sae, circ = data["gripper-sae-L9.json"], data["gripper-circuit.json"]
    ratio = trace["layers"]["grip_over_arm_frame"]
    ratio_rand = trace_rand["layers"]["grip_over_arm_frame"]
    peak = int(np.argmax(ratio[1:]) + 1)
    randoms = {k: v for k, v in sae["arms"].items() if k.startswith("random:")}
    best_random = max(v["consistency"]["value"] for v in randoms.values())
    picks = [p for p in circ["picks"].values() if p]
    shared = sorted(set.intersection(*[{c for c in p["comps"] if c.endswith(".attn")} for p in picks]), key=lambda c: int(c[1:].split(".")[0]))
    blocks = [int(c[1:].split(".")[0]) for c in shared]
    sg, sa = np.asarray(circ["parts"]["s_grip"]), np.asarray(circ["parts"]["s_arm"])
    removes = [-sg[b, 0] for b in blocks]
    adds = [sa[b, 0] for b in blocks]
    base_curve = np.asarray(circ["curve"])
    attn_curve = np.asarray(circ["knockout"]["all_attn"]["curve"])
    after = slice(circ["picks"]["0"]["peak"], len(base_curve) - 2)
    steer = circ["steer"][str(circ["rules"]["steer_alpha"])]
    return {
        "n_events": probe["n_events"],
        "n_episodes": probe["n_episodes"],
        "prefers": probe["stats"]["prefers_correct"],
        "prefers_random": probe_rand["stats"]["prefers_correct"],
        "max_arm_cm": round(100 * probe["rules"]["max_arm_m"]),
        "arm_step_cm": round(200 * trace["rules"]["arm_step"]),
        "action_after_1": trace["layers"]["recovered_action"][1],
        "frame_after_1": trace["layers"]["recovered_frame"][1],
        "ratio": ratio,
        "ratio_rand": ratio_rand,
        "peak": peak,
        "ratio_peak": ratio[peak],
        "ratio_out": ratio[-1],
        "ratio_rand_first": ratio_rand[1],
        "ratio_rand_range": (min(ratio_rand[1:]), max(ratio_rand[1:])),
        "encoder_auroc": trace["readout"]["encoder"],
        "pixels_auroc": trace["readout"]["pixels"],
        "sae": sae,
        "sae_trained": sae["arms"]["trained"],
        "sae_best_random": best_random,
        "sae_n_random": len(randoms),
        "suspect_blocks": [b + 1 for b in blocks],
        "removes": (min(removes), max(removes)),
        "adds": (min(adds), max(adds)),
        "knock": circ["knockout"]["comps"],
        "knock_heads": circ["knockout"]["heads"],
        "knock_attn": circ["knockout"]["all_attn"],
        "attn_larger": float(np.max(attn_curve[after] / base_curve[after]) - 1),
        "attn_out": float(attn_curve[-1]),
        "base_out": float(base_curve[-1]),
        "baseline": circ["baseline"]["prefers_correct"],
        "steer": steer,
        "steer_random": (min(circ["steer_random"]), max(circ["steer_random"])),
        "n_dirs": len(circ["steer_random"]),
        "fix_lo": circ["rules"]["fix_lo"],
        "circuit": circ,
    }


def teaser(f: dict) -> str:
    return (
        f"Meta's V-JEPA 2-AC picks the right gripper command {_pct(f['prefers']['value'])} of the time. "
        "We traced the command through its predictor, knocked parts out and tried to steer it back."
    )


def _chart(f: dict) -> str:
    """Gripper effect over arm effect after each block, trained against untrained, as inline SVG."""
    w, h, left, bottom = 560, 200, 34, 26
    top = 10
    n = len(f["ratio"]) - 1
    ymax = max(max(f["ratio"]), max(f["ratio_rand"])) * 1.08

    def pts(ys):
        out = []
        for i in range(1, n + 1):
            x = left + (i - 1) * (w - left - 70) / (n - 1)
            y = top + (1 - ys[i] / ymax) * (h - top - bottom)
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    def ylab(v):
        y = top + (1 - v / ymax) * (h - top - bottom)
        return f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text><line x1="{left}" x2="{w - 70}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'

    def end(ys, label, cls):
        y = top + (1 - ys[n] / ymax) * (h - top - bottom)
        return f'<text x="{w - 64}" y="{y + 4:.1f}" class="{cls}">{label}</text>'

    xticks = "".join(
        f'<text x="{left + (i - 1) * (w - left - 70) / (n - 1):.1f}" y="{h - 8}" text-anchor="middle">{i}</text>' for i in (1, 6, 12, 18, 24)
    )
    return f"""
    <figure class="chart" data-testid="ratio-chart">
      <svg viewBox="0 0 {w} {h}" role="img" aria-label="Gripper effect over arm effect after each predictor block, trained against untrained predictor">
        {ylab(0)}{ylab(1)}{ylab(2)}
        <polyline points="{pts(f['ratio_rand'])}" class="line-quiet"/>
        <polyline points="{pts(f['ratio'])}" class="line"/>
        {end(f['ratio'], 'trained', 'label')}{end(f['ratio_rand'], 'untrained', 'label-quiet')}
        {xticks}
      </svg>
      <figcaption>Gripper effect divided by the effect of a {f['arm_step_cm']} cm arm step, on the frame tokens after each of the 24 predictor blocks. {f['n_events']} events.</figcaption>
    </figure>"""


def write_post(dest: Path, data_dir: Path, page) -> None:
    """``page(title, current, body, description)`` is ``export_site._page``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data = load(data_dir)
    for name in FILES:
        shutil.copyfile(Path(data_dir) / name, dest / name)
    f = facts(data)
    k, ka, kh = f["knock"], f["knock_attn"], f["knock_heads"]
    st, tr = f["steer"], f["sae_trained"]
    suspects = ", ".join(str(b) for b in f["suspect_blocks"][:-1]) + f" and {f['suspect_blocks'][-1]}"
    v = f["circuit"]["verdict"]
    sources = ", ".join(f'<a href="{n}"><code>{n}</code></a>' for n in FILES)
    body = f"""
    <article class="post">
    <p class="back"><a href="/blog/">Research</a></p>
    <h1>{escape(TITLE)}</h1>
    <p class="lede">Meta's V-JEPA 2-AC plans robot arm moves by imagining the next frame and scoring how close it gets to a goal picture. We used <code>ii</code> to look inside its predictor and ask one narrow question: does it take the gripper into account?</p>

    <h2>The blind spot</h2>
    <p data-testid="post-headline">We found {f['n_events']} moments in {f['n_episodes']} DROID episodes where the gripper opens or closes while the arm moves less than {f['max_arm_cm']} cm. With the arm step fixed, the planning score preferred the right gripper command {_pct(f['prefers']['value'])} of the time (95% interval {_interval(f['prefers'])}). That is a coin flip. An untrained predictor scores {_pct(f['prefers_random']['value'])}, because it always prefers one direction.</p>

    <h2>Where the command goes</h2>
    <p>The predictor reads the command once. After block 1, copying the action token from the right-command run into the wrong one restores {_pct(f['action_after_1'])} of the difference, and copying the frame tokens restores {_pct(f['frame_after_1'])}.</p>
    <p data-testid="post-fade">Then the effect fades. Against a {f['arm_step_cm']} cm arm step, the gripper's effect grows to {f['ratio_peak']:.2f} times the arm's at block {f['peak']}, then falls to {f['ratio_out']:.2f} by the output. In an untrained predictor it stays between {f['ratio_rand_range'][0]:.1f} and {f['ratio_rand_range'][1]:.1f} the whole way, so the fade is learned.</p>
{_chart(f)}

    <h2>Looking for what turns it down</h2>
    <ul>
      <li data-testid="post-sae">A sparse dictionary at block {f['sae']['layer']} has a latent that moves with the command on {_pct(tr['consistency']['value'])} of held-out events, against {_pct(f['sae_best_random'])} for the best of {f['sae_n_random']} untrained dictionaries. Setting it and the next two latents to their right-command values changes the final prediction by {_pct(abs(tr['recovered']['value']))}.</li>
      <li data-testid="post-suspects">Attention in blocks {suspects} each removes {_range(*f['removes'])} of the gripper difference it receives, while adding {_range(*f['adds'])} to the arm's. Each half of the episodes picked them on its own.</li>
      <li data-testid="post-knockout">Knocking the top parts out recovered none of what was lost ({100 * k['recovered']['value']:.0f}%), the same as random parts ({100 * k['random_mean']:.0f}%). With every attention output after block {f['peak']} knocked out, the difference stays up to {_pct(f['attn_larger'])} larger, then the last two blocks bring it back to where it was ({100 * ka['recovered']['value']:.1f}% recovered).</li>
      <li data-testid="post-steer">Adding the block-{f['peak']} effect back at the output raised the right-command preference to {_pct(st['prefers_correct']['value'])} ({_interval(st['prefers_correct'])}). That is above every random direction ({_range(*f['steer_random'])}), but short of the {_pct(f['fix_lo'])} bar we set before the run.</li>
    </ul>

    <h2>What we think is going on</h2>
    <p>A probe on the encoder reads gripper open or closed on new episodes at AUROC {f['encoder_auroc']['value']:.2f}, about the same as raw pixels ({f['pixels_auroc']['value']:.2f}). The predictor is trained to match those latents, and they barely change when the gripper does. Our reading is that the model learned the gripper hardly matters to its target, and its last blocks enforce that however we edit the middle. If so, the fix is a view of the gripper the encoder can see, not an edit inside the predictor. We have not tested that.</p>
    <p data-testid="post-verdict">None of this is a finding. The dictionary latent passed its own check by a small margin. The knockouts and the steering did not pass. It is all offline, on recorded video, one planning step ahead.</p>

    <details class="appendix" data-testid="appendix">
      <summary>Every number behind this post</summary>
      <p>Model: V-JEPA 2-AC (ViT-g encoder, 24-block action-conditioned predictor) from Meta's vjepa2 code, fp16 on a T4. Data: 100 clips from LeRobot's droid_100 sample, 12 s each, at the model's 4 frames per second. Intervals resample whole episodes.</p>
      <p>Goal score: prefers the right command {_pct(f['prefers']['value'])} ({_interval(f['prefers'])}); untrained predictor {_pct(f['prefers_random']['value'])} ({_interval(f['prefers_random'])}). Verdict <code>{escape(data['gripper-probe.json']['verdict'])}</code>.</p>
      <p>Encoder probe AUROC {f['encoder_auroc']['value']:.3f} ({f['encoder_auroc']['lo']:.3f}-{f['encoder_auroc']['hi']:.3f}); pixels {f['pixels_auroc']['value']:.3f}. Gripper over arm by block, trained: {' '.join(f'{x:.2f}' for x in f['ratio'][1:])}. Untrained: {' '.join(f'{x:.2f}' for x in f['ratio_rand'][1:])}.</p>
      <p>Dictionary: BatchTopK, {f['sae']['sae']['n_latents']} latents, k={f['sae']['sae']['k']}, explained variance {f['sae']['sae']['health']['ev_raw']:.2f}, {f['sae']['n_test_events']} held-out events in {f['sae']['n_test_episodes']} episodes. Top latent {_pct(tr['consistency']['value'])} ({_interval(tr['consistency'])}). Verdict <code>{escape(f['sae']['verdict'])}</code>.</p>
      <p>Knockouts, held-out fold: top parts {k['recovered']['value']:+.3f} ({k['recovered']['lo']:+.3f} to {k['recovered']['hi']:+.3f}), random sets mean {k['random_mean']:+.3f}; top heads {kh['recovered']['value']:+.3f}, random {kh['random_mean']:+.3f}; all attention after the peak {ka['recovered']['value']:+.3f}. Final gripper drift {f['attn_out']:.4f} with all attention knocked out, {f['base_out']:.4f} without. Steering: baseline {_pct(f['baseline']['value'])}, steered {_pct(st['prefers_correct']['value'])}, arm check {_pct(st['arm_pref']['value'])}, score change {100 * st['damage']['value']:.1f}%, {f['n_dirs']} random directions {_range(*f['steer_random'])}. Verdicts: parts <code>{v['components']}</code>, heads <code>{v['heads']}</code>, steering <code>{v['steer']}</code>.</p>
      <p>Commands: <code>ii gripper-probe</code>, <code>ii gripper-trace</code>, <code>ii gripper-sae --layer 9</code>, <code>ii gripper-circuit</code>. Sources: {sources}.</p>
    </details>
    </article>
    """
    (dest / "index.html").write_text(page(f"{TITLE} · interpretable intelligence", "post", body, teaser(f)), encoding="utf-8")
