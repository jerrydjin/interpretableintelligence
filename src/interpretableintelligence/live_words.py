"""The words readout for the live playground (``ii export-live --readout words``).

A vision model's final tokens are hard to read: the tokens readout can only say how far an edit
turned them. An image-text model (CLIP) also says what it sees, in words. This module adds what
the page needs for that, next to the usual pack files:

``front.onnx``
    Pixels to the hidden state after block ``layer`` (CLIP's ``pre_layrnorm`` included).
``words.onnx``
    That hidden state and a 512-vector ``u`` to the image embedding ``e`` (before
    normalisation) and ``grad = d(e . u) / d hidden``. With ``u = d log p(word) / d e`` the
    page gets the gradient of one word's log-probability with respect to every token, so it can
    rank the features that make the model see that word (activation x gradient along each
    feature's direction, summed over patches). The same graph reads an edited hidden state, so
    switching features off costs one call.
Both graphs store their weight matrices as int8 with a float scale per channel
(:func:`int8_onnx`), vectors as float16; the page casts them back to float32.
``text.bin``
    The words' text embeddings, normalised, float16 ``[n_words, dim]``: the page needs no text
    model. ``vocab.json`` lists the words in the same order.

The words come from ``live_words.txt`` (or ``--words``), one prompt each: "a photo of a <word>.".
The model's probabilities are a softmax over that list at CLIP's logit scale.

Per feature, ``features.json`` also gets ``sees``: the words that rise most when the feature is
written onto every patch of a few pictures the dictionary did not train on, and on how many of
them each word was the top riser. It is measured, not a label, and exploration only: nothing
here sets a badge.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

WORDS_FILE = Path(__file__).with_name("live_words.txt")
PROMPT = "a photo of {article} {word}."


class WordsError(ValueError):
    """The model cannot give a words readout."""


def load_words(path: Path | None = None) -> list[str]:
    """The word list: one per line, '#' comments and blank lines skipped, duplicates dropped."""
    text = Path(path or WORDS_FILE).read_text(encoding="utf-8")
    seen: dict[str, None] = {}
    for line in text.splitlines():
        word = line.split("#", 1)[0].strip()
        if word:
            seen.setdefault(word, None)
    words = list(seen)
    if len(words) < 2:
        raise WordsError(f"{path or WORDS_FILE} must list at least two words")
    return words


def prompt(word: str) -> str:
    return PROMPT.format(article="an" if word[:1].lower() in "aeiou" else "a", word=word)


def load_image_text_model(adapter):
    """The full CLIP model behind the adapter's vision tower, checked to hold the same vision weights."""
    from transformers import AutoModel, AutoTokenizer

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = AutoModel.from_pretrained(adapter._source, dtype=torch.float32, attn_implementation="eager").eval()
    needed = ("vision_model", "visual_projection", "text_model", "text_projection", "logit_scale")
    if not all(hasattr(model, name) for name in needed):
        raise WordsError(
            f"{adapter.model_name} is not a CLIP-style image-text model (it needs a visual and a text projection); "
            "the words readout takes CLIP models for now"
        )
    model.requires_grad_(False)
    tower = model.vision_model
    ours = adapter.model.state_dict()
    theirs = tower.state_dict()
    if set(ours) != set(theirs) or any(not torch.equal(ours[k].float(), theirs[k].float()) for k in ours):
        raise WordsError("the image-text model's vision tower does not match the run's model weights")
    tokenizer = AutoTokenizer.from_pretrained(adapter._source)
    return model, tokenizer


def _run_blocks(blocks, hidden: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        out = block(hidden, None)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


class Front(torch.nn.Module):
    """Pixels to the hidden state after block ``layer``, as the run's hook saw it."""

    def __init__(self, tower, layer: int) -> None:
        super().__init__()
        self.embeddings = tower.embeddings
        self.pre = tower.pre_layrnorm
        self.blocks = torch.nn.ModuleList(tower.encoder.layers[:layer])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _run_blocks(self.blocks, self.pre(self.embeddings(pixel_values)))


class WordsHead(torch.nn.Module):
    """The hidden state after block ``layer`` to the image embedding, and the gradient of ``e . u``."""

    def __init__(self, model, layer: int) -> None:
        super().__init__()
        tower = model.vision_model
        self.blocks = torch.nn.ModuleList(tower.encoder.layers[layer:])
        self.post = tower.post_layernorm
        self.proj = model.visual_projection

    def embed(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.post(_run_blocks(self.blocks, hidden)[:, 0]))

    def forward(self, hidden: torch.Tensor, u: torch.Tensor):
        with torch.enable_grad():
            hidden = hidden.detach().requires_grad_(True)
            e = self.embed(hidden)
            (grad,) = torch.autograd.grad((e * u).sum(), hidden)
        return e.detach(), grad


def text_table(model, tokenizer, words: list[str], batch: int = 256) -> np.ndarray:
    """Normalised text embeddings ``[n_words, dim]`` for the prompts."""
    out = []
    with torch.no_grad():
        for start in range(0, len(words), batch):
            tokens = tokenizer([prompt(w) for w in words[start : start + batch]], padding=True, return_tensors="pt")
            t = model.text_projection(model.text_model(**tokens).pooler_output)
            out.append(t / t.norm(dim=-1, keepdim=True))
    return torch.cat(out).numpy().astype(np.float32)


def word_logprobs(embed: torch.Tensor, table: torch.Tensor, scale: float) -> torch.Tensor:
    e = embed / embed.norm(dim=-1, keepdim=True)
    return torch.log_softmax(scale * e @ table.T, dim=-1)


def export_words_onnx(head: WordsHead, hidden: torch.Tensor, path: Path) -> None:
    """Trace forward and backward into one plain graph (no autograd at run time) and write it."""
    from torch._decomp import core_aten_decompositions
    from torch.fx.experimental.proxy_tensor import make_fx

    u = torch.zeros(1, head.proj.out_features)
    traced = make_fx(lambda h, v: head(h, v), decomposition_table=core_aten_decompositions())(hidden, u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(
                traced,
                (hidden, u),
                str(path),
                input_names=["hidden", "u"],
                output_names=["embed", "grad"],
                opset_version=18,
                dynamo=True,
                external_data=False,
                verbose=False,
            )
        except ImportError as exc:  # the torch.export-based exporter writes through onnxscript
            raise WordsError("the words readout needs onnxscript: uv sync --extra live") from exc


def export_front_onnx(front: Front, pixels: torch.Tensor, path: Path) -> None:
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            front, (pixels,), str(path), input_names=["pixel_values"], output_names=["hidden"],
            opset_version=18, dynamo=False, do_constant_folding=False,
        )


def halve_onnx(path: Path, min_elems: int = 1024) -> None:
    """Store every float32 initializer of ``min_elems`` or more values as float16, cast back in the graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    casts = []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(init)
        if array.size < min_elems:
            continue
        name = init.name
        half = numpy_helper.from_array(array.astype(np.float16), name=f"{name}__f16")
        graph.initializer.remove(init)
        graph.initializer.append(half)
        casts.append(helper.make_node("Cast", [half.name], [name], to=TensorProto.FLOAT, name=f"{name}__cast"))
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.save(model, str(path))


def int8_onnx(path: Path) -> int:
    """Store every float16 matrix of ``path`` (after :func:`halve_onnx`) as int8 with one scale per channel.

    Each ``Cast`` from a float16 weight of two or more dimensions becomes a ``DequantizeLinear``
    from int8, so the graph (and its gradient) is unchanged and the file about halves. Matrices
    are scaled per column (axis 1), convolution kernels per output channel (axis 0); vectors and
    the embeddings stay float16. On CLIP ViT-B/16 the top word matched the float16 graph on 19
    of 19 sample pictures (its probability within 4 points). Returns how many tensors it stored.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    done = 0
    for node in list(graph.node):
        if node.op_type != "Cast" or node.input[0] not in inits or not node.input[0].endswith("__f16"):
            continue
        name = node.input[0]
        array = numpy_helper.to_array(inits[name]).astype(np.float32)
        if array.ndim < 2 or "embedding" in name:
            continue
        axis = 1 if array.ndim == 2 else 0
        others = tuple(i for i in range(array.ndim) if i != axis)
        scale = np.abs(array).max(axis=others) / 127.0
        scale[scale == 0] = 1.0
        shape = [1] * array.ndim
        shape[axis] = -1
        q = np.clip(np.round(array / scale.reshape(shape)), -127, 127).astype(np.int8)
        stem = name[: -len("__f16")]
        graph.initializer.remove(inits[name])
        graph.initializer.extend([numpy_helper.from_array(q, f"{stem}__q8"), numpy_helper.from_array(scale.astype(np.float32), f"{stem}__q8s")])
        at = list(graph.node).index(node)
        graph.node.remove(node)
        graph.node.insert(at, helper.make_node("DequantizeLinear", [f"{stem}__q8", f"{stem}__q8s"], [node.output[0]], axis=axis, name=f"{stem}__dq"))
        done += 1
    onnx.save(model, str(path))
    return done


def check_no_float64(path: Path) -> None:
    """ONNX Runtime Web's WebGPU build has no float64 kernels."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path))
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                raise WordsError(f"{path.name} casts to float64 ({node.name})")
    for init in model.graph.initializer:
        if init.data_type == TensorProto.DOUBLE:
            raise WordsError(f"{path.name} stores a float64 tensor ({init.name})")


def sees(
    head: WordsHead,
    hiddens: torch.Tensor,
    decoder: torch.Tensor,
    peaks: torch.Tensor,
    table: torch.Tensor,
    scale: float,
    prefix: int,
    words: list[str],
    strength: float = 2.0,
    top: int = 3,
    batch: int = 8,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[list[dict]]:
    """For each feature, the words that rise most when it is written onto every patch.

    ``hiddens`` ``[n, tokens, d]``: hidden states of pictures the dictionary did not train on.
    The feature is set to ``strength`` times its training peak on every patch token (the
    playground's force on). Returns, per feature, up to ``top`` words with the mean rise in
    probability and on how many of the ``n`` pictures that word rose most.
    """
    n = hiddens.shape[0]
    with torch.no_grad():
        base = word_logprobs(head.embed(hiddens), table, scale).exp()
    mask = torch.zeros(1, 1, hiddens.shape[1], 1)
    mask[..., prefix:, :] = 1.0
    live = [f for f in range(decoder.shape[0]) if float(peaks[f]) > 0]
    found: dict[int, list[dict]] = {}
    for start in range(0, len(live), batch):
        if start % 512 < batch:
            log(f"what each feature makes the model see: {start}/{len(live)}")
        chunk = live[start : start + batch]
        deltas = mask * (strength * peaks[chunk].view(-1, 1, 1, 1) * decoder[chunk].view(len(chunk), 1, 1, -1))
        edited = (hiddens.unsqueeze(0) + deltas).reshape(len(chunk) * n, *hiddens.shape[1:])
        with torch.no_grad():
            probs = word_logprobs(head.embed(edited), table, scale).exp().view(len(chunk), n, -1)
        rise = probs - base.unsqueeze(0)
        for k, f in enumerate(chunk):
            mean = rise[k].mean(0)
            firsts = torch.bincount(rise[k].argmax(-1), minlength=len(words))
            order = torch.argsort(mean, descending=True)[:top]
            rows = [{"word": words[int(i)], "rise": round(float(mean[i]), 4), "top_on": int(firsts[i]), "of": n} for i in order]
            found[f] = [row for row in rows if row["rise"] > 0]
    return [found.get(f, []) for f in range(decoder.shape[0])]
