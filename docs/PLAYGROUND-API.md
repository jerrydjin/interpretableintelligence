# Playground API v1

The contract between the live playground page (`demo-site/live/`) and the service that runs the model. The page is a thin client: it shows pictures, labels and badges from the static pack, and asks this API to read a picture and to make edits.

- Compute: `src/interpretableintelligence/playground/compute.py` (PyTorch; the model and the SAE).
- Routing: `src/interpretableintelligence/playground/http.py` (FastAPI; validation, errors, CORS). It only calls `PackModel.read` and `PackModel.steer`.
- Run it locally with `ii serve-live`, host it on Modal with `ii deploy-live` (docs/HOSTING.md).

Anything that speaks this contract can sit in front of the compute: a gateway written in Rust (next to `server/`) could take over routing, auth and rate limits by forwarding `read` and `steer` unchanged. The page and the compute do not change.

Exploration only. Nothing here reads or writes a run folder or a scoreboard, and no response can make a feature verified.

## Conventions

- Every path starts with `/v1`. A breaking change gets `/v2`; v1 keeps working while the page moves over. Adding a field to a response is not breaking, so clients ignore fields they do not know.
- JSON in and out, except the picture in `read`, which is raw bytes.
- A pack is addressed by its directory name under `demo-site/live/packs/` (for example `dinov2-small-droid`). The page uses the same name for the static files.
- Responses over 2 KB are gzip-compressed when the client accepts it.
- CORS: `GET` and `POST` from the origins the deployment allows (`--origins`, default any). No credentials or cookies.
- No auth in v1. The deployment caps cost with its container limit (docs/HOSTING.md).

## Errors

Every error has the same body, with an HTTP status and a stable `code`:

```json
{"error": {"code": "unknown_image", "message": "image 3f2a… is not loaded here; read it again"}}
```

| Status | `code` | When |
| --- | --- | --- |
| 400 | `bad_request` | A malformed body: wrong picture size or content type, unknown edit kind, a feature id out of range, a value too large, an unknown field. |
| 404 | `pack_not_found` | No pack with that name on this server. |
| 404 | `unknown_image` | `steer` named an `image_id` this server does not hold (it forgot it, or another replica read it). Call `read` again with the same picture, then retry. |

## `GET /v1/health`

Cheap. The page calls it first, which also wakes a scaled-to-zero deployment.

```json
{"api": "ii-playground", "version": 1, "packs": ["dinov2-small-droid"]}
```

## `GET /v1/packs` and `GET /v1/packs/{pack}`

`/v1/packs` returns `{"packs": [<pack>, …]}`; `/v1/packs/{pack}` returns one:

```json
{
  "id": "dinov2-small-droid",
  "title": "DINOv2-small, trained on 80 DROID robot videos",
  "model": "facebook/dinov2-small",
  "layer": 6,
  "image_size": 224,
  "grid": {"h": 16, "w": 16},
  "n_features": 3072,
  "sae": {"activation": "topk", "k": 32},
  "device": "cuda"
}
```

## `POST /v1/packs/{pack}/read`

Body: the picture already squashed to `image_size` × `image_size`, as raw 8-bit RGB, row by row (`image_size² × 3` bytes; 150,528 for 224). `Content-Type: application/octet-stream`. The page resizes in a canvas, the same whole-frame squash the `hf` adapter uses; normalisation happens on the server from `pack.json`'s mean and std.

Response: every feature that fires on every patch token (row-major over the grid), strongest first within a patch, as compressed rows. Patch `t`'s features are `ids[offsets[t]:offsets[t+1]]` with activations `values[…]` over the same range.

```json
{
  "image_id": "8d1c0b6e4a2f…",
  "grid": {"h": 16, "w": 16},
  "n_features": 3072,
  "patches": {
    "offsets": [0, 32, 64, …, 8192],
    "ids": [1104, 87, …],
    "values": [3.91, 2.07, …]
  },
  "timing_ms": {"model": 12.4, "total": 14.0}
}
```

`image_id` is a hash of the pack and the bytes, so the same picture always gets the same id. The server keeps the model's hidden state for the most recent pictures (64 by default) so that edits skip the first half of the network.

## `POST /v1/packs/{pack}/steer`

Edit some features on every patch of a picture that was read, run the rest of the network, and measure how far each patch's final token turned.

```json
{"image_id": "8d1c0b6e4a2f…", "features": [1104], "kind": "scale", "value": 3}
```

| Field | Meaning |
| --- | --- |
| `features` | 1 to 16 distinct feature ids. |
| `kind` | `ablate` (set to 0; `value` ignored), `scale` (multiply where it fires), `clamp` (set to `value` on every patch), `add` (add `value`). |
| `value` | A number within ±10,000. |

The edit is `x' = x + Σ (f'ᵢ − fᵢ) W_dec[:, i]` on the patch tokens at the SAE's layer, the same as `steer.steer_edit` and the in-browser engine.

```json
{
  "image_id": "8d1c0b6e4a2f…",
  "shift": [0.0012, 0.0431, …],
  "mean": 0.0187,
  "max": 0.0911,
  "loss_change": 0.0042,
  "timing_ms": {"total": 9.8}
}
```

`shift` has one number per patch (row-major): `1 − cos(clean final token, edited final token)`. `loss_change` is the mean squared change over all final patch tokens.

## What the client does

1. `GET /v1/health` (wakes the server), then static `pack.json` and `features.json`.
2. `read` each picture; keep its bytes and `image_id`.
3. `steer` for each edit. On `unknown_image`, `read` the same bytes and retry once.
4. While one edit is in flight, newer edits replace any that are waiting, so a dragged slider sends the latest value instead of a queue.

`demo-site/live/remote.js` is that client. With `config.json`'s `api` set to `null` the page falls back to running the pack in the browser (`engine.js`).
