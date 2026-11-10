//! JSON documents. Field meanings match `interpretableintelligence.view` and `interpretableintelligence.explain`.
//! Key order is not significant; the contract test sorts keys.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use duckdb::Connection;
use serde_json::{json, Map, Value};

use crate::clips::{self, LoadedClip};
use crate::error::{db_err, finite_f64, io_err, ApiError};
use crate::query::{self, EffectCell, TopClipRow};

const SCORE_FIELDS: [&str; 7] = [
    "distinct_clips_top20",
    "label_holds_heldout",
    "ablation_effect",
    "random_percentile",
    "n_random",
    "absent_in_random_init",
    "real",
];

pub struct RunContext {
    pub dir: PathBuf,
    pub run_id: String,
    pub model_sha: String,
    pub layer: i64,
    pub adapter: String,
    pub manifest: Value,
    pub scoreboard: Value,
    features_by_id: BTreeMap<i64, Value>,
}

struct Grid {
    t: i64,
    h: i64,
    w: i64,
    tubelet: i64,
    patch_px: i64,
}

pub fn resolve_run_id(runs_dir: &Path, run_id: &str) -> Result<PathBuf, ApiError> {
    if run_id.is_empty() || run_id.trim() != run_id {
        return Err(ApiError::bad("run_id must be the run folder name"));
    }
    if run_id.contains('/') || run_id.contains('\\') || run_id == "." || run_id == ".." {
        return Err(ApiError::bad(format!(
            "run_id must be a single folder name, got '{run_id}'"
        )));
    }
    if !runs_dir.is_dir() {
        return Err(ApiError::bad(format!(
            "runs directory not found: {}",
            runs_dir.display()
        )));
    }
    let path = runs_dir.join(run_id);
    if !path.is_dir() {
        return Err(ApiError::not_found(
            format!("run not found: {run_id}"),
            Some(run_id.to_string()),
            None,
            None,
        ));
    }
    let root = runs_dir.canonicalize().map_err(io_err)?;
    let resolved = path.canonicalize().map_err(io_err)?;
    if resolved.parent() != Some(root.as_path()) {
        return Err(ApiError::bad(format!(
            "run_id escapes the runs directory: '{run_id}'"
        )));
    }
    Ok(resolved)
}

pub fn load_run(run_dir: &Path) -> Result<RunContext, ApiError> {
    let manifest = read_object(&run_dir.join("manifest.json"))?;
    let scoreboard = read_object(&run_dir.join("scoreboard.json"))?;
    let run_id = manifest
        .get("run_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| ApiError::bad(format!("{} has no run_id", run_dir.display())))?
        .to_string();
    if run_dir.file_name().and_then(|name| name.to_str()) != Some(run_id.as_str()) {
        return Err(ApiError::bad(format!(
            "manifest.run_id '{run_id}' does not match the folder name '{}'",
            run_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
        )));
    }
    let model = manifest
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad(format!("run {run_id} is missing model or scoreboard.run")))?;
    let block = scoreboard
        .get("run")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad(format!("run {run_id} is missing model or scoreboard.run")))?;
    let model_sha = model
        .get("sha")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::bad(format!(
                "run {run_id} model sha does not match between manifest and scoreboard"
            ))
        })?
        .to_string();
    let board_sha = block.get("model_sha").and_then(Value::as_str);
    if Some(model_sha.as_str()) != board_sha {
        return Err(ApiError::bad(format!(
            "run {run_id} model sha does not match between manifest and scoreboard"
        )));
    }
    let layer = json_i64(manifest.get("layer").unwrap_or(&Value::Null)).ok_or_else(|| {
        ApiError::bad(format!(
            "run {run_id} layer does not match between manifest and scoreboard"
        ))
    })?;
    if Some(layer) != block.get("layer").and_then(json_i64) {
        return Err(ApiError::bad(format!(
            "run {run_id} layer does not match between manifest and scoreboard"
        )));
    }
    let adapter = model
        .get("adapter")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::bad(format!("run {run_id} has no adapter name")))?
        .to_string();
    let features = scoreboard
        .get("features")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or_else(|| ApiError::bad(format!("run {run_id} scoreboard has no features")))?;
    let mut features_by_id = BTreeMap::new();
    for record in features {
        let id = record.get("id").and_then(json_i64).ok_or_else(|| {
            ApiError::bad(format!(
                "run {run_id} has a scoreboard feature without an integer id"
            ))
        })?;
        if features_by_id.insert(id, record.clone()).is_some() {
            return Err(ApiError::bad(format!(
                "run {run_id} lists feature {id} more than once"
            )));
        }
    }
    let expected: Vec<i64> = (0..features.len() as i64).collect();
    if features_by_id.keys().copied().collect::<Vec<_>>() != expected {
        return Err(ApiError::bad(format!(
            "run {run_id} feature ids must be 0..{}",
            features.len() - 1
        )));
    }
    Ok(RunContext {
        dir: run_dir.to_path_buf(),
        run_id,
        model_sha,
        layer,
        adapter,
        manifest,
        scoreboard,
        features_by_id,
    })
}

pub fn list_runs(runs_dir: &Path) -> Result<Value, ApiError> {
    if !runs_dir.is_dir() {
        return Err(ApiError::bad(format!(
            "runs directory not found: {}",
            runs_dir.display()
        )));
    }
    let mut children = Vec::new();
    for entry in std::fs::read_dir(runs_dir).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        if entry.path().is_dir() {
            children.push(entry.path());
        }
    }
    children.sort();
    let mut runs = Vec::new();
    for path in children {
        if !path.join("manifest.json").is_file() || !path.join("scoreboard.json").is_file() {
            continue;
        }
        let ctx = load_run(&path)?;
        let usable =
            ctx.scoreboard.get("sae").and_then(|sae| sae.get("usable")) == Some(&Value::Bool(true));
        runs.push(json!({
            "run_id": ctx.run_id,
            "model_sha": ctx.model_sha,
            "layer": ctx.layer,
            "usable": usable,
            "adapter": ctx.adapter,
        }));
    }
    Ok(json!({ "runs": runs }))
}

pub fn feature_document(ctx: &RunContext, feature_id: i64) -> Result<Value, ApiError> {
    if !ctx.features_by_id.contains_key(&feature_id) {
        return Err(missing_feature(ctx, feature_id));
    }
    let path = ctx.dir.join("features").join("features.jsonl");
    if !path.is_file() {
        return Err(ApiError::bad(format!(
            "run {} is missing features/features.jsonl",
            ctx.run_id
        ))
        .traced(&ctx.run_id, &ctx.model_sha, ctx.layer));
    }
    let rows = read_jsonl(&path)?;
    let matched: Vec<&Value> = rows
        .iter()
        .filter(|row| row.get("id").and_then(json_i64) == Some(feature_id))
        .collect();
    if matched.len() != 1 {
        return Err(ApiError::bad(format!(
            "run {} features.jsonl does not contain exactly one record for feature {feature_id}",
            ctx.run_id
        ))
        .traced(&ctx.run_id, &ctx.model_sha, ctx.layer));
    }
    let record = ctx.features_by_id.get(&feature_id).unwrap();
    let mut feature = matched[0].as_object().cloned().ok_or_else(|| {
        ApiError::bad(format!(
            "run {} features.jsonl record for feature {feature_id} is not an object",
            ctx.run_id
        ))
    })?;
    for key in ["label"].into_iter().chain(SCORE_FIELDS) {
        let value = record.get(key).cloned().ok_or_else(|| {
            ApiError::bad(format!(
                "run {} scoreboard feature {feature_id} is missing {key}",
                ctx.run_id
            ))
        })?;
        feature.insert(key.to_string(), value);
    }
    feature.insert("ablation_effect_kind".into(), json!("ablation"));
    Ok(json!({
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "feature": Value::Object(feature),
    }))
}

pub fn run_document(
    conn: &Connection,
    ctx: &RunContext,
    clips_dir: Option<&Path>,
) -> Result<Value, ApiError> {
    let grid = parse_grid(&ctx.manifest, ctx)?;
    let pixels = clips::resolve_all(&ctx.manifest, &ctx.dir, clips_dir)?;
    let tops = query::top_clips(conn, &acts_path(ctx), None).map_err(db_err)?;
    let mut by_feature: BTreeMap<i64, Vec<&TopClipRow>> = BTreeMap::new();
    for row in &tops {
        by_feature.entry(row.feature_id).or_default().push(row);
    }
    let labels = label_models(ctx)?;
    let mut features = Vec::new();
    for (feature_id, record) in &ctx.features_by_id {
        features.push(feature_record(
            *feature_id,
            record,
            labels.get(feature_id).cloned(),
            by_feature.get(feature_id).map(Vec::as_slice).unwrap_or(&[]),
            &grid,
            &ctx.manifest,
        )?);
    }
    let model = ctx.manifest.get("model").cloned().unwrap_or(Value::Null);
    let sae = ctx.scoreboard.get("sae").cloned().unwrap_or(Value::Null);
    let run_block = ctx.scoreboard.get("run").cloned().unwrap_or(Value::Null);
    Ok(json!({
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "model": {
            "adapter": model.get("adapter").cloned().unwrap_or(Value::Null),
            "name": model.get("name").cloned().unwrap_or(Value::Null),
            "sha": model.get("sha").cloned().unwrap_or(Value::Null),
        },
        "layer": ctx.layer,
        "grid": {
            "t": grid.t,
            "h": grid.h,
            "w": grid.w,
            "tubelet": grid.tubelet,
            "patch_px": grid.patch_px,
        },
        "split": {
            "n_train": json_i64(run_block.get("n_train_clips").unwrap_or(&Value::Null)).unwrap_or(0),
            "n_heldout": json_i64(run_block.get("n_heldout_clips").unwrap_or(&Value::Null)).unwrap_or(0),
        },
        "sae": {
            "usable": sae.get("usable") == Some(&Value::Bool(true)),
            "ev_heldout": number_or_null(sae.get("ev_heldout")),
            "loss_recovered": number_or_null(sae.get("loss_recovered")),
        },
        "clips": clip_summaries(&ctx.manifest, &pixels),
        "features": features,
    }))
}

pub fn top_clips_document(
    conn: &Connection,
    ctx: &RunContext,
    feature_id: i64,
) -> Result<Value, ApiError> {
    let record = ctx
        .features_by_id
        .get(&feature_id)
        .ok_or_else(|| missing_feature(ctx, feature_id))?;
    let grid = parse_grid(&ctx.manifest, ctx)?;
    let rows = query::top_clips(conn, &acts_path(ctx), Some(feature_id)).map_err(db_err)?;
    let borrowed: Vec<&TopClipRow> = rows.iter().collect();
    Ok(json!({
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "feature_id": feature_id,
        "label": label_text(record),
        "badge": badge(record.get("real")),
        "top_clips": top_clip_json(&borrowed, &grid, &ctx.manifest),
    }))
}

pub fn frame_document(
    conn: &Connection,
    ctx: &RunContext,
    clip_id: &str,
    frame: i64,
) -> Result<Value, ApiError> {
    let clips = ctx
        .manifest
        .get("clips")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad("manifest.clips must be a list"))?;
    let clip = clips
        .iter()
        .find(|clip| clip.get("clip_id").and_then(Value::as_str) == Some(clip_id));
    let Some(clip) = clip else {
        return Err(ApiError::not_found(
            format!("run {} has no clip {clip_id}", ctx.run_id),
            Some(ctx.run_id.clone()),
            Some(ctx.model_sha.clone()),
            Some(ctx.layer),
        ));
    };
    let n_frames = json_i64(clip.get("n_frames").unwrap_or(&Value::Null)).unwrap_or(0);
    if frame < 0 || frame >= n_frames {
        return Err(ApiError::not_found(
            format!("run {} clip {clip_id} has no frame {frame}", ctx.run_id),
            Some(ctx.run_id.clone()),
            Some(ctx.model_sha.clone()),
            Some(ctx.layer),
        ));
    }
    let grid = parse_grid(&ctx.manifest, ctx)?;
    let (window, t) = frame_index(frame, &grid);
    let acts = query::acts_at(conn, &acts_path(ctx), clip_id, window, t).map_err(db_err)?;
    let effects =
        query::effects_at(conn, &effects_path(ctx), clip_id, window, t).map_err(db_err)?;
    let mut grouped: BTreeMap<i64, Vec<(i64, i64, f64)>> = BTreeMap::new();
    for cell in acts {
        grouped
            .entry(cell.feature_id)
            .or_default()
            .push((cell.h, cell.w, cell.value));
    }
    let mut effect_groups: HashMap<(i64, String), Vec<f64>> = HashMap::new();
    for EffectCell {
        feature_id,
        effect_kind,
        effect,
    } in effects
    {
        effect_groups
            .entry((feature_id, effect_kind))
            .or_default()
            .push(effect);
    }
    let mut rows = Vec::new();
    for (feature_id, sites) in grouped {
        let mut activation = f64::NEG_INFINITY;
        let mut heatmap = vec![vec![0.0_f64; grid.w.max(0) as usize]; grid.h.max(0) as usize];
        for (h, w, value) in sites {
            activation = activation.max(value);
            if h >= 0 && w >= 0 && (h as usize) < heatmap.len() && (w as usize) < heatmap[0].len() {
                heatmap[h as usize][w as usize] = value;
            }
        }
        if !activation.is_finite() {
            activation = 0.0;
        }
        let record = ctx.features_by_id.get(&feature_id);
        let label = record.map(label_text).unwrap_or_default();
        let badge_text = record
            .map(|row| badge(row.get("real")))
            .unwrap_or("unverified");
        let mut effect_rows = Vec::new();
        for ((id, kind), values) in &effect_groups {
            if *id != feature_id || values.is_empty() {
                continue;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            effect_rows.push((kind.clone(), mean));
        }
        effect_rows.sort_by(|left, right| kind_key(&left.0).cmp(&kind_key(&right.0)));
        rows.push((
            activation,
            feature_id,
            label,
            badge_text,
            effect_rows,
            heatmap,
        ));
    }
    rows.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.1.cmp(&right.1))
    });
    let features: Vec<Value> = rows
        .into_iter()
        .map(
            |(activation, feature_id, label, badge_text, effect_rows, heatmap)| {
                json!({
                    "id": feature_id,
                    "label": label,
                    "badge": badge_text,
                    "activation": finite_f64(activation),
                    "effects": effect_rows.into_iter().map(|(kind, effect)| json!({
                        "effect_kind": kind,
                        "effect": finite_f64(effect),
                    })).collect::<Vec<_>>(),
                    "heatmap": heatmap.into_iter().map(|row| {
                        row.into_iter().map(finite_f64).collect::<Vec<_>>()
                    }).collect::<Vec<_>>(),
                })
            },
        )
        .collect();
    Ok(json!({
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "clip_id": clip_id,
        "frame": frame,
        "window": window,
        "t": t,
        "features": features,
    }))
}

pub fn clip_path(
    ctx: &RunContext,
    clips_dir: Option<&Path>,
    clip_id: &str,
) -> Result<PathBuf, ApiError> {
    let known = ctx
        .manifest
        .get("clips")
        .and_then(Value::as_array)
        .map(|clips| {
            clips
                .iter()
                .any(|clip| clip.get("clip_id").and_then(Value::as_str) == Some(clip_id))
        })
        .unwrap_or(false);
    if !known {
        return Err(ApiError::not_found(
            format!("run {} has no clip {clip_id}", ctx.run_id),
            Some(ctx.run_id.clone()),
            Some(ctx.model_sha.clone()),
            Some(ctx.layer),
        ));
    }
    let resolved = clips::resolve_all(&ctx.manifest, &ctx.dir, clips_dir)?;
    match resolved.get(clip_id) {
        Some(LoadedClip {
            status,
            path: Some(path),
            ..
        }) if status == "ok" => Ok(path.clone()),
        Some(loaded) => Err(ApiError::not_found(
            if loaded.message.is_empty() {
                "Clip file not found.".into()
            } else {
                loaded.message.clone()
            },
            Some(ctx.run_id.clone()),
            Some(ctx.model_sha.clone()),
            Some(ctx.layer),
        )),
        None => Err(ApiError::not_found(
            "Clip file not found. Pass --clips, or record a path in the manifest.",
            Some(ctx.run_id.clone()),
            Some(ctx.model_sha.clone()),
            Some(ctx.layer),
        )),
    }
}

fn feature_record(
    feature_id: i64,
    record: &Value,
    label_model: Option<Value>,
    rows: &[&TopClipRow],
    grid: &Grid,
    manifest: &Value,
) -> Result<Value, ApiError> {
    let mut object = Map::new();
    object.insert("id".into(), json!(feature_id));
    object.insert("label".into(), json!(label_text(record)));
    object.insert("label_model".into(), label_model.unwrap_or(Value::Null));
    object.insert("badge".into(), json!(badge(record.get("real"))));
    object.insert(
        "top_clips".into(),
        Value::Array(top_clip_json(rows, grid, manifest)),
    );
    for key in SCORE_FIELDS {
        let value = record.get(key).cloned().ok_or_else(|| {
            ApiError::bad(format!("scoreboard feature {feature_id} is missing {key}"))
        })?;
        object.insert(key.to_string(), value);
    }
    Ok(Value::Object(object))
}

fn top_clip_json(rows: &[&TopClipRow], grid: &Grid, manifest: &Value) -> Vec<Value> {
    let tubelet = grid.tubelet.max(1);
    let span = grid.t.max(1) * tubelet;
    let mut splits = HashMap::new();
    if let Some(clips) = manifest.get("clips").and_then(Value::as_array) {
        for clip in clips {
            if let Some(id) = clip.get("clip_id").and_then(Value::as_str) {
                splits.insert(
                    id.to_string(),
                    clip.get("split").cloned().unwrap_or(Value::Null),
                );
            }
        }
    }
    rows.iter()
        .map(|row| {
            json!({
                "clip_id": row.clip_id,
                "split": splits.get(&row.clip_id).cloned().unwrap_or(Value::Null),
                "window": row.window,
                "t": row.t,
                "h": row.h,
                "w": row.w,
                "value": finite_f64(row.value),
                "frame": row.window * span + row.t * tubelet,
            })
        })
        .collect()
}

fn clip_summaries(manifest: &Value, pixels: &HashMap<String, LoadedClip>) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen = HashMap::<String, usize>::new();
    let Some(clips) = manifest.get("clips").and_then(Value::as_array) else {
        return out;
    };
    for clip in clips {
        let clip_id = clip
            .get("clip_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let status = pixels
            .get(&clip_id)
            .map(|loaded| loaded.status.as_str())
            .unwrap_or("missing");
        let summary = json!({
            "clip_id": clip_id,
            "fps": fps_of(clip),
            "n_frames": json_i64(clip.get("n_frames").unwrap_or(&Value::Null)).unwrap_or(0),
            "split": clip.get("split").cloned().unwrap_or(Value::Null),
            "pixels": status,
        });
        if let Some(index) = seen.get(&clip_id) {
            out[*index] = summary;
        } else {
            seen.insert(clip_id, out.len());
            out.push(summary);
        }
    }
    out
}

fn label_models(ctx: &RunContext) -> Result<HashMap<i64, Value>, ApiError> {
    let path = ctx.dir.join("features").join("features.jsonl");
    if !path.is_file() {
        return Ok(HashMap::new());
    }
    let rows = read_jsonl(&path)?;
    let mut labels = HashMap::new();
    for row in rows {
        let Some(id) = row.get("id").and_then(json_i64) else {
            continue;
        };
        labels.insert(id, row.get("label_model").cloned().unwrap_or(Value::Null));
    }
    Ok(labels)
}

fn frame_index(frame: i64, grid: &Grid) -> (i64, i64) {
    let tubelet = grid.tubelet.max(1);
    let bins = grid.t.max(1);
    let span = bins * tubelet;
    let window = frame / span;
    let t = (frame % span) / tubelet;
    (window, t)
}

fn parse_grid(manifest: &Value, ctx: &RunContext) -> Result<Grid, ApiError> {
    let grid = manifest
        .get("grid")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::bad(format!(
                "run {} manifest.grid must be an object",
                ctx.run_id
            ))
        })?;
    let number = |key: &str| -> Result<i64, ApiError> {
        json_i64(grid.get(key).unwrap_or(&Value::Null)).ok_or_else(|| {
            ApiError::bad(format!(
                "run {} manifest.grid.{key} must be an integer",
                ctx.run_id
            ))
        })
    };
    Ok(Grid {
        t: number("t")?,
        h: number("h")?,
        w: number("w")?,
        tubelet: number("tubelet")?,
        patch_px: number("patch_px")?,
    })
}

fn kind_key(kind: &str) -> (u8, &str) {
    match kind {
        "ablation" => (0, kind),
        "estimate" => (1, kind),
        _ => (2, kind),
    }
}

fn badge(real: Option<&Value>) -> &'static str {
    if real == Some(&Value::Bool(true)) {
        "real"
    } else {
        "unverified"
    }
}

fn label_text(record: &Value) -> String {
    record
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn fps_of(clip: &Value) -> Value {
    match clip.get("fps") {
        Some(Value::Number(number)) => {
            let value = number.as_f64().unwrap_or(0.0);
            if value == 0.0 {
                finite_f64(0.0)
            } else {
                finite_f64(value)
            }
        }
        _ => finite_f64(0.0),
    }
}

fn number_or_null(value: Option<&Value>) -> Value {
    match value {
        Some(Value::Number(number)) => finite_f64(number.as_f64().unwrap_or(f64::NAN)),
        Some(Value::Bool(_)) | None => Value::Null,
        Some(other) => other.clone(),
    }
}

fn missing_feature(ctx: &RunContext, feature_id: i64) -> ApiError {
    ApiError::not_found(
        format!("run {} has no feature {feature_id}", ctx.run_id),
        Some(ctx.run_id.clone()),
        Some(ctx.model_sha.clone()),
        Some(ctx.layer),
    )
}

fn acts_path(ctx: &RunContext) -> PathBuf {
    ctx.dir.join("features").join("feature_acts.parquet")
}

fn effects_path(ctx: &RunContext) -> PathBuf {
    ctx.dir.join("features").join("clip_effects.parquet")
}

fn read_object(path: &Path) -> Result<Value, ApiError> {
    let text = std::fs::read_to_string(path).map_err(io_err)?;
    serde_json::from_str(&text).map_err(|err| ApiError::bad(format!("{}: {err}", path.display())))
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, ApiError> {
    let text = std::fs::read_to_string(path).map_err(io_err)?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str(line).map_err(|err| {
            ApiError::bad(format!(
                "{}:{} is not JSON ({err})",
                path.display(),
                index + 1
            ))
        })?;
        rows.push(value);
    }
    Ok(rows)
}

fn json_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|value| i64::try_from(value).ok())),
        _ => None,
    }
}
