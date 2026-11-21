//! The local viewer for one run folder (`ii view`).
//!
//! `/api/run` is built once at load. Clip payloads, top clips and heatmaps are queried from
//! `features/feature_acts.parquet` through DuckDB when they are first asked for; the parquet is
//! never loaded into memory. Clip pixels come from a path on the manifest or from `--clips`: the
//! clip's `name` first, then the filename stem, then sha256.

pub mod frames;
pub mod http;
pub mod windows;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use duckdb::{params, Connection};
use serde_json::{json, Map, Value};

use crate::clips::file_sha256;
use crate::npy;
use frames::ClipArray;
use windows::{as_int, clip_sampling, frame_to_token, starts_covering, token_to_frame, Grid};

const REQUIRED: [&str; 2] = ["manifest.json", "features/feature_acts.parquet"];
/// Stem matches prefer a browser-native video over a transcode, and a transcode over npy.
const MEDIA_SUFFIXES: [&str; 9] = ["mp4", "webm", "m4v", "mov", "avi", "mkv", "mpeg", "mpg", "npy"];
pub const NATIVE_VIDEO: [(&str, &str); 3] = [("mp4", "video/mp4"), ("webm", "video/webm"), ("m4v", "video/mp4")];
const TRANSCODE_VIDEO: [&str; 5] = ["mov", "avi", "mkv", "mpeg", "mpg"];
const SCORE_FIELDS: [&str; 7] = [
    "distinct_clips_top20",
    "label_holds_heldout",
    "ablation_effect",
    "random_percentile",
    "n_random",
    "absent_in_random_init",
    "real",
];
const TOP_FEATURES: i64 = 40;
const TOP_CLIPS: i64 = 20;
const SHA_EAGER_LIMIT: u64 = 8 * 1024 * 1024;
const MISSING: &str = "Clip file not found. Pass --clips, or record a path in the manifest.";

#[derive(Debug, Clone)]
pub struct LoadedClip {
    pub status: String,
    pub message: String,
    /// `array`, `video`, or `none`.
    pub kind: &'static str,
    pub path: Option<PathBuf>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

impl LoadedClip {
    fn missing() -> Self {
        Self::status("missing", MISSING, None)
    }

    fn status(status: &str, message: impl Into<String>, path: Option<PathBuf>) -> Self {
        Self {
            status: status.into(),
            message: message.into(),
            kind: "none",
            path,
            width: None,
            height: None,
        }
    }

    pub fn media(&self) -> &'static str {
        match (self.kind, self.status.as_str()) {
            ("video", "ok") => "video",
            ("array", "ok") => "frames",
            _ => "none",
        }
    }
}

/// One loaded run folder. Cheap to share behind an `Arc`.
pub struct RunView {
    pub run_dir: PathBuf,
    pub run_payload: Value,
    pub grid: Grid,
    pub extra: Option<Map<String, Value>>,
    pub adapter: Option<String>,
    pub manifest_clips: HashMap<String, Map<String, Value>>,
    pub loaded: HashMap<String, LoadedClip>,
    pub features_by_id: BTreeMap<i64, Value>,
    viz: HashMap<i64, PathBuf>,
    acts: Acts,
    clip_cache: Mutex<HashMap<String, Arc<Value>>>,
    arrays: Mutex<HashMap<String, Arc<ClipArray>>>,
    png_cache: Mutex<HashMap<(String, i64), Arc<Vec<u8>>>>,
}

pub fn viewer_ready(run_dir: &Path) -> bool {
    REQUIRED.iter().all(|rel| run_dir.join(rel).is_file())
}

fn nearby_runs(run_dir: &Path) -> String {
    let parent = run_dir.parent().unwrap_or(Path::new("."));
    let mut found: Vec<String> = std::fs::read_dir(parent)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().join("manifest.json").is_file())
                .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    if found.is_empty() {
        return ". `ii run` (or `ii try`) writes one; pass the folder it printed.".into();
    }
    let mut shown = found.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
    if found.len() > 8 {
        shown.push_str(", ...");
    }
    format!(". Run folders in {}: {shown}", parent.display())
}

impl RunView {
    pub fn load(run_dir: &Path, clips_dir: Option<&Path>) -> Result<RunView, String> {
        if !run_dir.is_dir() {
            return Err(format!("run folder not found: {}{}", run_dir.display(), nearby_runs(run_dir)));
        }
        let missing: Vec<&str> = REQUIRED.iter().copied().filter(|rel| !run_dir.join(rel).is_file()).collect();
        if !missing.is_empty() {
            return Err(format!("{} is missing {}", run_dir.display(), missing.join(", ")));
        }
        if let Some(dir) = clips_dir {
            if !dir.is_dir() {
                return Err(format!("clips directory not found: {}", dir.display()));
            }
        }
        let manifest = read_json(&run_dir.join("manifest.json"))?;
        let manifest = manifest.as_object().ok_or("manifest must be an object")?.clone();
        require_keys(&manifest, &["run_id", "model", "layer", "grid", "clips"], "manifest")?;
        let raw_grid = manifest["grid"].as_object().ok_or("manifest.grid must be an object")?;
        require_keys(raw_grid, &["t", "h", "w"], "manifest.grid")?;
        let model = manifest["model"].as_object().ok_or("manifest.model must be an object")?;
        let clips = manifest["clips"].as_array().ok_or("manifest.clips must be a list")?;

        let scoreboard = read_optional_object(&run_dir.join("scoreboard.json"))?;
        let labels = stream_labels(&run_dir.join("features").join("features.jsonl"))?;
        let candidates = read_candidates(&run_dir.join("features").join("candidates.json"))?;
        let acts_path = run_dir.join("features").join("feature_acts.parquet");
        let effects_path = run_dir.join("features").join("clip_effects.parquet");
        let acts = Acts::open(&acts_path, effects_path.is_file().then_some(effects_path.as_path()))?;

        let grid = Grid::parse(raw_grid);
        let extra = manifest.get("extra").and_then(Value::as_object).cloned();
        let (features, viz) = feature_index(scoreboard.as_ref(), &labels, &candidates, run_dir);
        let moments = read_moments(&run_dir.join("heldout_loss.json"));
        let steering = read_optional_object(&run_dir.join("steering").join("scorecard.json"))
            .ok()
            .flatten()
            .filter(|card| card.get("rows").map(Value::is_array).unwrap_or(false));
        let pixels = resolve_pixels(&manifest, run_dir, clips_dir);

        let mut manifest_clips = HashMap::new();
        let mut loaded = HashMap::new();
        let mut meta = Vec::new();
        for clip in clips {
            let Some(clip) = clip.as_object() else { continue };
            let Some(raw_id) = clip.get("clip_id") else { continue };
            let clip_id = py_str(raw_id);
            let info = pixels.get(&clip_id).cloned().unwrap_or_else(LoadedClip::missing);
            meta.push(json!({
                "clip_id": clip_id,
                "fps": fps_of(clip),
                "n_frames": clip.get("n_frames").and_then(as_int).unwrap_or(0),
                "split": clip.get("split").cloned().unwrap_or(Value::Null),
                "pixels": info.status,
                "media": info.media(),
            }));
            manifest_clips.insert(clip_id.clone(), clip.clone());
            loaded.insert(clip_id, info);
        }
        let adapter = model.get("adapter").and_then(Value::as_str).map(str::to_string);
        let run_payload = json!({
            "run_id": manifest["run_id"],
            "model": {
                "adapter": model.get("adapter").cloned().unwrap_or(Value::Null),
                "name": model.get("name").cloned().unwrap_or(Value::Null),
                "sha": model.get("sha").cloned().unwrap_or(Value::Null),
            },
            "layer": loose_i64(&manifest["layer"]).ok_or("manifest.layer must be an integer")?,
            "grid": grid.to_json(),
            "scoreboard": scoreboard.is_some(),
            "split": split_payload(scoreboard.as_ref(), clips),
            "sae": sae_payload(scoreboard.as_ref()),
            "clips": meta,
            "features": features,
            "moments": moments,
            "steering": steering.map(Value::Object).unwrap_or(Value::Null),
        });
        let features_by_id = run_payload["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|feature| (feature["id"].as_i64().unwrap(), feature.clone()))
            .collect();
        Ok(RunView {
            run_dir: run_dir.to_path_buf(),
            run_payload,
            grid,
            extra,
            adapter,
            manifest_clips,
            loaded,
            features_by_id,
            viz,
            acts,
            clip_cache: Mutex::new(HashMap::new()),
            arrays: Mutex::new(HashMap::new()),
            png_cache: Mutex::new(HashMap::new()),
        })
    }

    /// One feature, with its top clips queried on demand.
    pub fn feature(&self, feature_id: i64) -> Result<Option<Value>, String> {
        let base = self.features_by_id.get(&feature_id);
        let tops = self.top_for(feature_id)?;
        if base.is_none() && tops.is_empty() {
            return Ok(None);
        }
        let mut detail = base.cloned().unwrap_or_else(|| blank_feature(feature_id));
        detail["top_clips"] = Value::Array(tops);
        detail["viz"] = Value::Bool(self.viz_file(feature_id).is_some());
        Ok(Some(detail))
    }

    pub fn viz_file(&self, feature_id: i64) -> Option<PathBuf> {
        if let Some(found) = self.viz.get(&feature_id) {
            return Some(found.clone());
        }
        default_viz(&self.run_dir, feature_id)
    }

    fn top_for(&self, feature_id: i64) -> Result<Vec<Value>, String> {
        let empty = Map::new();
        let rows = self.acts.top_clips(feature_id)?;
        Ok(rows
            .into_iter()
            .map(|(clip_id, window, t, h, w, value)| {
                let clip = self.manifest_clips.get(&clip_id).unwrap_or(&empty);
                let (starts, stride) = clip_sampling(Some(clip), &self.grid, window, self.extra.as_ref());
                json!({
                    "clip_id": clip_id,
                    "split": clip.get("split").cloned().unwrap_or(Value::Null),
                    "window": window,
                    "t": t,
                    "h": h,
                    "w": w,
                    "value": value,
                    "frame": token_to_frame(window, t, &starts, &self.grid, stride),
                })
            })
            .collect())
    }

    /// Patch grid for one token. Empty cells are 0. `None` when the frame is not in the clip.
    pub fn heatmap(&self, clip_id: &str, frame: i64, feature_id: i64) -> Result<Option<Vec<Vec<f64>>>, String> {
        let Some(clip) = self.manifest_clips.get(clip_id) else {
            return Ok(None);
        };
        let n_frames = clip.get("n_frames").and_then(as_int).unwrap_or(0);
        if frame < 0 || frame >= n_frames {
            return Ok(None);
        }
        let (height, width) = (self.grid.h.max(0) as usize, self.grid.w.max(0) as usize);
        let mut grid = vec![vec![0.0f64; width]; height];
        let (starts, stride) = starts_covering(Some(clip), &self.grid, self.extra.as_ref(), frame);
        let (window, t) = frame_to_token(frame, &starts, &self.grid, stride);
        let Some(t) = t else {
            return Ok(Some(grid));
        };
        for (h, w, value) in self.acts.cells(clip_id, window, t, feature_id)? {
            if (0..height as i64).contains(&h) && (0..width as i64).contains(&w) {
                grid[h as usize][w as usize] = value;
            }
        }
        Ok(Some(grid))
    }

    /// Per-frame top features for one clip, built the first time it is asked for.
    pub fn clip(&self, clip_id: &str) -> Result<Option<Arc<Value>>, String> {
        if let Some(cached) = self.clip_cache.lock().unwrap().get(clip_id) {
            return Ok(Some(cached.clone()));
        }
        let Some(clip) = self.manifest_clips.get(clip_id) else {
            return Ok(None);
        };
        let loaded = self.loaded.get(clip_id).cloned().unwrap_or_else(LoadedClip::missing);
        let timeline = self.acts.timeline(clip_id)?;
        let effects = self.acts.effects(clip_id)?;
        let payload = Arc::new(self.clip_payload(clip_id, clip, &loaded, &timeline, &effects));
        self.clip_cache.lock().unwrap().insert(clip_id.to_string(), payload.clone());
        Ok(Some(payload))
    }

    fn clip_payload(
        &self,
        clip_id: &str,
        clip: &Map<String, Value>,
        loaded: &LoadedClip,
        timeline: &[(i64, i64, i64, f64)],
        effect_rows: &[(i64, i64, i64, Option<String>, Option<f64>)],
    ) -> Value {
        let n_frames = clip.get("n_frames").and_then(as_int).unwrap_or(0);
        let mut grouped: HashMap<(i64, i64), Vec<(i64, f64)>> = HashMap::new();
        let mut max_window = 0;
        for (window, t, feature_id, activation) in timeline {
            max_window = max_window.max(*window);
            grouped.entry((*window, *t)).or_default().push((*feature_id, *activation));
        }
        let mut effects: HashMap<(i64, i64, i64), Vec<(String, f64)>> = HashMap::new();
        for (feature_id, window, t, kind, effect) in effect_rows {
            max_window = max_window.max(*window);
            let (Some(kind), Some(effect)) = (kind, effect.filter(|value| value.is_finite())) else {
                continue;
            };
            effects.entry((*feature_id, *window, *t)).or_default().push((kind.clone(), effect));
        }
        for bucket in effects.values_mut() {
            bucket.sort_by(|a, b| (kind_order(&a.0), &a.0).cmp(&(kind_order(&b.0), &b.0)));
        }
        let (starts, stride) = clip_sampling(Some(clip), &self.grid, max_window, self.extra.as_ref());
        let hw = displayed_hw(clip, loaded);
        let crop = overlay_rect(hw, self.extra.as_ref(), self.adapter.as_deref());
        let mut frames = Vec::with_capacity(n_frames.max(0) as usize);
        for frame in 0..n_frames {
            let (window, t) = frame_to_token(frame, &starts, &self.grid, stride);
            let mut features = Vec::new();
            if let Some(t) = t {
                let mut rows: Vec<(f64, i64, f64, i64, Value)> = Vec::new();
                for (feature_id, activation) in grouped.get(&(window, t)).map(Vec::as_slice).unwrap_or(&[]) {
                    let meta = self.features_by_id.get(feature_id);
                    let field = |key: &str| meta.and_then(|meta| meta.get(key)).cloned().unwrap_or(Value::Null);
                    let label = meta.and_then(|meta| meta.get("label")).and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("");
                    let badge = meta.and_then(|meta| meta.get("badge")).and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("unverified");
                    let status = field("status");
                    let passed = if status.as_str().map(ablation_passed).unwrap_or(false) { 0 } else { 1 };
                    let cited: Vec<Value> = effects
                        .get(&(*feature_id, window, t))
                        .map(|bucket| bucket.iter().map(|(kind, effect)| json!({"effect_kind": kind, "effect": effect})).collect())
                        .unwrap_or_default();
                    rows.push((
                        -round1(*activation),
                        passed,
                        -*activation,
                        *feature_id,
                        json!({
                            "id": feature_id,
                            "label": label,
                            "badge": badge,
                            "status": status,
                            "activation": activation,
                            "effects": cited,
                        }),
                    ));
                }
                rows.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                        .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                        .then(a.3.cmp(&b.3))
                });
                features = rows.into_iter().map(|row| row.4).collect();
            }
            frames.push(json!({"frame": frame, "window": window, "t": t, "features": features}));
        }
        json!({
            "clip_id": clip_id,
            "fps": fps_of(clip),
            "n_frames": n_frames,
            "split": clip.get("split").cloned().unwrap_or(Value::Null),
            "pixels": loaded.status,
            "media": loaded.media(),
            "message": loaded.message,
            "width": loaded.width,
            "height": loaded.height,
            "hw": hw.map(|(h, w)| vec![h, w]),
            "crop": crop,
            "layout": token_layout(self.adapter.as_deref()),
            "window_starts": starts,
            "stride": stride,
            "grid": self.grid.clip_json(),
            "frames": frames,
        })
    }

    /// PNG of one frame of an array clip. `None` for a missing clip, a video, or a bad index.
    pub fn frame_png(&self, clip_id: &str, frame: i64) -> Result<Option<Arc<Vec<u8>>>, String> {
        let Some(loaded) = self.loaded.get(clip_id) else {
            return Ok(None);
        };
        if loaded.status != "ok" || loaded.kind != "array" {
            return Ok(None);
        }
        let n_frames = self.manifest_clips[clip_id].get("n_frames").and_then(as_int).unwrap_or(0);
        if frame < 0 || frame >= n_frames {
            return Ok(None);
        }
        let key = (clip_id.to_string(), frame);
        if let Some(cached) = self.png_cache.lock().unwrap().get(&key) {
            return Ok(Some(cached.clone()));
        }
        let array = self.array(clip_id, loaded)?;
        if frame as usize >= array.frames {
            return Ok(None);
        }
        let rgb = frames::render_frame(&array, frame as usize, array.peak());
        let (big, height, width) = frames::upsample(&rgb, array.height, array.width);
        let png = Arc::new(frames::encode_png(&big, height, width));
        self.png_cache.lock().unwrap().insert(key, png.clone());
        Ok(Some(png))
    }

    fn array(&self, clip_id: &str, loaded: &LoadedClip) -> Result<Arc<ClipArray>, String> {
        if let Some(found) = self.arrays.lock().unwrap().get(clip_id) {
            return Ok(found.clone());
        }
        let path = loaded.path.as_ref().ok_or("clip has no file")?;
        let array = Arc::new(frames::load_npy(path)?);
        self.arrays.lock().unwrap().insert(clip_id.to_string(), array.clone());
        Ok(array)
    }

    /// A browser-playable file for a video clip, and its content type.
    pub fn media_file(&self, clip_id: &str) -> Option<(PathBuf, &'static str)> {
        let loaded = self.loaded.get(clip_id)?;
        if loaded.media() != "video" {
            return None;
        }
        let path = loaded.path.as_ref()?;
        let suffix = suffix_of(path);
        NATIVE_VIDEO
            .iter()
            .find(|(ext, _)| *ext == suffix)
            .map(|(_, content_type)| (path.clone(), *content_type))
    }
}

// ---------------------------------------------------------------------------------------------
// DuckDB

struct Acts {
    acts_path: String,
    effects_path: Option<String>,
    clip_is_int: bool,
    effects_clip_is_int: bool,
    con: Mutex<Connection>,
}

enum ClipParam {
    Int(i64),
    Text(String),
    Never,
}

impl Acts {
    fn open(acts: &Path, effects: Option<&Path>) -> Result<Acts, String> {
        let con = Connection::open_in_memory().map_err(|err| format!("duckdb: {err}"))?;
        let _ = con.execute_batch("SET memory_limit='768MB'; SET threads=2; SET preserve_insertion_order=false;");
        let required = ["feature_id", "clip_id", "window", "t", "h", "w", "value"];
        let acts_path = acts.to_string_lossy().into_owned();
        let clip_is_int = column_check(&con, &acts_path, &required, "feature_acts.parquet")?;
        let mut effects_clip_is_int = false;
        let effects_path = match effects {
            Some(path) => {
                let text = path.to_string_lossy().into_owned();
                effects_clip_is_int = column_check(
                    &con,
                    &text,
                    &["feature_id", "clip_id", "window", "t", "effect", "effect_kind"],
                    "clip_effects.parquet",
                )?;
                Some(text)
            }
            None => None,
        };
        Ok(Acts {
            acts_path,
            effects_path,
            clip_is_int,
            effects_clip_is_int,
            con: Mutex::new(con),
        })
    }

    fn param(clip_id: &str, integer: bool) -> ClipParam {
        if integer {
            match clip_id.trim().parse::<i64>() {
                Ok(value) => ClipParam::Int(value),
                Err(_) => ClipParam::Never,
            }
        } else {
            ClipParam::Text(clip_id.to_string())
        }
    }

    fn timeline(&self, clip_id: &str) -> Result<Vec<(i64, i64, i64, f64)>, String> {
        let sql = "WITH summed AS (
                SELECT CAST(\"window\" AS BIGINT) AS \"window\", CAST(\"t\" AS BIGINT) AS \"t\",
                       CAST(feature_id AS BIGINT) AS feature_id, sum(value)::DOUBLE AS activation
                FROM read_parquet(?) WHERE clip_id = ?
                GROUP BY \"window\", \"t\", feature_id
            ),
            ranked AS (
                SELECT \"window\", \"t\", feature_id, activation,
                       row_number() OVER (PARTITION BY \"window\", \"t\" ORDER BY activation DESC, feature_id) AS rn
                FROM summed
            )
            SELECT \"window\", \"t\", feature_id, activation FROM ranked WHERE rn <= ?";
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(sql).map_err(db)?;
        let mut rows = match Self::param(clip_id, self.clip_is_int) {
            ClipParam::Int(id) => stmt.query(params![self.acts_path, id, TOP_FEATURES]),
            ClipParam::Text(id) => stmt.query(params![self.acts_path, id, TOP_FEATURES]),
            ClipParam::Never => return Ok(Vec::new()),
        }
        .map_err(db)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().map_err(db)? {
            out.push((row.get(0).map_err(db)?, row.get(1).map_err(db)?, row.get(2).map_err(db)?, row.get(3).map_err(db)?));
        }
        Ok(out)
    }

    #[allow(clippy::type_complexity)]
    fn effects(&self, clip_id: &str) -> Result<Vec<(i64, i64, i64, Option<String>, Option<f64>)>, String> {
        let Some(path) = &self.effects_path else {
            return Ok(Vec::new());
        };
        let sql = "SELECT CAST(feature_id AS BIGINT), CAST(\"window\" AS BIGINT), CAST(\"t\" AS BIGINT),
                          CAST(effect_kind AS VARCHAR), avg(effect)::DOUBLE
                   FROM read_parquet(?) WHERE clip_id = ?
                   GROUP BY feature_id, \"window\", \"t\", effect_kind";
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(sql).map_err(db)?;
        let mut rows = match Self::param(clip_id, self.effects_clip_is_int) {
            ClipParam::Int(id) => stmt.query(params![path, id]),
            ClipParam::Text(id) => stmt.query(params![path, id]),
            ClipParam::Never => return Ok(Vec::new()),
        }
        .map_err(db)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().map_err(db)? {
            out.push((
                row.get(0).map_err(db)?,
                row.get(1).map_err(db)?,
                row.get(2).map_err(db)?,
                row.get(3).map_err(db)?,
                row.get(4).map_err(db)?,
            ));
        }
        Ok(out)
    }

    fn cells(&self, clip_id: &str, window: i64, t: i64, feature_id: i64) -> Result<Vec<(i64, i64, f64)>, String> {
        let sql = "SELECT CAST(h AS BIGINT), CAST(w AS BIGINT), max(value)::DOUBLE
                   FROM read_parquet(?)
                   WHERE clip_id = ? AND \"window\" = ? AND \"t\" = ? AND feature_id = ?
                   GROUP BY h, w";
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(sql).map_err(db)?;
        let mut rows = match Self::param(clip_id, self.clip_is_int) {
            ClipParam::Int(id) => stmt.query(params![self.acts_path, id, window, t, feature_id]),
            ClipParam::Text(id) => stmt.query(params![self.acts_path, id, window, t, feature_id]),
            ClipParam::Never => return Ok(Vec::new()),
        }
        .map_err(db)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().map_err(db)? {
            out.push((row.get(0).map_err(db)?, row.get(1).map_err(db)?, row.get(2).map_err(db)?));
        }
        Ok(out)
    }

    #[allow(clippy::type_complexity)]
    fn top_clips(&self, feature_id: i64) -> Result<Vec<(String, i64, i64, i64, i64, f64)>, String> {
        let sql = "WITH ranked AS (
                SELECT CAST(clip_id AS VARCHAR) AS clip_id, CAST(\"window\" AS BIGINT) AS \"window\",
                       CAST(\"t\" AS BIGINT) AS \"t\", CAST(h AS BIGINT) AS h, CAST(w AS BIGINT) AS w,
                       value::DOUBLE AS value,
                       row_number() OVER (PARTITION BY clip_id ORDER BY value DESC, \"window\", \"t\", h, w) AS rn
                FROM read_parquet(?) WHERE feature_id = ?
            )
            SELECT clip_id, \"window\", \"t\", h, w, value FROM ranked WHERE rn = 1
            ORDER BY value DESC, clip_id LIMIT ?";
        let con = self.con.lock().unwrap();
        let mut stmt = con.prepare(sql).map_err(db)?;
        let mut rows = stmt.query(params![self.acts_path, feature_id, TOP_CLIPS]).map_err(db)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().map_err(db)? {
            out.push((
                row.get(0).map_err(db)?,
                row.get(1).map_err(db)?,
                row.get(2).map_err(db)?,
                row.get(3).map_err(db)?,
                row.get(4).map_err(db)?,
                row.get(5).map_err(db)?,
            ));
        }
        Ok(out)
    }
}

fn db(err: duckdb::Error) -> String {
    format!("duckdb: {err}")
}

/// Checks the required columns; returns whether `clip_id` is an integer column.
fn column_check(con: &Connection, path: &str, required: &[&str], name: &str) -> Result<bool, String> {
    let mut stmt = con
        .prepare("SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))")
        .map_err(db)?;
    let mut rows = stmt.query(params![path]).map_err(db)?;
    let mut columns: HashMap<String, String> = HashMap::new();
    while let Some(row) = rows.next().map_err(db)? {
        columns.insert(row.get(0).map_err(db)?, row.get(1).map_err(db)?);
    }
    let missing: Vec<&str> = required.iter().copied().filter(|column| !columns.contains_key(*column)).collect();
    if !missing.is_empty() {
        return Err(format!("{name} is missing columns {}", missing.join(", ")));
    }
    let clip_type = columns.get("clip_id").map(String::as_str).unwrap_or("");
    Ok(matches!(
        clip_type,
        "TINYINT" | "SMALLINT" | "INTEGER" | "BIGINT" | "HUGEINT" | "UTINYINT" | "USMALLINT" | "UINTEGER" | "UBIGINT"
    ))
}

// ---------------------------------------------------------------------------------------------
// Run folder files

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("could not read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("{} is not valid JSON: {err}", path.display()))
}

fn read_optional_object(path: &Path) -> Result<Option<Map<String, Value>>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(read_json(path)?.as_object().cloned())
}

fn require_keys(object: &Map<String, Value>, keys: &[&str], name: &str) -> Result<(), String> {
    let missing: Vec<&str> = keys.iter().copied().filter(|key| !object.contains_key(*key)).collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("{name} is missing {}", missing.join(", ")))
    }
}

struct Label {
    label: String,
    label_model: Value,
}

fn stream_labels(path: &Path) -> Result<HashMap<i64, Label>, String> {
    let mut found = HashMap::new();
    if !path.is_file() {
        return Ok(found);
    }
    let text = std::fs::read_to_string(path).map_err(|err| format!("could not read {}: {err}", path.display()))?;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line).map_err(|_| format!("{} has a line that is not JSON", path.display()))?;
        let Some(row) = row.as_object() else { continue };
        let Some(id) = row.get("id").and_then(as_int) else { continue };
        found.insert(
            id,
            Label {
                label: row.get("label").and_then(Value::as_str).unwrap_or("").to_string(),
                label_model: row.get("label_model").filter(|value| value.is_string()).cloned().unwrap_or(Value::Null),
            },
        );
    }
    Ok(found)
}

fn read_candidates(path: &Path) -> Result<Vec<Map<String, Value>>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let mut payload = read_json(path)?;
    if let Value::Object(object) = &payload {
        let ranked = object.get("ranked");
        payload = match ranked {
            Some(Value::Array(items)) if items.first().map(Value::is_object).unwrap_or(false) => Value::Array(items.clone()),
            _ => ["candidates", "features"]
                .iter()
                .find_map(|key| object.get(*key).filter(|value| truthy(value)).cloned())
                .unwrap_or(Value::Array(Vec::new())),
        };
    }
    let Value::Array(items) = payload else {
        return Ok(Vec::new());
    };
    Ok(items
        .into_iter()
        .filter_map(|item| match item {
            Value::Object(row) if row.get("feature_id").and_then(as_int).is_some() => Some(row),
            _ => None,
        })
        .collect())
}

fn blank_feature(feature_id: i64) -> Value {
    let mut record = json!({
        "id": feature_id,
        "label": "",
        "label_model": null,
        "badge": "unverified",
        "status": null,
        "scored": false,
        "candidate": false,
    });
    for key in SCORE_FIELDS {
        record[key] = Value::Null;
    }
    record
}

fn feature_index(
    scoreboard: Option<&Map<String, Value>>,
    labels: &HashMap<i64, Label>,
    candidates: &[Map<String, Value>],
    run_dir: &Path,
) -> (Vec<Value>, HashMap<i64, PathBuf>) {
    let mut scored: HashMap<i64, &Map<String, Value>> = HashMap::new();
    if let Some(Value::Array(rows)) = scoreboard.and_then(|board| board.get("features")) {
        for row in rows {
            if let Some(row) = row.as_object() {
                if let Some(id) = row.get("id").and_then(as_int) {
                    scored.insert(id, row);
                }
            }
        }
    }
    let mut ordered = Vec::new();
    let mut seen = HashSet::new();
    let mut viz = HashMap::new();
    for row in candidates {
        let id = row["feature_id"].as_i64().unwrap();
        if !seen.insert(id) {
            continue;
        }
        ordered.push(id);
        if let Some(picture) = resolve_viz(run_dir, id, row.get("viz")) {
            viz.insert(id, picture);
        }
    }
    let mut rest: Vec<i64> = scored.keys().chain(labels.keys()).copied().filter(|id| !seen.contains(id)).collect();
    rest.sort();
    rest.dedup();
    ordered.extend(rest);
    let mut records = Vec::new();
    for id in ordered {
        let meta = labels.get(&id);
        let jsonl_label = meta.map(|meta| meta.label.clone()).unwrap_or_default();
        let label_model = meta.map(|meta| meta.label_model.clone()).unwrap_or(Value::Null);
        let mut record = match scored.get(&id) {
            None => {
                let mut record = blank_feature(id);
                record["label"] = jsonl_label.into();
                record["label_model"] = label_model;
                record
            }
            Some(row) => {
                let label = match row.get("label") {
                    Some(Value::String(label)) => label.clone(),
                    _ => jsonl_label,
                };
                let status = status_text(row.get("status"));
                let mut record = json!({
                    "id": id,
                    "label": label,
                    "label_model": label_model,
                    "badge": badge_for(row.get("real"), status.as_deref()),
                    "status": status,
                    "scored": true,
                    "candidate": seen.contains(&id),
                });
                for key in SCORE_FIELDS {
                    record[key] = row.get(key).map(json_value).unwrap_or(Value::Null);
                }
                record
            }
        };
        record["candidate"] = Value::Bool(seen.contains(&id));
        records.push(record);
    }
    (records, viz)
}

fn default_viz(run_dir: &Path, feature_id: i64) -> Option<PathBuf> {
    for name in [format!("feature_{feature_id:05}.png"), format!("feature_{feature_id}.png")] {
        let candidate = run_dir.join("features").join("viz").join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn resolve_viz(run_dir: &Path, feature_id: i64, raw: Option<&Value>) -> Option<PathBuf> {
    if let Some(Value::String(raw)) = raw {
        let relative = Path::new(raw);
        let escapes = relative.components().any(|part| matches!(part, std::path::Component::ParentDir));
        if !raw.is_empty() && !escapes && !relative.is_absolute() {
            if let (Ok(root), Ok(candidate)) = (run_dir.canonicalize(), run_dir.join(relative).canonicalize()) {
                if candidate.starts_with(&root) && candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    default_viz(run_dir, feature_id)
}

fn read_moments(path: &Path) -> Vec<Value> {
    let Ok(payload) = read_json(path) else {
        return Vec::new();
    };
    let payload = match payload {
        Value::Object(object) => ["moments", "losses", "windows"]
            .iter()
            .find_map(|key| object.get(*key).filter(|value| truthy(value)).cloned())
            .unwrap_or(Value::Array(Vec::new())),
        other => other,
    };
    let Value::Array(items) = payload else {
        return Vec::new();
    };
    let mut rows: Vec<(f64, usize, Value)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let Some(item) = item.as_object() else { continue };
        let Some(clip_id) = item.get("clip_id").and_then(as_int) else { continue };
        let (Some(mut start), Some(mut end)) = (
            item.get("frame_start").and_then(as_int),
            item.get("frame_end").and_then(as_int),
        ) else {
            continue;
        };
        let Some(loss) = item.get("loss").and_then(num) else { continue };
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let task = item
            .get("task_text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| Value::String(text.to_string()))
            .unwrap_or(Value::Null);
        let clip_id = clip_id.to_string();
        let name = item
            .get("clip_name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| clip_id.clone());
        rows.push((
            loss,
            index,
            json!({
                "clip_id": clip_id,
                "clip_name": name,
                "window": item.get("window").and_then(as_int),
                "frame_start": start,
                "frame_end": end,
                "frame": (start + end).div_euclid(2),
                "loss": loss,
                "task_text": task,
                "loss_source": item.get("loss_source").filter(|value| value.is_string()).cloned().unwrap_or(Value::Null),
            }),
        ));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    rows.into_iter().map(|row| row.2).collect()
}

fn split_payload(scoreboard: Option<&Map<String, Value>>, clips: &[Value]) -> Value {
    if let Some(run) = scoreboard.and_then(|board| board.get("run")).and_then(Value::as_object) {
        if let (Some(train), Some(heldout)) = (
            run.get("n_train_clips").and_then(as_int),
            run.get("n_heldout_clips").and_then(as_int),
        ) {
            return json!({"n_train": train, "n_heldout": heldout});
        }
    }
    let count = |split: &str| clips.iter().filter(|clip| clip.get("split").and_then(Value::as_str) == Some(split)).count();
    json!({"n_train": count("train"), "n_heldout": count("heldout")})
}

fn sae_payload(scoreboard: Option<&Map<String, Value>>) -> Value {
    let Some(sae) = scoreboard.and_then(|board| board.get("sae")).and_then(Value::as_object) else {
        return Value::Null;
    };
    if !["usable", "ev_heldout", "loss_recovered"].iter().any(|key| sae.contains_key(*key)) {
        return Value::Null;
    }
    json!({
        "usable": sae.get("usable") == Some(&Value::Bool(true)),
        "ev_heldout": sae.get("ev_heldout").and_then(num),
        "loss_recovered": sae.get("loss_recovered").and_then(num),
    })
}

// ---------------------------------------------------------------------------------------------
// Clip files

fn suffix_of(path: &Path) -> String {
    path.extension().and_then(|ext| ext.to_str()).map(str::to_ascii_lowercase).unwrap_or_default()
}

fn resolve_pixels(manifest: &Map<String, Value>, run_dir: &Path, clips_dir: Option<&Path>) -> HashMap<String, LoadedClip> {
    let directory = match clips_dir {
        Some(dir) => Some(dir.to_path_buf()),
        None => manifest_clips_dir(manifest, run_dir),
    };
    let files = directory.as_deref().map(candidate_files).unwrap_or_default();
    let by_stem = index_by_stem(&files);
    let mut sha_index: Option<HashMap<String, PathBuf>> = None;
    let mut resolved = HashMap::new();
    for clip in manifest["clips"].as_array().into_iter().flatten() {
        let Some(clip) = clip.as_object() else { continue };
        let Some(raw_id) = clip.get("clip_id") else { continue };
        let clip_id = py_str(raw_id);
        let mut path = if clips_dir.is_none() { manifest_clip_path(clip, run_dir) } else { None };
        let expected = clip.get("sha256").and_then(Value::as_str);
        let named = named_clip_file(directory.as_deref(), &files, clip);
        let stem_path = by_stem.get(&clip_id).cloned();
        if path.is_none() && sha_ok(named.as_deref(), expected) {
            path = named;
        } else if path.is_none() && sha_ok(stem_path.as_deref(), expected) {
            path = stem_path;
        } else if path.is_none() {
            if let Some(expected) = expected {
                let index = sha_index.get_or_insert_with(|| index_by_sha(&files));
                path = index.get(expected).cloned();
            }
            if path.is_none() {
                path = named.or(stem_path);
            }
        }
        resolved.insert(clip_id, load_clip_file(path.as_deref(), clip));
    }
    resolved
}

fn load_clip_file(path: Option<&Path>, clip: &Map<String, Value>) -> LoadedClip {
    let Some(path) = path.filter(|path| path.is_file()) else {
        return LoadedClip::missing();
    };
    if let Some(expected) = clip.get("sha256").and_then(Value::as_str) {
        let small = std::fs::metadata(path).map(|meta| meta.len() <= SHA_EAGER_LIMIT).unwrap_or(false);
        if small && file_sha256(path).ok().as_deref() != Some(expected) {
            return LoadedClip::status(
                "sha_mismatch",
                "Clip file does not match the sha256 recorded in the manifest.",
                Some(path.to_path_buf()),
            );
        }
    }
    let suffix = suffix_of(path);
    if suffix == "npy" {
        return load_npy(path, clip);
    }
    if NATIVE_VIDEO.iter().any(|(ext, _)| *ext == suffix) || TRANSCODE_VIDEO.contains(&suffix.as_str()) {
        return LoadedClip {
            kind: "video",
            ..LoadedClip::status("ok", "", Some(path.to_path_buf()))
        };
    }
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    LoadedClip::status("unreadable", format!("Unsupported clip file {name}."), Some(path.to_path_buf()))
}

fn load_npy(path: &Path, clip: &Map<String, Value>) -> LoadedClip {
    let header = match npy::read_header(path) {
        Ok(header) => header,
        Err(err) => {
            return LoadedClip::status("unreadable", format!("Clip file could not be read: {err}"), Some(path.to_path_buf()));
        }
    };
    let shape = &header.shape;
    if !(shape.len() == 3 || shape.len() == 4) || shape[0] == 0 {
        return LoadedClip::status(
            "unreadable",
            format!("Clip array must be T×H×W or T×H×W×C, got {}.", npy::shape_tuple(shape)),
            Some(path.to_path_buf()),
        );
    }
    let n_frames = clip.get("n_frames").and_then(as_int).unwrap_or(0);
    if n_frames != 0 && shape[0] as i64 != n_frames {
        return LoadedClip::status(
            "length_mismatch",
            format!("Clip file has {} frames; the manifest says {n_frames}.", shape[0]),
            Some(path.to_path_buf()),
        );
    }
    LoadedClip {
        status: "ok".into(),
        message: String::new(),
        kind: "array",
        path: Some(path.to_path_buf()),
        width: Some(shape[2] as i64),
        height: Some(shape[1] as i64),
    }
}

fn sha_ok(path: Option<&Path>, expected: Option<&str>) -> bool {
    let Some(path) = path else { return false };
    let Some(expected) = expected else { return true };
    if std::fs::metadata(path).map(|meta| meta.len() > SHA_EAGER_LIMIT).unwrap_or(false) {
        return true;
    }
    file_sha256(path).ok().as_deref() == Some(expected)
}

fn named_clip_file(directory: Option<&Path>, files: &[PathBuf], clip: &Map<String, Value>) -> Option<PathBuf> {
    let raw = clip.get("name").and_then(Value::as_str).filter(|raw| !raw.trim().is_empty())?;
    let filename = Path::new(raw).file_name()?.to_str()?.to_string();
    if filename.is_empty() || filename == "." || filename == ".." {
        return None;
    }
    if let Some(directory) = directory {
        let direct = directory.join(&filename);
        if direct.is_file() {
            return Some(direct);
        }
    }
    files.iter().find(|path| path.file_name().and_then(|name| name.to_str()) == Some(filename.as_str())).cloned()
}

fn candidate_files(directory: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(directory)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file() && MEDIA_SUFFIXES.contains(&suffix_of(path).as_str()))
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

fn index_by_stem(files: &[PathBuf]) -> HashMap<String, PathBuf> {
    let rank = |path: &Path| MEDIA_SUFFIXES.iter().position(|ext| *ext == suffix_of(path)).unwrap_or(usize::MAX);
    let mut chosen: HashMap<String, PathBuf> = HashMap::new();
    for path in files {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else { continue };
        match chosen.get(stem) {
            Some(current) if rank(path) >= rank(current) => {}
            _ => {
                chosen.insert(stem.to_string(), path.clone());
            }
        }
    }
    chosen
}

fn index_by_sha(files: &[PathBuf]) -> HashMap<String, PathBuf> {
    files.iter().filter_map(|path| file_sha256(path).ok().map(|sha| (sha, path.clone()))).collect()
}

fn manifest_clip_path(clip: &Map<String, Value>, run_dir: &Path) -> Option<PathBuf> {
    let raw = [clip.get("path"), clip.get("file")]
        .into_iter()
        .flatten()
        .find(|value| truthy(value))?
        .as_str()?;
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() { path } else { run_dir.join(path) };
    path.is_file().then_some(path)
}

fn manifest_clips_dir(manifest: &Map<String, Value>, run_dir: &Path) -> Option<PathBuf> {
    let raw = manifest.get("clips_dir").and_then(Value::as_str).filter(|raw| !raw.is_empty())?;
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() { path } else { run_dir.join(path) };
    path.is_dir().then_some(path)
}

// ---------------------------------------------------------------------------------------------
// Small rules shared with the page

fn kind_order(kind: &str) -> u8 {
    match kind {
        "ablation" => 0,
        "estimate" => 1,
        _ => 2,
    }
}

/// Python's `round(x, 1)`: correctly rounded, ties to even.
fn round1(value: f64) -> f64 {
    format!("{value:.1}").parse().unwrap_or(value)
}

pub fn badge_for(real: Option<&Value>, status: Option<&str>) -> &'static str {
    if real == Some(&Value::Bool(true)) {
        "real"
    } else if status.is_some() {
        "status"
    } else {
        "unverified"
    }
}

/// Badge copy for a non-real feature. The word `real` is never a status.
fn status_text(status: Option<&Value>) -> Option<String> {
    let text = status?.as_str()?.trim();
    if text.is_empty() || text.to_lowercase() == "real" {
        return None;
    }
    Some(text.to_string())
}

fn ablation_passed(status: &str) -> bool {
    status.to_lowercase().contains("ablation-passed")
}

fn token_layout(adapter: Option<&str>) -> &'static str {
    match adapter {
        Some(adapter) if adapter.starts_with("groot:") => "actions",
        _ => "patches",
    }
}

fn displayed_hw(clip: &Map<String, Value>, loaded: &LoadedClip) -> Option<(f64, f64)> {
    if let Some(Value::Array(raw)) = clip.get("hw") {
        if raw.len() == 2 {
            if let (Some(height), Some(width)) = (plain_num(&raw[0]), plain_num(&raw[1])) {
                if height > 0.0 && width > 0.0 {
                    return Some((height, width));
                }
            }
        }
    }
    match (loaded.height, loaded.width) {
        (Some(height), Some(width)) if height > 0 && width > 0 => Some((height as f64, width as f64)),
        _ => None,
    }
}

fn center_square(height: f64, width: f64) -> Value {
    let side = height.min(width);
    json!({"x": (width - side) / 2.0, "y": (height - side) / 2.0, "width": side, "height": side})
}

fn is_center(text: &str) -> bool {
    matches!(text.trim().to_lowercase().as_str(), "center" | "centre" | "center_square" | "centre_square")
}

fn overlay_rect(hw: Option<(f64, f64)>, extra: Option<&Map<String, Value>>, adapter: Option<&str>) -> Value {
    if let Some(explicit) = extra.and_then(|extra| extra.get("crop")) {
        if !explicit.is_null() {
            if let Some(parsed) = parse_crop(explicit, hw) {
                return parsed;
            }
        }
    }
    match (adapter, hw) {
        (Some("vjepa2"), Some((height, width))) => center_square(height, width),
        _ => Value::Null,
    }
}

fn parse_crop(raw: &Value, hw: Option<(f64, f64)>) -> Option<Value> {
    let rect = |x: Option<f64>, y: Option<f64>, width: Option<f64>, height: Option<f64>| -> Option<Value> {
        let (x, y, width, height) = (x?, y?, width?, height?);
        if width <= 0.0 || height <= 0.0 {
            return None;
        }
        Some(json!({"x": x, "y": y, "width": width, "height": height}))
    };
    match raw {
        Value::String(text) if is_center(text) => hw.map(|(h, w)| center_square(h, w)),
        Value::Object(object) => {
            let mode = match object.get("mode") {
                Some(Value::String(mode)) => Some(mode.as_str()),
                _ => object.get("type").and_then(Value::as_str),
            };
            if mode.map(is_center).unwrap_or(false) {
                return hw.map(|(h, w)| center_square(h, w));
            }
            let pick = |keys: &[&str]| keys.iter().find_map(|key| object.get(*key)).and_then(plain_num);
            rect(
                pick(&["x", "left"]),
                pick(&["y", "top"]),
                pick(&["width", "w", "size"]),
                pick(&["height", "h", "size"]),
            )
        }
        Value::Array(items) if items.len() == 4 => rect(
            plain_num(&items[0]),
            plain_num(&items[1]),
            plain_num(&items[2]),
            plain_num(&items[3]),
        ),
        _ => None,
    }
}

/// Python's `float(x)` when it is finite, else `None`.
fn num(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Number(number) => number.as_f64()?,
        Value::Bool(flag) => *flag as u8 as f64,
        Value::String(text) => text.trim().replace('_', "").parse::<f64>().ok()?,
        _ => return None,
    };
    number.is_finite().then_some(number)
}

fn plain_num(value: &Value) -> Option<f64> {
    if value.is_boolean() {
        return None;
    }
    num(value)
}

fn json_value(value: &Value) -> Value {
    match value {
        Value::Number(number) if !number.is_i64() && !number.is_u64() => {
            number.as_f64().filter(|f| f.is_finite()).map(Value::from).unwrap_or(Value::Null)
        }
        other => other.clone(),
    }
}

fn loose_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64().or_else(|| number.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(flag) => Some(*flag as i64),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn fps_of(clip: &Map<String, Value>) -> f64 {
    clip.get("fps").filter(|value| truthy(value)).and_then(num).unwrap_or(0.0)
}

/// Python truthiness for JSON values.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(object) => !object.is_empty(),
    }
}

/// Python's `str(x)` for a clip id.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}
