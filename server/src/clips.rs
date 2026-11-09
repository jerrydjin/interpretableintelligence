//! Clip file resolution. The rules are the viewer's: `--clips` wins, then a
//! manifest path, then `clips.json` / `*.npy`, then a sha256 match.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::ApiError;
use crate::npy::{self, shape_tuple};

#[derive(Debug)]
pub struct LoadedClip {
    pub status: String,
    pub message: String,
    pub path: Option<PathBuf>,
}

pub fn resolve_all(
    manifest: &Value,
    run_dir: &Path,
    clips_dir: Option<&Path>,
) -> Result<HashMap<String, LoadedClip>, ApiError> {
    let clips = manifest
        .get("clips")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad("manifest.clips must be a list"))?;
    let from_dir = if let Some(dir) = clips_dir {
        index_clip_dir(dir)?
    } else if let Some(dir) = manifest_clips_dir(manifest, run_dir) {
        index_clip_dir(&dir)?
    } else {
        HashMap::new()
    };
    let mut sha_index: Option<HashMap<String, PathBuf>> = None;
    let mut resolved = HashMap::new();
    for clip in clips {
        let clip_id = clip
            .get("clip_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut path = if clips_dir.is_none() {
            manifest_clip_path(clip, run_dir)
        } else {
            None
        };
        if path.is_none() {
            path = from_dir.get(&clip_id).cloned();
        }
        if path.is_none() && clips_dir.is_some() {
            if sha_index.is_none() {
                sha_index = Some(index_by_sha(clips_dir.unwrap())?);
            }
            if let Some(expected) = clip.get("sha256").and_then(Value::as_str) {
                path = sha_index.as_ref().unwrap().get(expected).cloned();
            }
        }
        resolved.insert(clip_id, load_clip_file(path.as_deref(), clip)?);
    }
    Ok(resolved)
}

pub fn file_sha256(path: &Path) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn load_clip_file(path: Option<&Path>, clip: &Value) -> Result<LoadedClip, ApiError> {
    let Some(path) = path else {
        return Ok(missing());
    };
    if !path.is_file() {
        return Ok(missing());
    }
    if let Some(expected) = clip.get("sha256").and_then(Value::as_str) {
        let digest = file_sha256(path).map_err(crate::error::io_err)?;
        if digest != expected {
            return Ok(LoadedClip {
                status: "sha_mismatch".into(),
                message: "Clip file does not match the sha256 recorded in the manifest.".into(),
                path: None,
            });
        }
    }
    let header = match npy::read_header(path) {
        Ok(header) => header,
        Err(err) => {
            return Ok(LoadedClip {
                status: "unreadable".into(),
                message: format!("Clip file could not be read: {err}"),
                path: None,
            });
        }
    };
    let rank = header.shape.len();
    if (rank != 3 && rank != 4) || header.shape.first().copied().unwrap_or(0) == 0 {
        return Ok(LoadedClip {
            status: "unreadable".into(),
            message: format!(
                "Clip array must be T×H×W or T×H×W×C, got {}.",
                shape_tuple(&header.shape)
            ),
            path: None,
        });
    }
    let n_frames = match clip.get("n_frames") {
        Some(Value::Number(number)) => number.as_i64().unwrap_or(0),
        _ => 0,
    };
    if n_frames != 0 && header.shape[0] as i64 != n_frames {
        return Ok(LoadedClip {
            status: "length_mismatch".into(),
            message: format!(
                "Clip file has {} frames; the manifest says {n_frames}.",
                header.shape[0]
            ),
            path: None,
        });
    }
    Ok(LoadedClip {
        status: "ok".into(),
        message: String::new(),
        path: Some(path.to_path_buf()),
    })
}

fn missing() -> LoadedClip {
    LoadedClip {
        status: "missing".into(),
        message: "Clip file not found. Pass --clips, or record a path in the manifest.".into(),
        path: None,
    }
}

fn manifest_clip_path(clip: &Value, run_dir: &Path) -> Option<PathBuf> {
    let raw = clip
        .get("path")
        .and_then(Value::as_str)
        .filter(|raw| !raw.is_empty())
        .or_else(|| {
            clip.get("file")
                .and_then(Value::as_str)
                .filter(|raw| !raw.is_empty())
        })?;
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() {
        path
    } else {
        run_dir.join(path)
    };
    path.is_file().then_some(path)
}

fn manifest_clips_dir(manifest: &Value, run_dir: &Path) -> Option<PathBuf> {
    let raw = manifest.get("clips_dir").and_then(Value::as_str)?;
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() {
        path
    } else {
        run_dir.join(path)
    };
    path.is_dir().then_some(path)
}

fn index_clip_dir(directory: &Path) -> Result<HashMap<String, PathBuf>, ApiError> {
    let mut mapping = HashMap::new();
    let index = directory.join("clips.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index).map_err(crate::error::io_err)?;
        let meta: Value = serde_json::from_str(&text)
            .map_err(|err| ApiError::bad(format!("could not read {}: {err}", index.display())))?;
        let entries = meta.get("clips").and_then(Value::as_array);
        if let Some(entries) = entries {
            for entry in entries {
                let Some(clip_id) = entry.get("clip_id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(file_name) = entry.get("file").and_then(Value::as_str) else {
                    continue;
                };
                if clip_id.is_empty() || file_name.is_empty() {
                    continue;
                }
                let candidate = directory.join(file_name);
                if candidate.is_file() {
                    mapping.insert(clip_id.to_string(), candidate);
                }
            }
        }
        if !mapping.is_empty() {
            return Ok(mapping);
        }
    }
    let mut files = npy_files(directory)?;
    files.sort();
    for file in files {
        if let Some(stem) = file.file_stem().and_then(|stem| stem.to_str()) {
            mapping.insert(stem.to_string(), file);
        }
    }
    Ok(mapping)
}

fn index_by_sha(directory: &Path) -> Result<HashMap<String, PathBuf>, ApiError> {
    let mut files = npy_files(directory)?;
    files.sort();
    let mut found = HashMap::new();
    for file in files {
        let digest = file_sha256(&file).map_err(crate::error::io_err)?;
        found.insert(digest, file);
    }
    Ok(found)
}

fn npy_files(directory: &Path) -> Result<Vec<PathBuf>, ApiError> {
    let entries = std::fs::read_dir(directory).map_err(crate::error::io_err)?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(crate::error::io_err)?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("npy") && path.is_file() {
            files.push(path);
        }
    }
    Ok(files)
}
