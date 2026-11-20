//! Routes of the local viewer. Errors are `{"error": "..."}`.
//!
//! `POST /api/steer` needs the model, so it is forwarded to the Python steering service
//! (`--steer-url`). A run that is still filling in (`ii run --remote`) is reloaded whenever the
//! JSON file passed as `--live` changes its `generation`; `/api/run` then carries that file's
//! contents under `live`.

use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, RwLock};

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};

use super::RunView;
use crate::routes::{content_type, embedded_static};

const STEER_LIMIT: usize = 64 * 1024;

#[derive(Clone)]
pub struct ViewState {
    inner: Arc<Inner>,
}

struct Inner {
    run_dir: PathBuf,
    clips_dir: Option<PathBuf>,
    static_dir: Option<PathBuf>,
    steer_url: Option<String>,
    steer_status: Option<Value>,
    live_file: Option<PathBuf>,
    current: RwLock<(Arc<RunView>, i64)>,
    http: reqwest::Client,
}

pub struct ViewOptions {
    pub run_dir: PathBuf,
    pub clips_dir: Option<PathBuf>,
    pub static_dir: Option<PathBuf>,
    pub steer_url: Option<String>,
    pub live_file: Option<PathBuf>,
}

impl ViewState {
    pub async fn open(options: ViewOptions) -> Result<ViewState, String> {
        let view = RunView::load(&options.run_dir, options.clips_dir.as_deref())?;
        let http = reqwest::Client::builder().build().map_err(|err| err.to_string())?;
        let steer_url = options.steer_url.map(|url| url.trim_end_matches('/').to_string());
        let steer_status = match &steer_url {
            Some(url) => Some(steer_status(&http, url).await?),
            None => None,
        };
        let generation = options.live_file.as_deref().map(live_generation).unwrap_or(0);
        Ok(ViewState {
            inner: Arc::new(Inner {
                run_dir: options.run_dir,
                clips_dir: options.clips_dir,
                static_dir: options.static_dir,
                steer_url,
                steer_status,
                live_file: options.live_file,
                current: RwLock::new((Arc::new(view), generation)),
                http,
            }),
        })
    }

    /// The loaded run, reloaded first when the live file moved on to a new generation.
    fn view(&self) -> Result<Arc<RunView>, String> {
        if let Some(file) = &self.inner.live_file {
            let generation = live_generation(file);
            let stale = self.inner.current.read().unwrap().1 != generation;
            if stale {
                let fresh = Arc::new(RunView::load(&self.inner.run_dir, self.inner.clips_dir.as_deref())?);
                *self.inner.current.write().unwrap() = (fresh, generation);
            }
        }
        Ok(self.inner.current.read().unwrap().0.clone())
    }
}

async fn steer_status(http: &reqwest::Client, url: &str) -> Result<Value, String> {
    let response = http
        .get(format!("{url}/api/steer"))
        .send()
        .await
        .map_err(|err| format!("steering service at {url} is unreachable: {err}"))?;
    let bytes = response.bytes().await.map_err(|err| format!("steering service at {url} answered unreadably: {err}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|err| format!("steering service at {url} answered with something that is not JSON: {err}"))
}

fn read_live(file: &FsPath) -> Option<Value> {
    let text = std::fs::read_to_string(file).ok()?;
    serde_json::from_str::<Value>(&text).ok().filter(Value::is_object)
}

fn live_generation(file: &FsPath) -> i64 {
    read_live(file).and_then(|live| live.get("generation").and_then(Value::as_i64)).unwrap_or(0)
}

pub fn router(state: ViewState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/static/{*path}", get(static_file))
        .route("/api/run", get(run))
        .route("/api/features/{feature}", get(feature))
        .route("/api/features/{feature}/viz", get(viz))
        .route("/api/clips/{clip}", get(clip))
        .route("/api/clips/{clip}/heatmap", get(heatmap))
        .route("/api/clips/{clip}/media", get(media))
        .route("/api/clips/{clip}/frames/{file}", get(frame_png))
        .route("/api/steer", axum::routing::post(steer))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "not found") })
        .with_state(state)
}

fn json_response(status: StatusCode, payload: &Value) -> Response {
    let body = serde_json::to_vec(payload).expect("JSON values serialize");
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONTENT_LENGTH, body.len())
        .body(Body::from(body))
        .unwrap()
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    json_response(status, &json!({"error": message.into()}))
}

fn internal(message: String) -> Response {
    error(StatusCode::INTERNAL_SERVER_ERROR, message)
}

fn png(bytes: Bytes) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/png")
        .header(header::CACHE_CONTROL, "private, max-age=3600")
        .header(header::CONTENT_LENGTH, bytes.len())
        .body(Body::from(bytes))
        .unwrap()
}

async fn index(State(state): State<ViewState>) -> Response {
    serve_asset(&state, "index.html")
}

async fn static_file(State(state): State<ViewState>, Path(path): Path<String>) -> Response {
    let parts_ok = !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && !path.split('/').any(|part| part == "..");
    if !parts_ok {
        return error(StatusCode::NOT_FOUND, "not found");
    }
    serve_asset(&state, &path)
}

/// From `--static` when given (so an edited page needs no rebuild), else the embedded copy.
fn serve_asset(state: &ViewState, relative: &str) -> Response {
    let bytes = match &state.inner.static_dir {
        Some(root) => {
            let (Ok(root), Ok(candidate)) = (root.canonicalize(), root.join(relative).canonicalize()) else {
                return error(StatusCode::NOT_FOUND, "not found");
            };
            if !candidate.starts_with(&root) || !candidate.is_file() {
                return error(StatusCode::NOT_FOUND, "not found");
            }
            match std::fs::read(&candidate) {
                Ok(bytes) => Bytes::from(bytes),
                Err(_) => return error(StatusCode::NOT_FOUND, "not found"),
            }
        }
        None => match embedded_static(relative) {
            Some(bytes) => Bytes::from_static(bytes),
            None => return error(StatusCode::NOT_FOUND, "not found"),
        },
    };
    let cache = if relative.ends_with(".woff2") { "public, max-age=86400" } else { "no-cache" };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type(relative))
        .header(header::CACHE_CONTROL, cache)
        .header(header::CONTENT_LENGTH, bytes.len())
        .body(Body::from(bytes))
        .unwrap()
}

async fn run(State(state): State<ViewState>) -> Response {
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    let mut payload = view.run_payload.clone();
    if let Some(status) = &state.inner.steer_status {
        payload["steer"] = status.clone();
    }
    if let Some(file) = &state.inner.live_file {
        payload["live"] = read_live(file).unwrap_or_else(|| json!({"generation": 0}));
    }
    json_response(StatusCode::OK, &payload)
}

fn parse_id(raw: &str) -> Option<i64> {
    (!raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit())).then(|| raw.parse().ok()).flatten()
}

async fn feature(State(state): State<ViewState>, Path(raw): Path<String>) -> Response {
    let Some(feature_id) = parse_id(&raw) else {
        return error(StatusCode::NOT_FOUND, "not found");
    };
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    match tokio::task::spawn_blocking(move || view.feature(feature_id)).await {
        Ok(Ok(Some(detail))) => json_response(StatusCode::OK, &detail),
        Ok(Ok(None)) => error(StatusCode::NOT_FOUND, "unknown feature"),
        Ok(Err(err)) => internal(err),
        Err(err) => internal(err.to_string()),
    }
}

async fn viz(State(state): State<ViewState>, Path(raw): Path<String>) -> Response {
    let Some(feature_id) = parse_id(&raw) else {
        return error(StatusCode::NOT_FOUND, "not found");
    };
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    match view.viz_file(feature_id).and_then(|path| std::fs::read(path).ok()) {
        Some(bytes) => png(Bytes::from(bytes)),
        None => error(StatusCode::NOT_FOUND, "no viz"),
    }
}

async fn clip(State(state): State<ViewState>, Path(clip_id): Path<String>) -> Response {
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    match tokio::task::spawn_blocking(move || view.clip(&clip_id)).await {
        Ok(Ok(Some(payload))) => json_response(StatusCode::OK, &payload),
        Ok(Ok(None)) => error(StatusCode::NOT_FOUND, "unknown clip"),
        Ok(Err(err)) => internal(err),
        Err(err) => internal(err.to_string()),
    }
}

/// `?frame=` and `?feature=` as Python's `str.isdigit()` reads them: the first value, digits only.
fn query_int(query: &[(String, String)], name: &str) -> Option<i64> {
    let raw = query.iter().find(|(key, value)| key == name && !value.is_empty())?.1.as_str();
    parse_id(raw)
}

async fn heatmap(
    State(state): State<ViewState>,
    Path(clip_id): Path<String>,
    Query(query): Query<Vec<(String, String)>>,
) -> Response {
    let (Some(frame), Some(feature_id)) = (query_int(&query, "frame"), query_int(&query, "feature")) else {
        return error(StatusCode::BAD_REQUEST, "frame and feature are required");
    };
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    match tokio::task::spawn_blocking(move || view.heatmap(&clip_id, frame, feature_id)).await {
        Ok(Ok(Some(grid))) => json_response(StatusCode::OK, &json!({"heatmap": grid})),
        Ok(Ok(None)) => error(StatusCode::NOT_FOUND, "frame not available"),
        Ok(Err(err)) => internal(err),
        Err(err) => internal(err.to_string()),
    }
}

async fn frame_png(State(state): State<ViewState>, Path((clip_id, file)): Path<(String, String)>) -> Response {
    let Some(frame) = file.strip_suffix(".png").and_then(parse_id) else {
        return error(StatusCode::NOT_FOUND, "not found");
    };
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    match tokio::task::spawn_blocking(move || view.frame_png(&clip_id, frame)).await {
        Ok(Ok(Some(bytes))) => png(Bytes::from(bytes.as_ref().clone())),
        Ok(Ok(None)) => error(StatusCode::NOT_FOUND, "frame not available"),
        Ok(Err(err)) => internal(err),
        Err(err) => internal(err.to_string()),
    }
}

async fn media(State(state): State<ViewState>, Path(clip_id): Path<String>, headers: HeaderMap) -> Response {
    let view = match state.view() {
        Ok(view) => view,
        Err(err) => return internal(err),
    };
    let Some((path, content_type)) = view.media_file(&clip_id) else {
        return error(StatusCode::NOT_FOUND, "video not available");
    };
    send_range(&path, content_type, headers.get(header::RANGE).and_then(|value| value.to_str().ok()))
}

/// A single `bytes=` range, read the way the Python viewer reads it.
fn send_range(path: &FsPath, content_type: &str, range: Option<&str>) -> Response {
    let Ok(bytes) = std::fs::read(path) else {
        return error(StatusCode::NOT_FOUND, "video not available");
    };
    let size = bytes.len() as i64;
    let (mut start, mut end, mut status) = (0i64, (size - 1).max(0), StatusCode::OK);
    if let Some(spec) = range.and_then(|range| range.strip_prefix("bytes=")).filter(|_| size > 0) {
        let spec = spec.split(',').next().unwrap_or("").trim();
        let (start_text, end_text) = spec.split_once('-').unwrap_or((spec, ""));
        let parsed = if start_text.is_empty() {
            let suffix: Result<i64, _> = if end_text.is_empty() { Ok(0) } else { end_text.parse() };
            suffix.map(|suffix| ((size - suffix).max(0), size - 1))
        } else {
            start_text.parse::<i64>().and_then(|first| {
                let last = if end_text.is_empty() { Ok(size - 1) } else { end_text.parse::<i64>() };
                last.map(|last| (first, last))
            })
        };
        match parsed {
            Err(_) => {
                start = 0;
                end = size - 1;
            }
            Ok((first, last)) => {
                if first < 0 || first >= size {
                    return Response::builder()
                        .status(StatusCode::RANGE_NOT_SATISFIABLE)
                        .header(header::CONTENT_RANGE, format!("bytes */{size}"))
                        .header(header::CONTENT_LENGTH, 0)
                        .body(Body::empty())
                        .unwrap();
                }
                start = first;
                end = last.min(size - 1);
                status = StatusCode::PARTIAL_CONTENT;
            }
        }
    }
    let length = if size == 0 { 0 } else { (end - start + 1).max(0) };
    let body = if length == 0 {
        Bytes::new()
    } else {
        Bytes::copy_from_slice(&bytes[start as usize..(start + length) as usize])
    };
    let mut response = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, length)
        .header(header::CACHE_CONTROL, "private, max-age=3600");
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{size}"));
    }
    response.body(Body::from(body)).unwrap()
}

async fn steer(State(state): State<ViewState>, request: Request) -> Response {
    let Some(base) = state.inner.steer_url.clone() else {
        return error(StatusCode::NOT_FOUND, "steering is off; start the viewer with --steer");
    };
    let body = match request.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return error(StatusCode::BAD_REQUEST, "request body must be a JSON object under 64 KiB"),
    };
    if body.is_empty() || body.len() > STEER_LIMIT {
        return error(StatusCode::BAD_REQUEST, "request body must be a JSON object under 64 KiB");
    }
    let upstream = state
        .inner
        .http
        .post(format!("{base}/api/steer"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await;
    let response = match upstream {
        Ok(response) => response,
        Err(err) => return error(StatusCode::BAD_GATEWAY, format!("steering service is unreachable: {err}")),
    };
    let status = StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    match response.bytes().await {
        Ok(bytes) => Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(bytes))
            .unwrap(),
        Err(err) => error(StatusCode::BAD_GATEWAY, format!("steering service answered unreadably: {err}")),
    }
}
