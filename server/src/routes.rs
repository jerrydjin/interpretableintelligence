use std::io::{Read, Seek, SeekFrom};

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use http_body_util::BodyExt;
use include_dir::{include_dir, Dir};
use serde::Deserialize;
use serde_json::Value;
use tower_http::trace::TraceLayer;

use crate::citation::{self, gates_from_scoreboard};
use crate::error::ApiError;
use crate::model::{self, RunContext};
use crate::AppState;

static STATIC_FILES: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/../src/physint/static");

const EXPLAIN_UNAVAILABLE: &str = "POST /explain runs in the Python service because it has to load the model. Start `physint serve` and set PHYSINT_EXPLAIN_URL (or pass --explain-url) to that service's base URL.";

#[derive(Debug, Deserialize)]
struct RunQuery {
    run_id: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/runs", get(list_runs))
        .route("/runs/{run_id}", get(run_detail))
        .route("/features/{feature_id}", get(feature))
        .route("/features/{feature_id}/top_clips", get(top_clips))
        .route("/clips/{clip_id}/frame/{frame}", get(frame))
        .route("/clips/{clip_id}/file", get(clip_file))
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/static/{*path}", get(static_file))
        .route("/explain", post(explain))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn list_runs(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    Ok(Json(model::list_runs(&state.runs_dir)?))
}

async fn run_detail(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let ctx = open_run(&state, &run_id)?;
    let conn = lock_db(&state);
    Ok(Json(model::run_document(
        &conn,
        &ctx,
        state.clips_dir.as_deref(),
    )?))
}

async fn feature(
    State(state): State<AppState>,
    Path(feature_id): Path<i64>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let ctx = open_run(&state, &query.run_id)?;
    Ok(Json(model::feature_document(&ctx, feature_id)?))
}

async fn top_clips(
    State(state): State<AppState>,
    Path(feature_id): Path<i64>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let ctx = open_run(&state, &query.run_id)?;
    let conn = lock_db(&state);
    Ok(Json(model::top_clips_document(&conn, &ctx, feature_id)?))
}

async fn frame(
    State(state): State<AppState>,
    Path((clip_id, frame)): Path<(String, i64)>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let ctx = open_run(&state, &query.run_id)?;
    let conn = lock_db(&state);
    Ok(Json(model::frame_document(&conn, &ctx, &clip_id, frame)?))
}

async fn clip_file(
    State(state): State<AppState>,
    Path(clip_id): Path<String>,
    Query(query): Query<RunQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let ctx = open_run(&state, &query.run_id)?;
    let path = model::clip_path(&ctx, state.clips_dir.as_deref(), &clip_id)?;
    let mut file = std::fs::File::open(&path).map_err(crate::error::io_err)?;
    let len = file.seek(SeekFrom::End(0)).map_err(crate::error::io_err)?;
    let range = headers
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok());
    if let Some(range) = range {
        let (start, end) = parse_byte_range(range, len).ok_or_else(|| {
            ApiError::new(
                StatusCode::RANGE_NOT_SATISFIABLE,
                format!("range not satisfiable for a {len}-byte clip"),
            )
        })?;
        let n = (end - start + 1) as usize;
        file.seek(SeekFrom::Start(start))
            .map_err(crate::error::io_err)?;
        let mut buf = vec![0u8; n];
        file.read_exact(&mut buf).map_err(crate::error::io_err)?;
        return Ok(Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}"))
            .header(header::CONTENT_LENGTH, buf.len())
            .body(Body::from(buf))
            .unwrap());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(crate::error::io_err)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(crate::error::io_err)?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, buf.len())
        .body(Body::from(buf))
        .unwrap())
}

async fn index() -> Response {
    serve_static("index.html")
}

async fn static_file(Path(path): Path<String>) -> Response {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| part == "..")
    {
        return ApiError::not_found("not found", None, None, None).into_response();
    }
    serve_static(&path)
}

fn serve_static(path: &str) -> Response {
    let Some(file) = STATIC_FILES.get_file(path) else {
        return ApiError::not_found("not found", None, None, None).into_response();
    };
    let cache = if path.ends_with(".woff2") {
        "public, max-age=86400"
    } else {
        "no-cache"
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type(path))
        .header(header::CACHE_CONTROL, cache)
        .header(header::CONTENT_LENGTH, file.contents().len())
        .body(Body::from(Bytes::copy_from_slice(file.contents())))
        .unwrap()
}

async fn explain(State(state): State<AppState>, request: Request) -> Response {
    let Some(base) = state.explain_url.clone() else {
        return ApiError::unavailable(EXPLAIN_UNAVAILABLE).into_response();
    };
    let url = format!("{}/explain", base.trim_end_matches('/'));
    let (parts, body) = request.into_parts();
    let content_type = parts.headers.get(header::CONTENT_TYPE).cloned();
    let bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(err) => {
            return ApiError::bad(format!("could not read the explain request body: {err}"))
                .into_response();
        }
    };
    let mut upstream = state.http.post(url).body(bytes);
    if let Some(content_type) = content_type {
        upstream = upstream.header(header::CONTENT_TYPE, content_type);
    }
    let response = match upstream.send().await {
        Ok(response) => response,
        Err(err) => {
            return ApiError::bad_gateway(format!(
                "Python explain service is unreachable ({err}). POST /explain stays there because it has to run the model."
            ))
            .into_response();
        }
    };
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let upstream_type = response.headers().get(header::CONTENT_TYPE).cloned();
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(err) => {
            return ApiError::bad_gateway(format!(
                "Python explain service returned an unreadable body: {err}"
            ))
            .into_response();
        }
    };
    if status != StatusCode::OK {
        return passthrough(status, upstream_type, body);
    }
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(_) => return passthrough(status, upstream_type, body),
    };
    if payload.get("answer").is_none() {
        return passthrough(status, upstream_type, body);
    }
    let Some(run_id) = payload
        .get("run_id")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return ApiError::bad_gateway(
            "refusing to present an explanation that has no run_id to check against the scoreboard",
        )
        .into_response();
    };
    let ctx = match open_run(&state, &run_id) {
        Ok(ctx) => ctx,
        Err(err) => {
            return ApiError::bad_gateway(format!(
                "refusing to present an explanation whose run could not be checked ({})",
                err.error
            ))
            .into_response();
        }
    };
    let gates = gates_from_scoreboard(&ctx.scoreboard);
    if let Err(err) = citation::sanitize(&mut payload, &gates) {
        return ApiError::bad_gateway(format!(
            "refusing to present an explanation that breaks the citation rule: {err}"
        ))
        .into_response();
    }
    Json(payload).into_response()
}

fn passthrough(
    status: StatusCode,
    content_type: Option<axum::http::HeaderValue>,
    body: Bytes,
) -> Response {
    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header(header::CONTENT_TYPE, content_type);
    } else {
        response = response.header(header::CONTENT_TYPE, "application/json");
    }
    response.body(Body::from(body)).unwrap()
}

fn open_run(state: &AppState, run_id: &str) -> Result<RunContext, ApiError> {
    let dir = model::resolve_run_id(&state.runs_dir, run_id)?;
    model::load_run(&dir)
}

fn lock_db(state: &AppState) -> std::sync::MutexGuard<'_, duckdb::Connection> {
    state.db.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("woff2") => "font/woff2",
        Some("txt") => "text/plain; charset=utf-8",
        Some("png") => "image/png",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

/// A single `bytes=` range. `end` is inclusive.
fn parse_byte_range(header: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let spec = header.trim().strip_prefix("bytes=")?.trim();
    let spec = spec.split(',').next()?.trim();
    if spec.is_empty() {
        return None;
    }
    if let Some(suffix) = spec.strip_prefix('-') {
        let n: u64 = suffix.parse().ok()?;
        if n == 0 {
            return None;
        }
        if n >= len {
            return Some((0, len - 1));
        }
        return Some((len - n, len - 1));
    }
    let (start_s, end_s) = spec.split_once('-')?;
    let start: u64 = start_s.parse().ok()?;
    if start >= len {
        return None;
    }
    let end = if end_s.is_empty() {
        len - 1
    } else {
        let end: u64 = end_s.parse().ok()?;
        end.min(len - 1)
    };
    if start > end {
        return None;
    }
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_state;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use serde_json::json;
    use tower::ServiceExt;

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes))
    }

    fn write_npy(path: &std::path::Path, n_frames: u64) {
        let dict =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({n_frames}, 2, 2, 1), }}");
        let mut header = dict.into_bytes();
        header.push(b'\n');
        let prefix = 10;
        let pad = (16 - ((prefix + header.len()) % 16)) % 16;
        for _ in 0..pad {
            header.insert(header.len() - 1, b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&vec![0u8; (n_frames * 2 * 2 * 4) as usize]);
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture() -> (tempfile::TempDir, tempfile::TempDir) {
        let runs = tempfile::tempdir().unwrap();
        let clips = tempfile::tempdir().unwrap();
        let clip = clips.path().join("clip-000.npy");
        write_npy(&clip, 2);
        let digest = sha256_hex(&std::fs::read(&clip).unwrap());
        let run = runs.path().join("demo");
        std::fs::create_dir_all(run.join("features")).unwrap();
        let manifest = json!({
            "run_id": "demo",
            "model": {"adapter": "fake", "name": "fake", "sha": "ab".repeat(32)},
            "layer": 16,
            "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 1, "patch_px": 1},
            "clips": [{
                "clip_id": "clip-000",
                "sha256": digest,
                "fps": 8.0,
                "n_frames": 2,
                "split": "train"
            }]
        });
        let scoreboard = json!({
            "run": {
                "model_sha": "ab".repeat(32),
                "layer": 16,
                "n_train_clips": 1,
                "n_heldout_clips": 1
            },
            "sae": {"usable": true, "ev_heldout": 0.9, "loss_recovered": 0.8},
            "features": [{
                "id": 0,
                "label": "stripe",
                "distinct_clips_top20": 1,
                "label_holds_heldout": true,
                "ablation_effect": 0.2,
                "random_percentile": 100.0,
                "n_random": 100,
                "absent_in_random_init": true,
                "real": true
            }]
        });
        std::fs::write(
            run.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run.join("scoreboard.json"),
            serde_json::to_vec_pretty(&scoreboard).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run.join("features").join("features.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&scoreboard["features"][0])
                    .unwrap()
                    .replace('}', ",\"label_model\":\"fake-cue-v1\",\"top\":[]}")
            ),
        )
        .unwrap();
        let db = duckdb::Connection::open_in_memory().unwrap();
        let acts = run.join("features").join("feature_acts.parquet");
        let effects = run.join("features").join("clip_effects.parquet");
        db.execute_batch(&format!(
            "COPY (
                SELECT 0::INTEGER AS feature_id, 'clip-000' AS clip_id,
                       0::SMALLINT AS \"window\", 0::SMALLINT AS t,
                       0::SMALLINT AS h, 1::SMALLINT AS w, 1.25::FLOAT AS value
             ) TO '{}' (FORMAT PARQUET);
             COPY (
                SELECT 0::INTEGER AS feature_id, 'clip-000' AS clip_id,
                       0::SMALLINT AS \"window\", 0::SMALLINT AS t,
                       0.2::FLOAT AS effect, 'ablation' AS effect_kind
             ) TO '{}' (FORMAT PARQUET);",
            acts.display(),
            effects.display()
        ))
        .unwrap();
        (runs, clips)
    }

    async fn body_bytes(response: Response) -> Bytes {
        response.into_body().collect().await.unwrap().to_bytes()
    }

    #[tokio::test]
    async fn explain_without_python_is_501() {
        let (runs, _clips) = fixture();
        let state = open_state(runs.path().to_path_buf(), None, None).unwrap();
        let response = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/explain")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let payload: Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert!(payload["detail"]["error"]
            .as_str()
            .unwrap()
            .contains("PHYSINT_EXPLAIN_URL"));
    }

    #[tokio::test]
    async fn clip_file_honours_a_byte_range() {
        let (runs, clips) = fixture();
        let state = open_state(
            runs.path().to_path_buf(),
            Some(clips.path().to_path_buf()),
            None,
        )
        .unwrap();
        let app = router(state);
        let full = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/clips/clip-000/file?run_id=demo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(full.status(), StatusCode::OK);
        assert_eq!(full.headers().get(header::ACCEPT_RANGES).unwrap(), "bytes");
        let full_bytes = body_bytes(full).await;
        let partial = app
            .oneshot(
                Request::builder()
                    .uri("/clips/clip-000/file?run_id=demo")
                    .header(header::RANGE, "bytes=0-3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(partial.status(), StatusCode::PARTIAL_CONTENT);
        let range = partial.headers().get(header::CONTENT_RANGE).unwrap();
        assert_eq!(
            range.to_str().unwrap(),
            format!("bytes 0-3/{}", full_bytes.len())
        );
        assert_eq!(body_bytes(partial).await.as_ref(), &full_bytes[..4]);
    }

    #[tokio::test]
    async fn proxy_strips_a_non_real_citation() {
        let (runs, _clips) = fixture();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let app = Router::new().route(
                "/explain",
                post(|| async {
                    Json(json!({
                        "run_id": "demo",
                        "model_sha": "ab".repeat(32),
                        "layer": 16,
                        "answer": [{
                            "window": 0,
                            "t": 0,
                            "features": [{
                                "feature_id": 0,
                                "label": "not-the-scoreboard-label",
                                "effect": 0.2,
                                "effect_kind": "estimate",
                                "real": true
                            }]
                        }],
                        "unverified": []
                    }))
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });
        let state = open_state(
            runs.path().to_path_buf(),
            None,
            Some(format!("http://{addr}")),
        )
        .unwrap();
        let response = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/explain")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(payload["answer"].as_array().unwrap().len(), 0);
        let row = &payload["unverified"][0];
        assert_eq!(row["effect_kind"], "estimate");
        assert_eq!(row["reason"], "estimate only");
        assert_eq!(row["label"], "stripe");
        assert_eq!(row["real"], true);
    }

    #[test]
    fn static_shell_is_embedded() {
        assert!(STATIC_FILES.get_file("index.html").is_some());
        assert!(STATIC_FILES.get_file("app.js").is_some());
        assert!(STATIC_FILES.get_file("fonts/geist-400.woff2").is_some());
    }
}
