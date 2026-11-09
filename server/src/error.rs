use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::Value;

/// Error body shared with the Python API: `{"detail": {error, run_id, model_sha, layer}}`.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub error: String,
    pub run_id: Option<String>,
    pub model_sha: Option<String>,
    pub layer: Option<i64>,
}

impl ApiError {
    pub fn new(status: StatusCode, error: impl Into<String>) -> Self {
        Self {
            status,
            error: error.into(),
            run_id: None,
            model_sha: None,
            layer: None,
        }
    }

    pub fn bad(error: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, error)
    }

    pub fn not_found(
        error: impl Into<String>,
        run_id: Option<String>,
        model_sha: Option<String>,
        layer: Option<i64>,
    ) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error: error.into(),
            run_id,
            model_sha,
            layer,
        }
    }

    pub fn unavailable(error: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, error)
    }

    pub fn bad_gateway(error: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, error)
    }

    pub fn traced(mut self, run_id: &str, model_sha: &str, layer: i64) -> Self {
        self.run_id = Some(run_id.to_string());
        self.model_sha = Some(model_sha.to_string());
        self.layer = Some(layer);
        self
    }
}

#[derive(Serialize)]
struct DetailBody {
    detail: Detail,
}

#[derive(Serialize)]
struct Detail {
    error: String,
    run_id: Option<String>,
    model_sha: Option<String>,
    layer: Option<i64>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status;
        let body = DetailBody {
            detail: Detail {
                error: self.error,
                run_id: self.run_id,
                model_sha: self.model_sha,
                layer: self.layer,
            },
        };
        (status, Json(body)).into_response()
    }
}

pub fn db_err(err: duckdb::Error) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("duckdb: {err}"))
}

pub fn io_err(err: std::io::Error) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("io: {err}"))
}

/// A JSON number, or null when the value is not finite.
pub fn finite_f64(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}
