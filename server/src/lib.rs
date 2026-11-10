//! Read API for interpretableintelligence run folders.
//!
//! Parquet is queried through DuckDB. JSON files (manifest, scoreboard,
//! features.jsonl) are read directly. POST /explain is proxied to Python.

pub mod bench;
pub mod citation;
pub mod clips;
pub mod error;
pub mod model;
pub mod npy;
pub mod query;
pub mod routes;

use std::path::PathBuf;
use std::sync::Mutex;

use duckdb::Connection;

pub use routes::router;

#[derive(Clone)]
pub struct AppState {
    pub runs_dir: PathBuf,
    pub clips_dir: Option<PathBuf>,
    pub explain_url: Option<String>,
    pub db: std::sync::Arc<Mutex<Connection>>,
    pub http: reqwest::Client,
}

pub fn open_state(
    runs_dir: PathBuf,
    clips_dir: Option<PathBuf>,
    explain_url: Option<String>,
) -> Result<AppState, duckdb::Error> {
    let db = Connection::open_in_memory()?;
    // Best-effort. The server still answers if this DuckDB build rejects the setting.
    let _ = db.execute_batch("SET threads=4;");
    let http = reqwest::Client::builder().build().expect("reqwest client");
    Ok(AppState {
        runs_dir,
        clips_dir,
        explain_url,
        db: std::sync::Arc::new(Mutex::new(db)),
        http,
    })
}
