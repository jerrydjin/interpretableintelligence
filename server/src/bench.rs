//! Latency of the read queries on one million synthetic rows.
//!
//! `docs/HOSTING.md` timed two different DuckDB scans in Python (a clip's
//! ablations, and one feature's aggregate). The numbers below are the queries
//! the server actually serves: a frame's effects, and top clips for a feature.

use std::path::Path;
use std::time::Instant;

use duckdb::Connection;
use serde::Serialize;

use crate::query;

const ROWS: i64 = 1_000_000;
const WARMUP: usize = 5;
const SAMPLES: usize = 80;

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub rows: i64,
    pub threads: i64,
    pub clip_effects_bytes: u64,
    pub feature_acts_bytes: u64,
    pub clip_effects_rows: i64,
    pub feature_acts_rows: i64,
    pub frame_lookup_p50_ms: f64,
    pub frame_lookup_p99_ms: f64,
    pub top_clips_p50_ms: f64,
    pub top_clips_p99_ms: f64,
    pub cold_frame_lookup_ms: f64,
    pub hosting_clip_scan_p50_ms: f64,
    pub hosting_clip_scan_p99_ms: f64,
    pub hosting_feature_scan_p50_ms: f64,
    pub hosting_feature_scan_p99_ms: f64,
}

pub fn run(dir: &Path) -> Result<BenchReport, duckdb::Error> {
    let effects = dir.join("clip_effects.parquet");
    let acts = dir.join("feature_acts.parquet");
    let setup = Connection::open_in_memory()?;
    setup.execute_batch("SET threads=4;")?;
    copy_effects(&setup, &effects)?;
    copy_acts(&setup, &acts)?;
    let threads: i64 = setup.query_row("SELECT current_setting('threads')::BIGINT", [], |row| {
        row.get(0)
    })?;

    let conn = Connection::open_in_memory()?;
    conn.execute_batch("SET threads=4;")?;

    let cold = {
        let cold_conn = Connection::open_in_memory()?;
        cold_conn.execute_batch("SET threads=4;")?;
        let started = Instant::now();
        let _ = query::effects_at(&cold_conn, &effects, "clip-0000", 0, 0)?;
        started.elapsed().as_secs_f64() * 1000.0
    };

    let frame_samples = time_ms(WARMUP, SAMPLES, || {
        query::effects_at(&conn, &effects, "clip-0000", 0, 0).unwrap();
    });
    let top_samples = time_ms(WARMUP, SAMPLES, || {
        query::top_clips(&conn, &acts, Some(0)).unwrap();
    });
    let clip_samples = time_ms(WARMUP, SAMPLES, || {
        hosting_clip(&conn, &effects).unwrap();
    });
    let feature_samples = time_ms(WARMUP, SAMPLES, || {
        hosting_feature(&conn, &effects).unwrap();
    });

    Ok(BenchReport {
        rows: ROWS,
        threads,
        clip_effects_bytes: std::fs::metadata(&effects)
            .map(|meta| meta.len())
            .unwrap_or(0),
        feature_acts_bytes: std::fs::metadata(&acts).map(|meta| meta.len()).unwrap_or(0),
        clip_effects_rows: count_rows(&conn, &effects)?,
        feature_acts_rows: count_rows(&conn, &acts)?,
        frame_lookup_p50_ms: percentile(&frame_samples, 50.0),
        frame_lookup_p99_ms: percentile(&frame_samples, 99.0),
        top_clips_p50_ms: percentile(&top_samples, 50.0),
        top_clips_p99_ms: percentile(&top_samples, 99.0),
        cold_frame_lookup_ms: cold,
        hosting_clip_scan_p50_ms: percentile(&clip_samples, 50.0),
        hosting_clip_scan_p99_ms: percentile(&clip_samples, 99.0),
        hosting_feature_scan_p50_ms: percentile(&feature_samples, 50.0),
        hosting_feature_scan_p99_ms: percentile(&feature_samples, 99.0),
    })
}

pub fn format_report(report: &BenchReport) -> String {
    format!(
        "\
rows requested: {rows}
clip_effects rows: {effect_rows}
feature_acts rows: {act_rows}
duckdb threads: {threads}
clip_effects.parquet: {effects_mb:.2} MB
feature_acts.parquet: {acts_mb:.2} MB

frame lookup (effects at one clip/window/t, {rows} clip_effects rows)
  p50 {frame_p50:.2} ms
  p99 {frame_p99:.2} ms
  cold {cold:.2} ms
top_clips (one feature, {rows} feature_acts rows)
  p50 {top_p50:.2} ms
  p99 {top_p99:.2} ms

HOSTING.md Python DuckDB, 1_000_000 effect rows (different machine, different queries):
  ablations for one clip, order by effect, limit 10: median 16.1 ms, p95 19.3 ms, max 20.5 ms
  aggregate for one feature: median 7.7 ms, p95 9.3 ms, max 9.4 ms
  first query on a fresh connection: 19.9 ms
same scans here:
  ablations for one clip, limit 10: p50 {clip_p50:.2} ms, p99 {clip_p99:.2} ms
  aggregate for one feature: p50 {feat_p50:.2} ms, p99 {feat_p99:.2} ms
",
        rows = report.rows,
        effect_rows = report.clip_effects_rows,
        act_rows = report.feature_acts_rows,
        threads = report.threads,
        effects_mb = report.clip_effects_bytes as f64 / (1024.0 * 1024.0),
        acts_mb = report.feature_acts_bytes as f64 / (1024.0 * 1024.0),
        frame_p50 = report.frame_lookup_p50_ms,
        frame_p99 = report.frame_lookup_p99_ms,
        cold = report.cold_frame_lookup_ms,
        top_p50 = report.top_clips_p50_ms,
        top_p99 = report.top_clips_p99_ms,
        clip_p50 = report.hosting_clip_scan_p50_ms,
        clip_p99 = report.hosting_clip_scan_p99_ms,
        feat_p50 = report.hosting_feature_scan_p50_ms,
        feat_p99 = report.hosting_feature_scan_p99_ms,
    )
}

fn copy_effects(conn: &Connection, path: &Path) -> Result<(), duckdb::Error> {
    let sql = format!(
        "COPY (
            SELECT (i % 4096)::INTEGER AS feature_id,
                   printf('clip-%04d', (i % 4000)) AS clip_id,
                   (i % 3)::SMALLINT AS \"window\",
                   (i % 16)::SMALLINT AS t,
                   ((hash(i) % 100000000) / 1e6)::FLOAT AS effect,
                   CASE WHEN hash(i) % 5 = 0 THEN 'ablation' ELSE 'estimate' END AS effect_kind
            FROM range({ROWS}) t(i)
         ) TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
        sql_path(path)
    );
    conn.execute_batch(&sql)
}

fn copy_acts(conn: &Connection, path: &Path) -> Result<(), duckdb::Error> {
    let sql = format!(
        "COPY (
            SELECT (i % 32)::INTEGER AS feature_id,
                   printf('clip-%04d', (i % 4000)) AS clip_id,
                   (i % 3)::SMALLINT AS \"window\",
                   (i % 16)::SMALLINT AS t,
                   (i % 8)::SMALLINT AS h,
                   ((i // 8) % 8)::SMALLINT AS w,
                   ((hash(i + 1) % 100000000) / 1e5)::FLOAT AS value
            FROM range({ROWS}) t(i)
         ) TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
        sql_path(path)
    );
    conn.execute_batch(&sql)
}

fn hosting_clip(conn: &Connection, path: &Path) -> Result<usize, duckdb::Error> {
    let mut stmt = conn.prepare(
        "SELECT feature_id, effect
         FROM read_parquet(?)
         WHERE clip_id = 'clip-0000' AND effect_kind = 'ablation'
         ORDER BY effect DESC
         LIMIT 10",
    )?;
    let rows = stmt.query(duckdb::params![path.to_string_lossy().to_string()])?;
    Ok(rows_len(rows))
}

fn hosting_feature(conn: &Connection, path: &Path) -> Result<(i64, f64), duckdb::Error> {
    conn.query_row(
        "SELECT COUNT(*)::BIGINT, COALESCE(AVG(CAST(effect AS DOUBLE)), 0)
         FROM read_parquet(?)
         WHERE feature_id = 0",
        duckdb::params![path.to_string_lossy().to_string()],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
}

fn rows_len(mut rows: duckdb::Rows<'_>) -> usize {
    let mut n = 0;
    while let Ok(Some(_)) = rows.next() {
        n += 1;
    }
    n
}

fn time_ms(warmup: usize, samples: usize, mut body: impl FnMut()) -> Vec<f64> {
    for _ in 0..warmup {
        body();
    }
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        body();
        out.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    out.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn count_rows(conn: &Connection, path: &Path) -> Result<i64, duckdb::Error> {
    conn.query_row(
        "SELECT COUNT(*)::BIGINT FROM read_parquet(?)",
        duckdb::params![path.to_string_lossy().to_string()],
        |row| row.get(0),
    )
}

fn sql_path(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}
