//! DuckDB queries over a run folder's parquet. The row order matches the viewer:
//! highest activation wins, and a tie keeps the smaller `(window, t, h, w)`.

use std::path::Path;

use duckdb::{params, Connection};

#[derive(Debug, Clone)]
pub struct ActCell {
    pub feature_id: i64,
    pub h: i64,
    pub w: i64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct EffectCell {
    pub feature_id: i64,
    pub effect_kind: String,
    pub effect: f64,
}

#[derive(Debug, Clone)]
pub struct TopClipRow {
    pub feature_id: i64,
    pub clip_id: String,
    pub window: i64,
    pub t: i64,
    pub h: i64,
    pub w: i64,
    pub value: f64,
}

pub fn acts_at(
    conn: &Connection,
    parquet: &Path,
    clip_id: &str,
    window: i64,
    t: i64,
) -> Result<Vec<ActCell>, duckdb::Error> {
    let mut stmt = conn.prepare(
        "SELECT CAST(feature_id AS BIGINT),
                CAST(h AS BIGINT),
                CAST(w AS BIGINT),
                MAX(CAST(value AS DOUBLE)) AS value
         FROM read_parquet(?)
         WHERE clip_id = ? AND \"window\" = ? AND t = ?
         GROUP BY feature_id, h, w",
    )?;
    let mut rows = stmt.query(params![path_str(parquet), clip_id, window, t])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(ActCell {
            feature_id: row.get(0)?,
            h: row.get(1)?,
            w: row.get(2)?,
            value: row.get(3)?,
        });
    }
    Ok(out)
}

pub fn effects_at(
    conn: &Connection,
    parquet: &Path,
    clip_id: &str,
    window: i64,
    t: i64,
) -> Result<Vec<EffectCell>, duckdb::Error> {
    let mut stmt = conn.prepare(
        "SELECT CAST(feature_id AS BIGINT),
                effect_kind,
                CAST(effect AS DOUBLE) AS effect
         FROM read_parquet(?)
         WHERE clip_id = ? AND \"window\" = ? AND t = ?",
    )?;
    let mut rows = stmt.query(params![path_str(parquet), clip_id, window, t])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(EffectCell {
            feature_id: row.get(0)?,
            effect_kind: row.get(1)?,
            effect: row.get(2)?,
        });
    }
    Ok(out)
}

pub fn top_clips(
    conn: &Connection,
    parquet: &Path,
    feature_id: Option<i64>,
) -> Result<Vec<TopClipRow>, duckdb::Error> {
    let filter = if feature_id.is_some() {
        "WHERE CAST(feature_id AS BIGINT) = ?"
    } else {
        ""
    };
    let sql = format!(
        "WITH cell AS (
            SELECT CAST(feature_id AS BIGINT) AS feature_id,
                   CAST(clip_id AS VARCHAR) AS clip_id,
                   CAST(\"window\" AS BIGINT) AS \"window\",
                   CAST(t AS BIGINT) AS t,
                   CAST(h AS BIGINT) AS h,
                   CAST(w AS BIGINT) AS w,
                   MAX(CAST(value AS DOUBLE)) AS value
            FROM read_parquet(?)
            {filter}
            GROUP BY feature_id, clip_id, \"window\", t, h, w
        ),
        best_site AS (
            SELECT feature_id, clip_id, \"window\", t, h, w, value
            FROM (
                SELECT feature_id, clip_id, \"window\", t, h, w, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY feature_id, clip_id, \"window\", t
                           ORDER BY value DESC, h ASC, w ASC
                       ) AS rn
                FROM cell
            )
            WHERE rn = 1
        ),
        best_time AS (
            SELECT feature_id, clip_id, \"window\", t, h, w, value
            FROM (
                SELECT feature_id, clip_id, \"window\", t, h, w, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY feature_id, clip_id
                           ORDER BY value DESC, \"window\" ASC, t ASC, h ASC, w ASC
                       ) AS rn
                FROM best_site
            )
            WHERE rn = 1
        ),
        ranked AS (
            SELECT feature_id, clip_id, \"window\", t, h, w, value
            FROM (
                SELECT feature_id, clip_id, \"window\", t, h, w, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY feature_id
                           ORDER BY value DESC, clip_id ASC
                       ) AS rn
                FROM best_time
            )
            WHERE rn <= 20
        )
        SELECT feature_id, clip_id, \"window\", t, h, w, value
        FROM ranked
        ORDER BY feature_id ASC, value DESC, clip_id ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = if let Some(feature_id) = feature_id {
        stmt.query(params![path_str(parquet), feature_id])?
    } else {
        stmt.query(params![path_str(parquet)])?
    };
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(TopClipRow {
            feature_id: row.get(0)?,
            clip_id: row.get(1)?,
            window: row.get(2)?,
            t: row.get(3)?,
            h: row.get(4)?,
            w: row.get(5)?,
            value: row.get(6)?,
        });
    }
    Ok(out)
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn() -> Connection {
        Connection::open_in_memory().unwrap()
    }

    #[test]
    fn frame_cells_keep_the_max_and_top_clips_break_ties() {
        let dir = tempfile::tempdir().unwrap();
        let acts = dir.path().join("acts.parquet");
        let db = conn();
        db.execute_batch(&format!(
            "CREATE TABLE acts (
                feature_id INTEGER, clip_id VARCHAR, \"window\" SMALLINT, t SMALLINT,
                h SMALLINT, w SMALLINT, value FLOAT
            );
            INSERT INTO acts VALUES
                (0, 'clip-b', 0, 0, 1, 0, 1.0),
                (0, 'clip-b', 0, 0, 0, 1, 1.0),
                (0, 'clip-b', 0, 1, 0, 0, 1.0),
                (0, 'clip-a', 0, 0, 0, 0, 0.5),
                (0, 'clip-a', 0, 0, 0, 0, 0.25),
                (1, 'clip-a', 0, 0, 0, 0, 3.0);
            COPY acts TO '{}' (FORMAT PARQUET);",
            acts.display()
        ))
        .unwrap();

        let cells = acts_at(&db, &acts, "clip-a", 0, 0).unwrap();
        let feature_0 = cells.iter().find(|cell| cell.feature_id == 0).unwrap();
        assert!((feature_0.value - 0.5).abs() < 1e-9);

        let tops = top_clips(&db, &acts, Some(0)).unwrap();
        assert_eq!(tops.len(), 2);
        // clip-b's sites are tied at 1.0. The best site at t=0 is (h=0, w=1).
        // Across time, (window=0, t=0, h=0, w=1) beats (0, 1, 0, 0). Value 1.0
        // ranks ahead of clip-a's 0.5.
        assert_eq!(tops[0].clip_id, "clip-b");
        assert_eq!((tops[0].t, tops[0].h, tops[0].w), (0, 0, 1));
        assert_eq!(tops[1].clip_id, "clip-a");
        assert!((tops[1].value - 0.5).abs() < 1e-9);
    }
}
