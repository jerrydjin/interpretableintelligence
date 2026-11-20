//! Frame mapping: which `(window, t)` token a source frame shows, and back.
//!
//! A token at `(window, t)` lands on `window_start + t * tubelet * stride`. Starts and stride
//! come from, in order: explicit per-clip starts (or the same keys on the grid), Jeff's placement
//! from `manifest.extra` (`frames`, `stride`, `windows`), or tiling from frame 0 at stride 1.

use serde_json::{Map, Value};

/// `manifest.grid`, parsed the way the viewer reports it back in `/api/run`.
#[derive(Debug, Clone, Default)]
pub struct Grid {
    pub t: i64,
    pub h: i64,
    pub w: i64,
    pub tubelet: i64,
    pub patch_px: i64,
    pub window_stride: Option<i64>,
    pub window_start: Option<i64>,
    pub window_starts: Option<Vec<i64>>,
    /// Which of t, h, w the manifest had (they are reported only when present).
    pub has: [bool; 3],
}

impl Grid {
    pub fn parse(raw: &Map<String, Value>) -> Grid {
        let int_or = |key: &str| raw.get(key).and_then(as_int);
        let loose = |key: &str| raw.get(key).and_then(loose_int);
        let tubelet_raw = int_or("tubelet");
        let starts = match raw.get("window_starts") {
            Some(Value::Array(items)) if !items.is_empty() && items.iter().all(|item| as_int(item).is_some()) => {
                Some(items.iter().filter_map(as_int).collect())
            }
            _ => None,
        };
        let mut grid = Grid {
            t: loose("t").unwrap_or(0),
            h: loose("h").unwrap_or(0),
            w: loose("w").unwrap_or(0),
            tubelet: 1,
            patch_px: int_or("patch_px").unwrap_or(1),
            window_stride: int_or("window_stride"),
            window_start: int_or("window_start"),
            window_starts: starts,
            has: [raw.contains_key("t"), raw.contains_key("h"), raw.contains_key("w")],
        };
        grid.tubelet = tubelet_raw.unwrap_or_else(|| {
            let fallback = loose("tubelet").unwrap_or(0);
            if fallback == 0 { 1 } else { fallback.max(1) }
        });
        grid
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        for (index, (key, value)) in [("t", self.t), ("h", self.h), ("w", self.w)].into_iter().enumerate() {
            if self.has[index] {
                out.insert(key.into(), value.into());
            }
        }
        out.insert("tubelet".into(), self.tubelet.into());
        out.insert("patch_px".into(), self.patch_px.into());
        if let Some(stride) = self.window_stride {
            out.insert("window_stride".into(), stride.into());
        }
        if let Some(start) = self.window_start {
            out.insert("window_start".into(), start.into());
        }
        if let Some(starts) = &self.window_starts {
            out.insert("window_starts".into(), starts.clone().into());
        }
        Value::Object(out)
    }

    /// The subset a clip payload carries.
    pub fn clip_json(&self) -> Value {
        let mut out = Map::new();
        for (index, (key, value)) in [("t", self.t), ("h", self.h), ("w", self.w)].into_iter().enumerate() {
            if self.has[index] {
                out.insert(key.into(), value.into());
            }
        }
        out.insert("tubelet".into(), self.tubelet.into());
        out.insert("patch_px".into(), self.patch_px.into());
        Value::Object(out)
    }

    fn n_tokens(&self) -> i64 {
        or_one(self.t).max(1)
    }

    fn tubelet_of(&self) -> i64 {
        or_one(self.tubelet).max(1)
    }

    fn span(&self) -> i64 {
        self.n_tokens() * self.tubelet_of()
    }

    fn stride_or(&self, span: Option<i64>) -> i64 {
        match self.window_stride {
            Some(stride) => stride.max(1),
            None => span.unwrap_or_else(|| self.span()),
        }
    }
}

/// A JSON integer (not a bool, not a float).
pub fn as_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        _ => None,
    }
}

/// Python's `int(x)` for the grid's required keys: floats truncate.
fn loose_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64().or_else(|| number.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(flag) => Some(*flag as i64),
        _ => None,
    }
}

fn or_one(value: i64) -> i64 {
    if value == 0 {
        1
    } else {
        value
    }
}

fn get_int(object: Option<&Map<String, Value>>, key: &str) -> Option<i64> {
    object.and_then(|object| object.get(key)).and_then(as_int)
}

/// Jeff's window placement. Returns `(starts, stride)`.
pub fn sampled_window_starts(n_frames: i64, n_windows: i64, frames: i64, stride: i64) -> (Vec<i64>, i64) {
    let mut stride = stride;
    let mut span = frames * stride;
    if n_frames < span && stride > 1 {
        stride = 1;
        span = frames;
    }
    let k = 1.max(n_windows.min(n_frames.div_euclid(span)));
    if k == 1 {
        return (vec![0.max((n_frames - span).div_euclid(2))], stride);
    }
    let gap = (n_frames - k * span) as f64 / (k - 1) as f64;
    let starts = (0..k)
        .map(|i| (i as f64 * (span as f64 + gap)).round_ties_even() as i64)
        .collect();
    (starts, stride)
}

fn sampling_spec(extra: Option<&Map<String, Value>>) -> Option<(i64, i64, Option<i64>)> {
    let extra = extra?;
    if !["frames", "stride", "windows"].iter().any(|key| extra.contains_key(*key)) {
        return None;
    }
    let frames = get_int(Some(extra), "frames").unwrap_or(16);
    let stride = get_int(Some(extra), "stride").unwrap_or(2);
    let windows = get_int(Some(extra), "windows");
    Some((frames, stride, windows))
}

fn extend_starts(starts: Vec<i64>, n_windows: i64, step: i64) -> Vec<i64> {
    let mut extended = starts;
    let step = step.max(1);
    while (extended.len() as i64) <= n_windows {
        let next = extended.last().map(|last| last + step).unwrap_or(0);
        extended.push(next);
    }
    extended
}

/// `(window_starts, sample_stride)` for one clip. `n_windows` is the highest window index needed.
pub fn clip_sampling(
    clip: Option<&Map<String, Value>>,
    grid: &Grid,
    n_windows: i64,
    extra: Option<&Map<String, Value>>,
) -> (Vec<i64>, i64) {
    let explicit = explicit_window_starts(clip, grid);
    let explicit_stride = get_int(clip, "stride");
    if let Some(explicit) = explicit {
        let stride = explicit_stride.unwrap_or(1);
        return (extend_starts(explicit, n_windows, grid.stride_or(None)), stride.max(1));
    }
    let n_frames = get_int(clip, "n_frames");
    if let (Some((frames, requested, extra_windows)), Some(n_frames)) = (sampling_spec(extra), n_frames) {
        let requested = explicit_stride.unwrap_or(requested);
        let count = get_int(clip, "n_windows").or(extra_windows).unwrap_or(n_windows + 1);
        let (starts, stride) = sampled_window_starts(n_frames, count.max(1), frames.max(1), requested.max(1));
        let step = (frames * stride.max(1)).max(1);
        return (extend_starts(starts, n_windows, step), stride.max(1));
    }
    let stride = explicit_stride.unwrap_or(1);
    (extend_starts(Vec::new(), n_windows, grid.stride_or(None)), stride.max(1))
}

/// Window starts that reach `frame`. A clip tiled from frame 0 grows one window at a time.
pub fn starts_covering(
    clip: Option<&Map<String, Value>>,
    grid: &Grid,
    extra: Option<&Map<String, Value>>,
    frame: i64,
) -> (Vec<i64>, i64) {
    let hint = get_int(clip, "n_windows").map(|n| n - 1).unwrap_or(0);
    let (mut starts, stride) = clip_sampling(clip, grid, hint.max(0), extra);
    if sampling_spec(extra).is_some() || explicit_window_starts(clip, grid).is_some() {
        return (starts, stride);
    }
    let step = grid.tubelet_of() * stride.max(1);
    let span = grid.n_tokens() * step;
    let mut guard = 0;
    while let Some(&last) = starts.last() {
        if !(last + span <= frame && guard < frame + 1) {
            break;
        }
        starts.push(last + span);
        guard += 1;
    }
    (starts, stride)
}

/// `(window, t)`. `t` is `None` when `frame` sits in no window; a frame in a gap takes the
/// nearer edge token.
pub fn frame_to_token(frame: i64, starts: &[i64], grid: &Grid, sample_stride: i64) -> (i64, Option<i64>) {
    let fallback = [0];
    let starts = if starts.is_empty() { &fallback[..] } else { starts };
    let step = grid.tubelet_of() * sample_stride.max(1);
    let n_tokens = grid.n_tokens();
    let span = n_tokens * step;
    if frame < starts[0] {
        return (0, None);
    }
    let mut window = 0usize;
    for (index, start) in starts.iter().enumerate() {
        if *start <= frame {
            window = index;
        } else {
            break;
        }
    }
    let offset = frame - starts[window];
    if offset < span {
        return (window as i64, Some(offset.div_euclid(step)));
    }
    let Some(&next_start) = starts.get(window + 1) else {
        return (window as i64, None);
    };
    if offset - span <= next_start - frame {
        return (window as i64, Some(n_tokens - 1));
    }
    (window as i64 + 1, Some(0))
}

/// Inverse of [`frame_to_token`]: `window_start + t * tubelet * stride`.
pub fn token_to_frame(window: i64, t: i64, starts: &[i64], grid: &Grid, sample_stride: i64) -> i64 {
    let step = grid.tubelet_of() * sample_stride.max(1);
    if window >= 0 && (window as usize) < starts.len() {
        return starts[window as usize] + t * step;
    }
    let base = starts.last().copied().unwrap_or(0);
    let beyond = if starts.is_empty() { window } else { window - (starts.len() as i64 - 1) };
    base + beyond * grid.stride_or(None) * sample_stride.max(1) + t * step
}

fn explicit_window_starts(clip: Option<&Map<String, Value>>, grid: &Grid) -> Option<Vec<i64>> {
    if let Some(Value::Array(items)) = clip.and_then(|clip| clip.get("window_starts")) {
        if !items.is_empty() && items.iter().all(|item| as_int(item).is_some()) {
            return Some(items.iter().filter_map(as_int).collect());
        }
    }
    if let Some(starts) = &grid.window_starts {
        return Some(starts.clone());
    }
    if let Some(clip) = clip {
        if let Some(parsed) = parse_windows(clip.get("windows"), grid) {
            return Some(parsed);
        }
        if let Some(start) = get_int(Some(clip), "window_start") {
            return Some(vec![start]);
        }
    }
    grid.window_start.map(|start| vec![start])
}

/// `dict.get(a, dict.get(b, ...))`: the first key that is present wins, even when its value is null.
fn first_present<'a>(item: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| item.get(*key))
}

fn parse_windows(windows: Option<&Value>, grid: &Grid) -> Option<Vec<i64>> {
    let Some(Value::Array(items)) = windows else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        match item {
            Value::Object(item) => {
                let start = first_present(item, &["start_frame", "start", "frame"]).and_then(as_int);
                let window = match first_present(item, &["index", "window"]) {
                    Some(value) => as_int(value),
                    None => Some(index as i64),
                };
                if let (Some(start), Some(window)) = (start, window) {
                    pairs.push((window, start));
                }
            }
            other => {
                if let Some(start) = as_int(other) {
                    pairs.push((index as i64, start));
                }
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort();
    let stride = grid.stride_or(None);
    let mut starts: Vec<i64> = Vec::new();
    for (window, start) in pairs {
        while (starts.len() as i64) < window {
            let next = starts.last().map(|last| last + stride).unwrap_or(0);
            starts.push(next);
        }
        if starts.len() as i64 == window {
            starts.push(start);
        } else if window >= 0 {
            starts[window as usize] = start;
        } else {
            // Python's negative index counts from the end.
            let len = starts.len() as i64;
            let at = len + window;
            if at >= 0 {
                starts[at as usize] = start;
            }
        }
    }
    Some(starts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn grid(raw: Value) -> Grid {
        Grid::parse(raw.as_object().unwrap())
    }

    #[test]
    fn tiles_from_zero_by_default() {
        let g = grid(json!({"t": 8, "h": 2, "w": 2, "tubelet": 2}));
        let (starts, stride) = clip_sampling(None, &g, 1, None);
        assert_eq!((starts, stride), (vec![0, 16], 1));
        assert_eq!(frame_to_token(5, &[0, 16], &g, 1), (0, Some(2)));
        assert_eq!(token_to_frame(1, 3, &[0, 16], &g, 1), 22);
    }

    #[test]
    fn jeffs_placement_centres_one_window_and_spreads_three() {
        assert_eq!(sampled_window_starts(40, 1, 16, 2), (vec![4], 2));
        assert_eq!(sampled_window_starts(20, 3, 16, 2), (vec![2], 1));
        assert_eq!(sampled_window_starts(167, 3, 16, 2), (vec![0, 68, 135], 2));
    }

    #[test]
    fn gap_frame_takes_the_nearer_edge() {
        let g = grid(json!({"t": 2, "h": 1, "w": 1, "tubelet": 2}));
        // Windows cover [0, 4) and [10, 14).
        assert_eq!(frame_to_token(5, &[0, 10], &g, 1), (0, Some(1)));
        assert_eq!(frame_to_token(8, &[0, 10], &g, 1), (1, Some(0)));
        assert_eq!(frame_to_token(20, &[0, 10], &g, 1), (1, None));
    }
}
