//! An explanation may cite a feature only when the scoreboard says `real` is
//! true and the number is an ablation. Everything else is `unverified`.

use std::collections::BTreeMap;

use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Gate {
    pub real: bool,
    pub label: String,
}

pub fn cites(real: bool, effect_kind: &str) -> bool {
    real && effect_kind == "ablation"
}

pub fn unverified_reason(real: bool, effect_kind: &str) -> &'static str {
    if effect_kind != "ablation" {
        "estimate only"
    } else if !real {
        "scoreboard real is false"
    } else {
        "not cited"
    }
}

/// Rewrite `payload` so `answer` contains only citations that the scoreboard allows.
pub fn sanitize(payload: &mut Value, gates: &BTreeMap<i64, Gate>) -> Result<(), String> {
    let Some(answer) = payload.get("answer") else {
        return Ok(());
    };
    if !answer.is_array() {
        return Err("explanation answer is not a list".into());
    }
    let mut kept_groups = Vec::new();
    let mut moved = Vec::new();
    for group in answer.as_array().unwrap() {
        let window = group.get("window").cloned().unwrap_or(Value::Null);
        let t = group.get("t").cloned().unwrap_or(Value::Null);
        let features = group
            .get("features")
            .and_then(Value::as_array)
            .ok_or_else(|| "explanation group has no features".to_string())?;
        let mut kept = Vec::new();
        for feature in features {
            let feature_id = feature
                .get("feature_id")
                .and_then(Value::as_i64)
                .ok_or_else(|| "explanation feature has no integer id".to_string())?;
            let kind = feature
                .get("effect_kind")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let claimed_real = feature.get("real") == Some(&Value::Bool(true));
            let gate = gates.get(&feature_id);
            let scoreboard_real = gate.is_some_and(|gate| gate.real);
            let effect = feature.get("effect").cloned().unwrap_or(Value::Null);
            if !effect.is_number() {
                return Err(format!(
                    "explanation effect for feature {feature_id} is not a finite number"
                ));
            }
            if cites(claimed_real && scoreboard_real, &kind) {
                let label = gate.map(|gate| gate.label.clone()).unwrap_or_default();
                kept.push(json!({
                    "feature_id": feature_id,
                    "label": label,
                    "effect": effect,
                    "effect_kind": "ablation",
                    "real": true,
                }));
            } else {
                let label = feature
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let real_value = match gate {
                    Some(gate) => Value::Bool(gate.real),
                    None => Value::Bool(false),
                };
                moved.push(json!({
                    "feature_id": feature_id,
                    "label": if let Some(gate) = gate { gate.label.clone() } else { label },
                    "window": window.clone(),
                    "t": t.clone(),
                    "effect": effect,
                    "effect_kind": kind,
                    "real": real_value,
                    "reason": unverified_reason(scoreboard_real, &kind),
                }));
            }
        }
        if !kept.is_empty() {
            kept_groups.push(json!({
                "window": window,
                "t": t,
                "features": kept,
            }));
        }
    }
    let mut unverified = payload
        .get("unverified")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    unverified.extend(moved);
    unverified.sort_by_key(unverified_key);
    if let Some(object) = payload.as_object_mut() {
        object.insert("answer".into(), Value::Array(kept_groups));
        object.insert("unverified".into(), Value::Array(unverified));
    }
    Ok(())
}

fn unverified_key(row: &Value) -> (i64, i64, i64, String) {
    (
        row.get("window").and_then(Value::as_i64).unwrap_or(0),
        row.get("t").and_then(Value::as_i64).unwrap_or(0),
        row.get("feature_id").and_then(Value::as_i64).unwrap_or(0),
        row.get("effect_kind")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    )
}

pub fn gates_from_scoreboard(scoreboard: &Value) -> BTreeMap<i64, Gate> {
    let mut gates = BTreeMap::new();
    let Some(features) = scoreboard.get("features").and_then(Value::as_array) else {
        return gates;
    };
    for feature in features {
        let Some(id) = feature.get("id").and_then(Value::as_i64) else {
            continue;
        };
        let real = feature.get("real") == Some(&Value::Bool(true));
        let label = feature
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        gates.insert(id, Gate { real, label });
    }
    gates
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_a_real_ablation_is_cited() {
        assert!(cites(true, "ablation"));
        assert!(!cites(true, "estimate"));
        assert!(!cites(false, "ablation"));
        assert!(!cites(false, "estimate"));
    }

    #[test]
    fn a_lying_citation_is_moved_to_unverified() {
        let mut payload = json!({
            "run_id": "demo",
            "answer": [{
                "window": 0,
                "t": 1,
                "features": [
                    {"feature_id": 0, "label": "stripe", "effect": 0.2, "effect_kind": "ablation", "real": true},
                    {"feature_id": 1, "label": "flicker", "effect": 0.01, "effect_kind": "ablation", "real": true},
                    {"feature_id": 0, "label": "stripe", "effect": 0.5, "effect_kind": "estimate", "real": true}
                ]
            }],
            "unverified": []
        });
        let mut gates = BTreeMap::new();
        gates.insert(
            0,
            Gate {
                real: true,
                label: "stripe".into(),
            },
        );
        gates.insert(
            1,
            Gate {
                real: false,
                label: "flicker".into(),
            },
        );
        sanitize(&mut payload, &gates).unwrap();
        let cited = &payload["answer"][0]["features"];
        assert_eq!(cited.as_array().unwrap().len(), 1);
        assert_eq!(cited[0]["feature_id"], 0);
        assert_eq!(cited[0]["effect_kind"], "ablation");
        assert_eq!(cited[0]["real"], true);
        let unverified = payload["unverified"].as_array().unwrap();
        assert_eq!(unverified.len(), 2);
        assert!(unverified
            .iter()
            .all(|row| row["real"] != true || row["effect_kind"] != "ablation"));
        assert!(unverified.iter().any(|row| row["feature_id"] == 1
            && row["reason"] == "scoreboard real is false"
            && row["label"] == "flicker"));
        assert!(unverified
            .iter()
            .any(|row| row["effect_kind"] == "estimate" && row["reason"] == "estimate only"));
    }
}
