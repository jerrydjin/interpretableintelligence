//! The download-lib DuckDB build links `libduckdb.so` without an rpath.
//! Point the server binary at that library, and also copy it beside the
//! executable so `target/debug/interpretableintelligence-server` runs outside `cargo run`.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let Some(library) = find_library() else {
        println!("cargo:warning=libduckdb.so was not found; the binary may fail to start");
        return;
    };
    if let Some(dir) = library.parent() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        if let Some(profile_dir) = Path::new(&out_dir)
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
        {
            let _ = std::fs::copy(&library, profile_dir.join("libduckdb.so"));
            let deps = profile_dir.join("deps");
            let _ = std::fs::create_dir_all(&deps);
            let _ = std::fs::copy(&library, deps.join("libduckdb.so"));
        }
    }
}

fn find_library() -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let mut cursor = PathBuf::from(out_dir);
        for _ in 0..6 {
            roots.push(cursor.join("duckdb-download"));
            if !cursor.pop() {
                break;
            }
        }
    }
    if let Ok(target) = std::env::var("CARGO_TARGET_DIR") {
        roots.push(PathBuf::from(target).join("duckdb-download"));
    }
    for root in roots {
        if let Some(found) = walk(&root) {
            return Some(found);
        }
    }
    None
}

fn walk(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|name| name.to_str()) == Some("libduckdb.so") {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = walk(&path) {
                return Some(found);
            }
        }
    }
    None
}
