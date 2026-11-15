//! The download-lib DuckDB build links the shared library (`libduckdb.so` on Linux,
//! `libduckdb.dylib` on macOS) without an rpath. Point the server binary at that library,
//! and also copy it beside the executable so `target/debug/interpretableintelligence-server`
//! runs outside `cargo run`.

use std::path::{Path, PathBuf};

/// The shared library's file name and the rpath token for "the executable's directory".
fn platform() -> (&'static str, &'static str) {
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") | Ok("ios") => ("libduckdb.dylib", "@loader_path"),
        _ => ("libduckdb.so", "$ORIGIN"),
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let (name, origin) = platform();
    let Some(library) = find_library(name) else {
        println!("cargo:warning={name} was not found; the binary may fail to start");
        return;
    };
    if let Some(dir) = library.parent() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{origin}");
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        if let Some(profile_dir) = Path::new(&out_dir)
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
        {
            let _ = std::fs::copy(&library, profile_dir.join(name));
            let deps = profile_dir.join("deps");
            let _ = std::fs::create_dir_all(&deps);
            let _ = std::fs::copy(&library, deps.join(name));
        }
    }
}

fn find_library(name: &str) -> Option<PathBuf> {
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
        if let Some(found) = walk(&root, name) {
            return Some(found);
        }
    }
    None
}

fn walk(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|file| file.to_str()) == Some(name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = walk(&path, name) {
                return Some(found);
            }
        }
    }
    None
}
