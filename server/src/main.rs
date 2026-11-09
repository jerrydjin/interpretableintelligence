use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use physint_server::{bench, open_state, router};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "physint-server",
    version,
    about = "Read API over a physint run folder"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve GET /runs, feature and frame reads, clip files, and the viewer assets.
    Serve {
        /// Directory of run folders. `run_id` is the folder name.
        #[arg(long)]
        runs: PathBuf,
        /// Directory of `.npy` clip files. Overrides paths recorded on the manifest.
        #[arg(long)]
        clips: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Use 0 to bind an ephemeral port. The bound URL is printed to stdout.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Base URL of `physint serve`. When unset, POST /explain returns 501.
        #[arg(long, env = "PHYSINT_EXPLAIN_URL")]
        explain_url: Option<String>,
    },
    /// Time frame lookup and top_clips on 1,000,000 synthetic rows.
    Bench {
        /// Directory for the synthetic parquet files. A temp directory is used when omitted.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    match Cli::parse().command {
        Command::Serve {
            runs,
            clips,
            host,
            port,
            explain_url,
        } => serve(runs, clips, host, port, explain_url),
        Command::Bench { out } => match bench_main(out) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(1)
            }
        },
    }
}

fn serve(
    runs: PathBuf,
    clips: Option<PathBuf>,
    host: String,
    port: u16,
    explain_url: Option<String>,
) -> ExitCode {
    if !runs.is_dir() {
        eprintln!("error: runs directory not found: {}", runs.display());
        return ExitCode::from(1);
    }
    if let Some(clips) = &clips {
        if !clips.is_dir() {
            eprintln!("error: clips directory not found: {}", clips.display());
            return ExitCode::from(1);
        }
    }
    let state = match open_state(runs, clips, explain_url) {
        Ok(state) => state,
        Err(err) => {
            eprintln!("error: could not open duckdb: {err}");
            return ExitCode::from(1);
        }
    };
    let app = router(state);
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(1);
        }
    };
    runtime.block_on(async move {
        let listener = match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
            Ok(listener) => listener,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(1);
            }
        };
        let addr = match listener.local_addr() {
            Ok(addr) => addr,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::from(1);
            }
        };
        println!("physint-server listening on http://{addr}");
        let _ = io::stdout().flush();
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("error: {err}");
            return ExitCode::from(1);
        }
        ExitCode::SUCCESS
    })
}

fn bench_main(out: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let temp;
    let dir = if let Some(out) = out {
        std::fs::create_dir_all(&out)?;
        out
    } else {
        temp = tempfile::tempdir()?;
        temp.path().to_path_buf()
    };
    let report = bench::run(&dir)?;
    print!("{}", bench::format_report(&report));
    let _ = io::stdout().flush();
    Ok(())
}
