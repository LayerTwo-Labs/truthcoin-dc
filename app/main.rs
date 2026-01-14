#![feature(try_find)]

use std::path::Path;

use clap::Parser as _;
use tokio::{signal::ctrl_c, sync::oneshot};
use tracing_subscriber::{
    Layer, filter as tracing_filter, layer::SubscriberExt,
};

mod app;
mod cli;
mod gui;
mod line_buffer;
mod rpc_server;
mod util;

use line_buffer::{LineBuffer, LineBufferWriter};
use util::saturating_pred_level;

/// The empty string target `""` can be used to set a default level.
fn targets_directive_str<'a, Targets>(targets: Targets) -> String
where
    Targets: IntoIterator<Item = (&'a str, tracing::Level)>,
{
    targets
        .into_iter()
        .map(|(target, level)| {
            let level = level.as_str().to_ascii_lowercase();
            if target.is_empty() {
                level
            } else {
                format!("{target}={level}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Must be held for the lifetime of the program in order to keep the file
/// logger alive.
type RollingLoggerGuard = tracing_appender::non_blocking::WorkerGuard;

/// Rolling file logger.
/// Returns a guard that must be held for the lifetime of the program in order
/// to keep the file logger alive.
fn rolling_logger<S>(
    log_dir: &Path,
    log_level: tracing::Level,
) -> anyhow::Result<(impl Layer<S>, RollingLoggerGuard)>
where
    S: tracing::Subscriber
        + for<'s> tracing_subscriber::registry::LookupSpan<'s>,
{
    const LOG_FILE_SUFFIX: &str = "log";
    let rolling_log_appender = tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_suffix(LOG_FILE_SUFFIX)
        .build(log_dir)?;
    let (non_blocking_rolling_log_writer, rolling_log_guard) =
        tracing_appender::non_blocking(rolling_log_appender);
    let level_filter = tracing_filter::Targets::new().with_default(log_level);
    let rolling_log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_ansi(false)
        .with_writer(non_blocking_rolling_log_writer)
        .with_filter(level_filter);
    Ok((rolling_log_layer, rolling_log_guard))
}

// Configure loggers.
// If the file logger is set, returns a guard that must be held for the
// lifetime of the program in order to keep the file logger alive.
fn set_tracing_subscriber(
    file_log_level: tracing::Level,
    log_dir: Option<&Path>,
    log_level: tracing::Level,
) -> anyhow::Result<(LineBuffer, Option<RollingLoggerGuard>)> {
    let targets_filter = {
        let default_directives_str = targets_directive_str([
            ("", saturating_pred_level(log_level)),
            ("bip300301", log_level),
            ("jsonrpsee_core::tracing", log_level),
            (
                "h2::codec::framed_read",
                saturating_pred_level(saturating_pred_level(log_level)),
            ),
            (
                "h2::codec::framed_write",
                saturating_pred_level(saturating_pred_level(log_level)),
            ),
            ("truthcoin_dc", log_level),
            ("truthcoin_dc_app", log_level),
            (
                "tower::buffer::worker",
                saturating_pred_level(saturating_pred_level(log_level)),
            ),
        ]);
        let directives_str =
            match std::env::var(tracing_filter::EnvFilter::DEFAULT_ENV) {
                Ok(env_directives) => {
                    format!("{default_directives_str},{env_directives}")
                }
                Err(std::env::VarError::NotPresent) => default_directives_str,
                Err(err) => return Err(anyhow::Error::from(err)),
            };
        tracing_filter::EnvFilter::builder().parse(directives_str)?
    };
    // Adding source location here means that the file name + line number
    // is included, in such a way that it can be clicked on from within
    // the IDE, and you're sent right to the specific line of code. Very handy!
    let stdout_format =
        tracing_subscriber::fmt::format().with_source_location(true);
    let mut stdout_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .event_format(stdout_format);
    let is_terminal =
        std::io::IsTerminal::is_terminal(&stdout_layer.writer()());
    stdout_layer.set_ansi(is_terminal);
    let (rolling_log_layer, rolling_log_guard) = match log_dir {
        None => (None, None),
        Some(log_dir) => {
            let (layer, guard) = rolling_logger(log_dir, file_log_level)?;
            (Some(layer), Some(guard))
        }
    };
    let line_buffer = LineBuffer::default();
    let capture_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_line_number(true)
        .with_ansi(false)
        .with_writer(LineBufferWriter::from(&line_buffer));
    let tracing_subscriber = tracing_subscriber::registry()
        .with(targets_filter)
        .with(stdout_layer)
        .with(capture_layer)
        .with(rolling_log_layer);
    tracing::subscriber::set_global_default(tracing_subscriber)
        .expect("setting default subscriber failed");
    Ok((line_buffer, rolling_log_guard))
}

fn run_egui_app(
    config: &crate::cli::Config,
    line_buffer: LineBuffer,
    app: Option<crate::app::App>,
) -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Plain Truthcoin",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(gui::EguiApp::new(
                app,
                cc,
                line_buffer,
                config.rpc_host.clone(),
                config.rpc_port,
                config.network,
            )))
        }),
    )
}

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let config = cli.get_config()?;
    let (line_buffer, _rolling_log_guard) = set_tracing_subscriber(
        config.file_log_level,
        config.log_dir.as_deref(),
        config.log_level,
    )?;
    let (app_tx, app_rx) = oneshot::channel::<anyhow::Error>();
    let app = app::App::new(&config).inspect(|app| {
        // spawn rpc server
        app.runtime.spawn({
            let app = app.clone();
            let rpc_url = config.rpc_url();
            async move {
                tracing::info!("starting RPC server at `{rpc_url}`");
                if let Err(err) = rpc_server::run_server(app, rpc_url).await {
                    app_tx.send(err).expect("failed to send error to app");
                }
            }
        });
    });
    if !config.headless {
        let app = match app {
            Ok(app) => Some(app),
            Err(err) => {
                let err = anyhow::Error::from(err);
                tracing::error!("{err:#}");
                None
            }
        };
        // For GUI mode we want the GUI to start, even if the app fails to start.
        return run_egui_app(&config, line_buffer, app)
            .map_err(|e| anyhow::anyhow!("failed to run egui app: {e:#}"));
    };
    tracing::info!("Running in headless mode");
    drop(line_buffer);
    // If we're headless, we want to exit hard if the app fails to start.
    let app = app?;
    app.runtime.block_on(async move {
        tokio::select! {
            Ok(_) = ctrl_c() => {
                tracing::info!("Shutting down due to process interruption");
                Ok(())
            }
            Ok(err) = app_rx => {
                Err(anyhow::anyhow!("received error from RPC server: {err:#} ({err:?})"))
            }
        }
    })
}
