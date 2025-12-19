//! Radio Speech-to-Text CLI Application

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

use stt_rs::{AudioCapture, AudioPreprocessor, Config, OutputWriter, SttEngine};

/// Radio Speech-to-Text System
#[derive(Parser)]
#[command(name = "stt-rs")]
#[command(about = "Real-time speech-to-text for radio audio streams", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Start real-time transcription
    Run {
        /// Audio input device name (uses default if not specified)
        #[arg(short, long)]
        device: Option<String>,

        /// Path to Whisper model file
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (text, json, srt, vtt)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Language code (e.g., en, ru, de, fr, es, zh, ja)
        #[arg(short, long)]
        language: Option<String>,

        /// Disable console output
        #[arg(long)]
        no_console: bool,
    },

    /// List available audio input devices
    Devices,

    /// Record audio to a WAV file (for testing)
    Record {
        /// Output WAV file path
        #[arg(short, long, default_value = "recording.wav")]
        output: PathBuf,

        /// Recording duration in seconds
        #[arg(short, long, default_value = "10")]
        duration: u32,

        /// Audio input device name
        #[arg(short = 'D', long)]
        device: Option<String>,
    },

    /// Transcribe a WAV file
    Transcribe {
        /// Input WAV file path
        input: PathBuf,

        /// Path to Whisper model file
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Output format (text, json, srt, vtt)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Language code (e.g., en, ru, de, fr, es, zh, ja)
        #[arg(short, long)]
        language: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging - quiet by default, use -v for more
    let log_level = match cli.verbose {
        0 => Level::ERROR,
        1 => Level::WARN,
        2 => Level::INFO,
        3 => Level::DEBUG,
        _ => Level::TRACE,
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(log_level.into()),
        )
        .init();

    // Load configuration
    let mut config = if let Some(ref config_path) = cli.config {
        Config::from_file(config_path)
            .with_context(|| format!("Failed to load config from {}", config_path.display()))?
    } else {
        Config::default()
    };

    match cli.command {
        Commands::Run {
            device,
            model,
            output,
            format,
            language,
            no_console,
        } => {
            // Apply CLI overrides
            if let Some(device) = device {
                config.audio.device = Some(device);
            }
            if let Some(model) = model {
                config.stt.model_path = model;
            }
            if let Some(language) = language {
                config.stt.language = language;
            }
            if let Some(output) = output {
                config.output.output_path = Some(output);
            }
            config.output.format = match format.as_str() {
                "json" => stt_rs::config::OutputFormat::Json,
                "srt" => stt_rs::config::OutputFormat::Srt,
                "vtt" => stt_rs::config::OutputFormat::Vtt,
                _ => stt_rs::config::OutputFormat::Text,
            };
            config.output.enable_console = !no_console;

            run_realtime(config)
        }
        Commands::Devices => list_devices(),
        Commands::Record {
            output,
            duration,
            device,
        } => {
            if let Some(device) = device {
                config.audio.device = Some(device);
            }
            record_audio(config, output, duration)
        }
        Commands::Transcribe {
            input,
            model,
            format,
            language,
        } => {
            if let Some(model) = model {
                config.stt.model_path = model;
            }
            if let Some(language) = language {
                config.stt.language = language;
            }
            config.output.format = match format.as_str() {
                "json" => stt_rs::config::OutputFormat::Json,
                "srt" => stt_rs::config::OutputFormat::Srt,
                "vtt" => stt_rs::config::OutputFormat::Vtt,
                _ => stt_rs::config::OutputFormat::Text,
            };
            transcribe_file(config, input)
        }
    }
}

/// Run real-time transcription
fn run_realtime(config: Config) -> Result<()> {
    info!("Starting real-time transcription");

    // Setup signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        info!("Received shutdown signal");
        r.store(false, Ordering::SeqCst);
    })?;

    // Initialize audio capture
    let mut capture = AudioCapture::new(config.audio.clone())
        .context("Failed to create audio capture")?;
    capture.init().context("Failed to initialize audio capture")?;

    let actual_sample_rate = capture.actual_sample_rate();
    info!("Audio capture initialized at {} Hz", actual_sample_rate);

    // Initialize preprocessor - disable filtering, just resample
    let mut preproc_config = config.preprocessing.clone();
    preproc_config.enable_filtering = false;
    preproc_config.enable_normalization = false;
    let mut preprocessor = AudioPreprocessor::new(
        preproc_config,
        actual_sample_rate,
        16000, // Whisper expects 16kHz
    )
    .context("Failed to create audio preprocessor")?;

    // Initialize STT engine
    info!("Loading STT model from: {}", config.stt.model_path.display());
    let engine = SttEngine::new(config.stt.clone())
        .context("Failed to initialize STT engine")?;
    info!("STT engine initialized");

    // Initialize output writer
    let mut output = OutputWriter::new(config.output.clone())
        .context("Failed to create output writer")?;
    output.write_header()?;

    // Audio buffer - 5 second windows
    let window_seconds = 5.0;
    let window_samples = (16000.0 * window_seconds) as usize;
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(window_samples * 2);

    // Energy threshold for silence detection (skip silent chunks)
    let silence_threshold = 0.01;

    // Max buffer size before dropping old audio (15 seconds = 3 windows)
    let max_buffer_samples = window_samples * 3;

    // Start capture
    capture.start().context("Failed to start audio capture")?;

    let start_time = Instant::now();
    info!("Listening... Press Ctrl+C to stop");

    while running.load(Ordering::SeqCst) {
        // Drain all available audio to prevent buffer overflow
        let mut got_samples = false;
        while let Some(samples) = capture.try_receive() {
            let processed = preprocessor.process(&samples)?;
            audio_buffer.extend(&processed);
            got_samples = true;
        }

        // If no samples available, wait a bit
        if !got_samples {
            if let Some(samples) = capture.receive_timeout(Duration::from_millis(50)) {
                let processed = preprocessor.process(&samples)?;
                audio_buffer.extend(&processed);
            }
        }

        // Drop old audio if we're falling behind
        if audio_buffer.len() > max_buffer_samples {
            let drop_count = audio_buffer.len() - window_samples;
            audio_buffer.drain(..drop_count);
        }

        // Process when we have enough samples
        if audio_buffer.len() >= window_samples {
            let window: Vec<f32> = audio_buffer.drain(..window_samples).collect();

            // Calculate RMS energy to detect silence
            let energy: f32 = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();

            if energy < silence_threshold {
                // Skip silent chunk
                continue;
            }

            let offset_ms = start_time.elapsed().as_millis() as i64 - (window_seconds * 1000.0) as i64;

            if let Ok(result) = engine.transcribe(&window) {
                if !result.text.is_empty() {
                    output.write(&result, offset_ms)?;
                }
            }
        }
    }

    // Flush remaining audio
    if audio_buffer.len() >= 8000 {
        // At least 0.5s
        if let Ok(result) = engine.transcribe(&audio_buffer) {
            if !result.text.is_empty() {
                let offset_ms = start_time.elapsed().as_millis() as i64;
                output.write(&result, offset_ms)?;
            }
        }
    }

    capture.stop();
    output.flush()?;

    info!("Transcription stopped");
    Ok(())
}

/// List available audio input devices
fn list_devices() -> Result<()> {
    let capture = AudioCapture::new(stt_rs::AudioConfig::default())?;
    let devices = capture.list_devices()?;

    if devices.is_empty() {
        println!("No audio input devices found");
    } else {
        println!("Available audio input devices:");
        for (i, name) in devices.iter().enumerate() {
            println!("  {}. {}", i + 1, name);
        }
    }

    Ok(())
}

/// Record audio to a WAV file
fn record_audio(config: Config, output_path: PathBuf, duration_secs: u32) -> Result<()> {
    info!("Recording audio to: {}", output_path.display());

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    let mut capture = AudioCapture::new(config.audio.clone())?;
    capture.init()?;

    let sample_rate = capture.actual_sample_rate();
    let mut samples: Vec<f32> = Vec::new();
    let target_samples = (sample_rate * duration_secs) as usize;

    capture.start()?;

    let start = Instant::now();
    println!("Recording for {} seconds... Press Ctrl+C to stop early", duration_secs);

    while running.load(Ordering::SeqCst) && samples.len() < target_samples {
        if let Some(chunk) = capture.receive_timeout(Duration::from_millis(100)) {
            samples.extend(chunk);
        }

        // Progress indicator
        let elapsed = start.elapsed().as_secs();
        print!("\rRecording: {}s / {}s", elapsed, duration_secs);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();

    capture.stop();

    // Write WAV file
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(&output_path, spec)
        .context("Failed to create WAV file")?;

    for sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    info!("Recording saved to: {}", output_path.display());

    Ok(())
}

/// Transcribe a WAV file
fn transcribe_file(config: Config, input_path: PathBuf) -> Result<()> {
    info!("Transcribing: {}", input_path.display());

    // Read WAV file
    let mut reader = hound::WavReader::open(&input_path)
        .context("Failed to open WAV file")?;

    let spec = reader.spec();
    info!(
        "WAV format: {} channels, {} Hz, {} bits",
        spec.channels,
        spec.sample_rate,
        spec.bits_per_sample
    );

    // Read samples
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let processed_samples = if spec.sample_rate != 16000 {
        let mut preprocessor = AudioPreprocessor::new(
            config.preprocessing.clone(),
            spec.sample_rate,
            16000,
        )?;
        preprocessor.process(&mono_samples)?
    } else {
        mono_samples
    };

    info!("Loaded {} samples ({:.2}s)", processed_samples.len(), processed_samples.len() as f32 / 16000.0);

    // Initialize STT engine
    let engine = SttEngine::new(config.stt)?;

    // Transcribe
    let result = engine.transcribe(&processed_samples)?;

    // Output
    let mut output = OutputWriter::new(config.output)?;
    output.write_header()?;
    output.write(&result, 0)?;

    Ok(())
}
