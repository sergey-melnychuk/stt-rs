//! Radio Speech-to-Text CLI Application

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::EnvFilter;

use stt_rs::{
    AudioCapture, AudioPreprocessor, Config, OutputWriter, SpeechSegment, SpeechSegmenter,
    SttEngine,
};

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

        /// Disable audio filtering (high-pass/low-pass)
        #[arg(long)]
        no_filter: bool,

        /// Disable audio normalization
        #[arg(long)]
        no_normalize: bool,

        /// Enable noise reduction
        #[arg(long)]
        noise_reduce: bool,
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

    /// Download a Whisper model
    DownloadModel {
        /// Model size (tiny, base, small, medium, large)
        #[arg(short, long, default_value = "base")]
        size: String,

        /// Download English-only model (smaller, faster)
        #[arg(long)]
        english_only: bool,

        /// Output directory for models
        #[arg(short, long, default_value = "./models")]
        output_dir: PathBuf,
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
        .with_env_filter(EnvFilter::from_default_env().add_directive(log_level.into()))
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
            no_filter,
            no_normalize,
            noise_reduce,
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

            // Allow disabling filtering/normalization via CLI flags
            if no_filter {
                config.preprocessing.enable_filtering = false;
            }
            if no_normalize {
                config.preprocessing.enable_normalization = false;
            }
            if noise_reduce {
                config.preprocessing.enable_noise_reduction = true;
            }

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
        Commands::DownloadModel {
            size,
            english_only,
            output_dir,
        } => download_model(&size, english_only, &output_dir),
    }
}

/// Statistics for real-time transcription
struct TranscriptionStats {
    total_segments: u64,
    total_errors: u64,
    dropped_segments: u64,
    start_time: Instant,
}

impl TranscriptionStats {
    fn new() -> Self {
        Self {
            total_segments: 0,
            total_errors: 0,
            dropped_segments: 0,
            start_time: Instant::now(),
        }
    }

    fn log_summary(&self) {
        let duration = self.start_time.elapsed();
        info!(
            "Session complete: {} segments processed, {} errors, {} dropped, duration: {:.1}s",
            self.total_segments,
            self.total_errors,
            self.dropped_segments,
            duration.as_secs_f32()
        );
    }
}

/// Process a single speech segment through the STT engine
fn process_segment(
    segment: &SpeechSegment,
    engine: &SttEngine,
    output: &mut OutputWriter,
    stats: &mut TranscriptionStats,
    min_duration: f32,
) {
    let duration = segment.end - segment.start;

    // Skip segments that are too short
    if duration < min_duration {
        debug!(
            "Skipping short segment: {:.2}s < {:.2}s minimum",
            duration, min_duration
        );
        return;
    }

    stats.total_segments += 1;
    debug!(
        "Processing segment {}: {:.2}s - {:.2}s ({} samples)",
        stats.total_segments,
        segment.start,
        segment.end,
        segment.samples.len()
    );

    match engine.transcribe(&segment.samples) {
        Ok(result) => {
            if !result.text.is_empty() {
                let offset_ms = (segment.start * 1000.0) as i64;
                if let Err(e) = output.write(&result, offset_ms) {
                    error!("Failed to write output: {}", e);
                }
            }
        }
        Err(e) => {
            stats.total_errors += 1;
            warn!(
                "Transcription error on segment {} ({:.2}s): {}",
                stats.total_segments, duration, e
            );
        }
    }
}

/// Initialize audio capture with optional device reconnection
fn init_audio_capture(config: &Config) -> Result<AudioCapture> {
    let mut capture = AudioCapture::new(config.audio.clone())
        .context("Failed to create audio capture")?;
    capture.init().context("Failed to initialize audio capture")?;
    Ok(capture)
}

/// Run real-time transcription using SpeechSegmenter
fn run_realtime(config: Config) -> Result<()> {
    info!("Starting real-time transcription");

    // Setup signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        info!("Received shutdown signal");
        r.store(false, Ordering::SeqCst);
    })?;

    // Initialize STT engine first (most likely to fail if model missing)
    info!(
        "Loading STT model from: {}",
        config.stt.model_path.display()
    );
    let engine = SttEngine::new(config.stt.clone()).context("Failed to initialize STT engine")?;
    info!("STT engine initialized (language: {})", engine.language());

    // Initialize output writer
    let mut output =
        OutputWriter::new(config.output.clone()).context("Failed to create output writer")?;
    output.write_header()?;

    // Tracking stats
    let mut stats = TranscriptionStats::new();
    let min_segment_duration = config.realtime.min_segment_seconds;
    let enable_degradation = config.realtime.enable_degradation;
    let max_lag = Duration::from_secs_f32(config.realtime.max_lag_seconds);

    // Device reconnection loop
    let mut reconnect_attempts = 0;
    const MAX_RECONNECT_ATTEMPTS: u32 = 5;
    const RECONNECT_DELAY: Duration = Duration::from_secs(2);

    while running.load(Ordering::SeqCst) {
        // Try to initialize audio capture
        let capture_result = init_audio_capture(&config);
        let mut capture = match capture_result {
            Ok(c) => {
                reconnect_attempts = 0; // Reset on success
                c
            }
            Err(e) => {
                reconnect_attempts += 1;
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS {
                    return Err(e.context("Max reconnection attempts exceeded"));
                }
                warn!(
                    "Audio capture failed (attempt {}/{}): {}. Retrying in {:?}...",
                    reconnect_attempts, MAX_RECONNECT_ATTEMPTS, e, RECONNECT_DELAY
                );
                std::thread::sleep(RECONNECT_DELAY);
                continue;
            }
        };

        let actual_sample_rate = capture.actual_sample_rate();
        let target_sample_rate = config.realtime.target_sample_rate;
        info!(
            "Audio capture initialized at {} Hz (target: {} Hz)",
            actual_sample_rate, target_sample_rate
        );

        // Log preprocessing settings
        info!(
            "Preprocessing: filtering={}, normalization={}, AGC={}, noise_reduction={}, VAD={}",
            config.preprocessing.enable_filtering,
            config.preprocessing.enable_normalization,
            config.preprocessing.enable_agc,
            config.preprocessing.enable_noise_reduction,
            config.preprocessing.enable_vad
        );

        // Initialize preprocessor
        let mut preprocessor = match AudioPreprocessor::new(
            config.preprocessing.clone(),
            actual_sample_rate,
            target_sample_rate,
        ) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to create preprocessor: {}", e);
                continue;
            }
        };

        // Initialize speech segmenter
        let mut segmenter = SpeechSegmenter::new(&config.preprocessing, target_sample_rate);

        // Start capture
        if let Err(e) = capture.start() {
            error!("Failed to start capture: {}", e);
            continue;
        }

        info!("Listening... Press Ctrl+C to stop");

        let mut last_process_time = Instant::now();
        let mut device_error = false;

        // Main processing loop
        while running.load(Ordering::SeqCst) && !device_error {
            // Check for graceful degradation
            let lag = last_process_time.elapsed();
            if enable_degradation && lag > max_lag {
                warn!(
                    "Processing lag {:.2}s exceeds max {:.2}s, dropping old audio",
                    lag.as_secs_f32(),
                    max_lag.as_secs_f32()
                );
                // Drain and discard old audio
                while capture.try_receive().is_some() {
                    stats.dropped_segments += 1;
                }
                segmenter.reset();
                preprocessor.reset();
                last_process_time = Instant::now();
                continue;
            }

            // Try to receive audio
            let samples = match capture.receive_timeout(Duration::from_millis(100)) {
                Some(s) => s,
                None => {
                    // Also try non-blocking drain
                    if let Some(s) = capture.try_receive() {
                        s
                    } else {
                        continue;
                    }
                }
            };

            // Process audio through preprocessing pipeline
            let processed = match preprocessor.process(&samples) {
                Ok(p) => p,
                Err(e) => {
                    error!("Preprocessing error: {}", e);
                    device_error = true;
                    break;
                }
            };

            // Feed to segmenter and process any complete segments
            for segment in segmenter.process(&processed) {
                process_segment(&segment, &engine, &mut output, &mut stats, min_segment_duration);
            }

            last_process_time = Instant::now();
        }

        // Stop capture
        capture.stop();

        // Flush remaining segment
        if let Some(segment) = segmenter.flush() {
            process_segment(&segment, &engine, &mut output, &mut stats, min_segment_duration);
        }

        // Secure clear sensitive audio data
        preprocessor.secure_clear();

        // If we got here due to device error, try to reconnect
        if device_error && running.load(Ordering::SeqCst) {
            warn!("Device error detected, attempting reconnection...");
            std::thread::sleep(RECONNECT_DELAY);
        } else {
            // Normal shutdown
            break;
        }
    }

    output.flush()?;
    stats.log_summary();

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

    println!(
        "Recording for {} seconds... Press Ctrl+C to stop early",
        duration_secs
    );

    while running.load(Ordering::SeqCst) && samples.len() < target_samples {
        if let Some(chunk) = capture.receive_timeout(Duration::from_millis(100)) {
            samples.extend(chunk);
        }

        // Progress indicator
        let elapsed = samples.len() as f32 / sample_rate as f32;
        print!("\rRecording: {:.1}s / {}s", elapsed, duration_secs);
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

    let mut writer =
        hound::WavWriter::create(&output_path, spec).context("Failed to create WAV file")?;

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
    let mut reader = hound::WavReader::open(&input_path).context("Failed to open WAV file")?;

    let spec = reader.spec();
    info!(
        "WAV format: {} channels, {} Hz, {} bits",
        spec.channels, spec.sample_rate, spec.bits_per_sample
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

    // Resample to target rate if needed
    let target_sample_rate = config.realtime.target_sample_rate;
    let processed_samples = if spec.sample_rate != target_sample_rate {
        let mut preprocessor = AudioPreprocessor::new(
            config.preprocessing.clone(),
            spec.sample_rate,
            target_sample_rate,
        )?;
        preprocessor.process(&mono_samples)?
    } else {
        mono_samples
    };

    info!(
        "Loaded {} samples ({:.2}s)",
        processed_samples.len(),
        processed_samples.len() as f32 / target_sample_rate as f32
    );

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

/// Download a Whisper model from Hugging Face
fn download_model(size: &str, english_only: bool, output_dir: &PathBuf) -> Result<()> {
    // Validate model size
    let valid_sizes = ["tiny", "base", "small", "medium", "large"];
    if !valid_sizes.contains(&size) {
        anyhow::bail!(
            "Invalid model size '{}'. Valid sizes: {}",
            size,
            valid_sizes.join(", ")
        );
    }

    // Construct filename and URL
    let suffix = if english_only { ".en" } else { "" };
    let filename = format!("ggml-{}{}.bin", size, suffix);
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        filename
    );

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create directory: {}", output_dir.display()))?;

    let output_path = output_dir.join(&filename);

    // Check if already exists
    if output_path.exists() {
        println!("Model already exists: {}", output_path.display());
        println!("Delete it first if you want to re-download.");
        return Ok(());
    }

    println!("Downloading {} model...", size);
    println!("URL: {}", url);
    println!("Destination: {}", output_path.display());
    println!();

    // Convert path to string, handling non-UTF8 gracefully
    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Output path contains invalid UTF-8 characters"))?;

    // Use curl for download with progress
    let status = std::process::Command::new("curl")
        .args(["-L", "--progress-bar", "-o", output_path_str, &url])
        .status()
        .context("Failed to execute curl. Make sure curl is installed.")?;

    if !status.success() {
        anyhow::bail!("Download failed with exit code: {:?}", status.code());
    }

    // Verify file exists and has reasonable size
    let metadata = std::fs::metadata(&output_path)
        .with_context(|| format!("Failed to read downloaded file: {}", output_path.display()))?;

    let size_mb = metadata.len() as f64 / 1_000_000.0;
    if size_mb < 10.0 {
        std::fs::remove_file(&output_path)?;
        anyhow::bail!(
            "Downloaded file is too small ({:.1} MB). Download may have failed.",
            size_mb
        );
    }

    println!();
    println!("Download complete: {:.1} MB", size_mb);
    println!();
    println!("To use this model:");
    println!("  stt-rs run -m {}", output_path.display());

    Ok(())
}
