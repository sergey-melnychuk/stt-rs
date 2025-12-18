//! Radio Speech-to-Text CLI Application

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{debug, info, warn, Level};
use tracing_subscriber::EnvFilter;

use stt_rs::{AudioCapture, AudioPreprocessor, Config, OutputWriter, SttEngine};
use stt_rs::audio::vad::SpeechSegmenter;

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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.verbose {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
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
            no_console,
        } => {
            // Apply CLI overrides
            if let Some(device) = device {
                config.audio.device = Some(device);
            }
            if let Some(model) = model {
                config.stt.model_path = model;
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
        } => {
            if let Some(model) = model {
                config.stt.model_path = model;
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

    // Initialize preprocessor
    let mut preprocessor = AudioPreprocessor::new(
        config.preprocessing.clone(),
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

    // Initialize speech segmenter
    let mut segmenter = SpeechSegmenter::new(&config.preprocessing, 16000);

    // Audio buffer for accumulating samples
    let mut audio_buffer: Vec<f32> = Vec::new();
    let window_duration = 5.0; // 5 second windows
    let window_samples = (16000.0 * window_duration) as usize;

    // Start capture
    capture.start().context("Failed to start audio capture")?;

    let start_time = Instant::now();
    let mut last_transcription = Instant::now();

    info!("Listening... Press Ctrl+C to stop");

    while running.load(Ordering::SeqCst) {
        // Get audio samples
        if let Some(samples) = capture.receive_timeout(Duration::from_millis(100)) {
            // Preprocess audio
            let processed = preprocessor.process(&samples)?;

            // Process through speech segmenter
            let segments = segmenter.process(&processed);

            for segment in segments {
                if segment.samples.len() >= 16000 / 4 {
                    // At least 250ms of speech
                    debug!(
                        "Speech segment: {:.2}s - {:.2}s ({} samples)",
                        segment.start,
                        segment.end,
                        segment.samples.len()
                    );

                    // Transcribe segment
                    match engine.transcribe(&segment.samples) {
                        Ok(result) => {
                            if !result.text.is_empty() {
                                let offset_ms = (segment.start * 1000.0) as i64;
                                output.write(&result, offset_ms)?;
                            }
                        }
                        Err(e) => {
                            warn!("Transcription error: {}", e);
                        }
                    }
                }
            }

            // Also accumulate for window-based processing
            audio_buffer.extend(&processed);

            // Process accumulated buffer if enough samples
            if audio_buffer.len() >= window_samples
                && last_transcription.elapsed() >= Duration::from_secs(3)
            {
                let window: Vec<f32> = audio_buffer.drain(..window_samples).collect();

                match engine.transcribe(&window) {
                    Ok(result) => {
                        if !result.text.is_empty() {
                            let _offset_ms = start_time.elapsed().as_millis() as i64
                                - (window_duration * 1000.0) as i64;
                            // Only output if VAD-based segments didn't catch it
                            debug!("Window transcription: {}", result.text);
                        }
                    }
                    Err(e) => {
                        debug!("Window transcription error: {}", e);
                    }
                }

                last_transcription = Instant::now();

                // Keep overlap
                let overlap = window_samples / 2;
                if audio_buffer.len() > overlap {
                    audio_buffer.drain(0..(audio_buffer.len() - overlap));
                }
            }
        }
    }

    // Flush remaining audio
    if let Some(segment) = segmenter.flush() {
        if segment.samples.len() >= 16000 / 4 {
            if let Ok(result) = engine.transcribe(&segment.samples) {
                if !result.text.is_empty() {
                    let offset_ms = (segment.start * 1000.0) as i64;
                    output.write(&result, offset_ms)?;
                }
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
