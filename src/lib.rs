//! Radio Speech-to-Text System
//!
//! A Rust-based system for capturing audio from radio streams and performing
//! real-time speech-to-text transcription using Whisper.
//!
//! # Architecture
//!
//! The system is organized into the following modules:
//!
//! - `audio`: Audio capture, preprocessing, VAD, and buffer management
//! - `stt`: Speech-to-text engine integration (Whisper)
//! - `output`: Output formatting and writing
//! - `config`: Configuration structures
//! - `error`: Error types
//!
//! # Example
//!
//! ```no_run
//! use stt_rs::{Config, AudioCapture, AudioPreprocessor, SttEngine};
//!
//! // Load configuration
//! let config = Config::default();
//!
//! // Initialize audio capture
//! let mut capture = AudioCapture::new(config.audio.clone()).unwrap();
//! capture.init().unwrap();
//!
//! // Initialize preprocessor
//! let preprocessor = AudioPreprocessor::new(
//!     config.preprocessing.clone(),
//!     capture.actual_sample_rate(),
//!     16000,
//! ).unwrap();
//!
//! // Initialize STT engine
//! let engine = SttEngine::new(config.stt.clone()).unwrap();
//! ```

pub mod audio;
pub mod config;
pub mod error;
pub mod output;
pub mod stt;

// Re-exports for convenience
pub use audio::{AudioBuffer, AudioCapture, AudioPreprocessor, VoiceActivityDetector};
pub use config::{AudioConfig, Config, OutputConfig, PreprocessingConfig, SttConfig};
pub use error::{AudioError, ConfigError, Result, SttEngineError, SttError};
pub use output::OutputWriter;
pub use stt::{SttEngine, TranscriptionResult};
