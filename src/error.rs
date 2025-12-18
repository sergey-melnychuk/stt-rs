//! Custom error types for the STT-RS system

use thiserror::Error;

/// Main error type for the STT-RS system
#[derive(Error, Debug)]
pub enum SttError {
    #[error("Audio error: {0}")]
    Audio(#[from] AudioError),

    #[error("STT engine error: {0}")]
    Stt(#[from] SttEngineError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Channel error: {0}")]
    Channel(String),
}

/// Audio-related errors
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No audio input device available")]
    NoInputDevice,

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Failed to get device configuration: {0}")]
    DeviceConfig(String),

    #[error("Failed to build audio stream: {0}")]
    StreamBuild(String),

    #[error("Stream playback error: {0}")]
    StreamPlay(String),

    #[error("Unsupported sample format: {0}")]
    UnsupportedFormat(String),

    #[error("Resampling error: {0}")]
    Resampling(String),

    #[error("Buffer error: {0}")]
    Buffer(String),
}

/// STT engine errors
#[derive(Error, Debug)]
pub enum SttEngineError {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Model file not found: {0}")]
    ModelNotFound(String),

    #[error("Transcription failed: {0}")]
    Transcription(String),

    #[error("Invalid audio data for transcription")]
    InvalidAudioData,

    #[error("Whisper error: {0}")]
    Whisper(String),
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to parse configuration: {0}")]
    Parse(String),

    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    #[error("Invalid configuration value: {field} = {value}")]
    InvalidValue { field: String, value: String },

    #[error("Missing required field: {0}")]
    MissingField(String),
}

pub type Result<T> = std::result::Result<T, SttError>;
