//! Configuration structures for the STT-RS system

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub preprocessing: PreprocessingConfig,
    pub stt: SttConfig,
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            stt: SttConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file(path: &PathBuf) -> Result<Self, crate::error::ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|_| {
            crate::error::ConfigError::FileNotFound(path.display().to_string())
        })?;

        toml::from_str(&content)
            .map_err(|e| crate::error::ConfigError::Parse(e.to_string()))
    }
}

/// Audio capture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Target sample rate (Hz)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Buffer size in samples
    pub buffer_size: u32,
    /// Audio device name (None = default device)
    pub device: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 512,
            device: None,
        }
    }
}

/// Audio preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessingConfig {
    /// Enable resampling to target sample rate
    pub enable_resampling: bool,
    /// Enable filtering
    pub enable_filtering: bool,
    /// High-pass filter cutoff frequency (Hz)
    pub high_pass_cutoff: f32,
    /// Low-pass filter cutoff frequency (Hz)
    pub low_pass_cutoff: f32,
    /// Enable audio normalization
    pub enable_normalization: bool,
    /// Enable Voice Activity Detection
    pub enable_vad: bool,
    /// VAD energy threshold (0.0 - 1.0)
    pub vad_threshold: f32,
    /// VAD minimum speech duration (seconds)
    pub vad_min_speech_duration: f32,
    /// VAD minimum silence duration (seconds)
    pub vad_min_silence_duration: f32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_resampling: true,
            enable_filtering: true,
            high_pass_cutoff: 300.0,
            low_pass_cutoff: 3400.0,
            enable_normalization: true,
            enable_vad: true,
            vad_threshold: 0.05,
            vad_min_speech_duration: 0.25,
            vad_min_silence_duration: 0.5,
        }
    }
}

/// STT engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    /// Path to the Whisper model file
    pub model_path: PathBuf,
    /// Model size identifier
    pub model_size: ModelSize,
    /// Language for transcription
    pub language: String,
    /// Number of threads for inference
    pub threads: u32,
    /// Enable translation to English
    pub translate: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models/ggml-base.en.bin"),
            model_size: ModelSize::Base,
            language: "en".to_string(),
            threads: 4,
            translate: false,
        }
    }
}

/// Whisper model sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSize::Tiny => write!(f, "tiny"),
            ModelSize::Base => write!(f, "base"),
            ModelSize::Small => write!(f, "small"),
            ModelSize::Medium => write!(f, "medium"),
            ModelSize::Large => write!(f, "large"),
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,
    /// Output file path (None = console only)
    pub output_path: Option<PathBuf>,
    /// Include timestamps in output
    pub enable_timestamps: bool,
    /// Enable console output
    pub enable_console: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Text,
            output_path: None,
            enable_timestamps: true,
            enable_console: true,
        }
    }
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Plain text
    Text,
    /// JSON with metadata
    Json,
    /// SRT subtitle format
    Srt,
    /// VTT subtitle format
    Vtt,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Text => write!(f, "text"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Srt => write!(f, "srt"),
            OutputFormat::Vtt => write!(f, "vtt"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.channels, 1);
        assert!(config.preprocessing.enable_vad);
        assert_eq!(config.stt.language, "en");
    }

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
            [audio]
            sample_rate = 44100
            channels = 2

            [stt]
            language = "de"
            threads = 8
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.audio.sample_rate, 44100);
        assert_eq!(config.audio.channels, 2);
        assert_eq!(config.stt.language, "de");
        assert_eq!(config.stt.threads, 8);
    }
}
