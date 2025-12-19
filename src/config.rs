//! Configuration structures for the STT-RS system

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub preprocessing: PreprocessingConfig,
    pub realtime: RealtimeConfig,
    pub stt: SttConfig,
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            realtime: RealtimeConfig::default(),
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
    /// Enable noise reduction (spectral subtraction)
    pub enable_noise_reduction: bool,
    /// Noise reduction strength (0.0 - 1.0, higher = more aggressive)
    pub noise_reduction_strength: f32,
    /// Enable automatic gain control
    pub enable_agc: bool,
    /// AGC target level (0.0 - 1.0)
    pub agc_target_level: f32,
    /// AGC attack time constant (seconds)
    pub agc_attack_time: f32,
    /// AGC release time constant (seconds)
    pub agc_release_time: f32,
    /// Enable Voice Activity Detection
    pub enable_vad: bool,
    /// VAD energy threshold (0.0 - 1.0)
    pub vad_threshold: f32,
    /// VAD minimum speech duration (seconds)
    pub vad_min_speech_duration: f32,
    /// VAD minimum silence duration (seconds)
    pub vad_min_silence_duration: f32,
    /// Pre-roll duration before speech (seconds)
    pub vad_pre_roll: f32,
    /// Maximum segment duration before forced split (seconds)
    pub vad_max_segment_duration: f32,
}

/// Real-time processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RealtimeConfig {
    /// Target sample rate for Whisper (Hz)
    pub target_sample_rate: u32,
    /// Enable graceful degradation when falling behind (drop old audio)
    pub enable_degradation: bool,
    /// Maximum processing lag before dropping segments (seconds)
    pub max_lag_seconds: f32,
    /// Minimum segment duration to process (seconds) - shorter segments are skipped
    pub min_segment_seconds: f32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_resampling: true,
            enable_filtering: true,
            high_pass_cutoff: 300.0,
            low_pass_cutoff: 3400.0,
            enable_normalization: true,
            enable_noise_reduction: false,  // Off by default, can be noisy
            noise_reduction_strength: 0.5,
            enable_agc: true,
            agc_target_level: 0.5,
            agc_attack_time: 0.01,   // 10ms attack
            agc_release_time: 0.1,   // 100ms release
            enable_vad: true,
            vad_threshold: 0.01,  // Lower threshold for better sensitivity
            vad_min_speech_duration: 0.25,
            vad_min_silence_duration: 0.3,  // Faster segment completion
            vad_pre_roll: 0.2,  // 200ms pre-roll
            vad_max_segment_duration: 10.0,  // 10 second max segment
        }
    }
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000,
            enable_degradation: true,
            max_lag_seconds: 5.0,
            min_segment_seconds: 0.5,
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
            threads: 8,
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
        assert_eq!(config.preprocessing.vad_pre_roll, 0.2);
        assert_eq!(config.preprocessing.vad_max_segment_duration, 10.0);
        assert_eq!(config.realtime.target_sample_rate, 16000);
        assert!(config.realtime.enable_degradation);
        assert_eq!(config.realtime.max_lag_seconds, 5.0);
        assert_eq!(config.realtime.min_segment_seconds, 0.5);
        assert_eq!(config.stt.language, "en");
    }

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
            [audio]
            sample_rate = 44100
            channels = 2

            [preprocessing]
            vad_pre_roll = 0.3
            vad_max_segment_duration = 15.0

            [realtime]
            target_sample_rate = 16000
            enable_degradation = false
            max_lag_seconds = 10.0
            min_segment_seconds = 1.0

            [stt]
            language = "de"
            threads = 8
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.audio.sample_rate, 44100);
        assert_eq!(config.audio.channels, 2);
        assert_eq!(config.preprocessing.vad_pre_roll, 0.3);
        assert_eq!(config.preprocessing.vad_max_segment_duration, 15.0);
        assert!(!config.realtime.enable_degradation);
        assert_eq!(config.realtime.max_lag_seconds, 10.0);
        assert_eq!(config.realtime.min_segment_seconds, 1.0);
        assert_eq!(config.stt.language, "de");
        assert_eq!(config.stt.threads, 8);
    }
}
