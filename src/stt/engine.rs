//! Whisper-based STT engine

use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::SttConfig;
use crate::error::{Result, SttEngineError};

/// Transcription result with timing information
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Start time in milliseconds
    pub start_ms: i64,
    /// End time in milliseconds
    pub end_ms: i64,
    /// Individual segments
    pub segments: Vec<TranscriptionSegment>,
}

/// Individual transcription segment
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,
    /// Start time in milliseconds
    pub start_ms: i64,
    /// End time in milliseconds
    pub end_ms: i64,
}

/// Whisper-based Speech-to-Text engine
pub struct SttEngine {
    ctx: Arc<WhisperContext>,
    config: SttConfig,
}

impl SttEngine {
    /// Create a new STT engine with the given configuration
    pub fn new(config: SttConfig) -> Result<Self> {
        let model_path = &config.model_path;

        if !model_path.exists() {
            return Err(SttEngineError::ModelNotFound(
                model_path.display().to_string(),
            )
            .into());
        }

        info!("Loading Whisper model from: {}", model_path.display());

        let ctx_params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().unwrap_or_default(),
            ctx_params,
        )
        .map_err(|e| SttEngineError::ModelLoad(e.to_string()))?;

        info!("Whisper model loaded successfully");

        Ok(Self {
            ctx: Arc::new(ctx),
            config,
        })
    }

    /// Load a model from a specific path
    pub fn from_model_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = SttConfig {
            model_path: path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Transcribe audio samples
    ///
    /// Audio must be 16kHz mono f32 samples
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscriptionResult> {
        if samples.is_empty() {
            return Err(SttEngineError::InvalidAudioData.into());
        }

        debug!("Transcribing {} samples ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure parameters
        params.set_n_threads(self.config.threads as i32);
        params.set_language(Some(&self.config.language));
        params.set_translate(self.config.translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);
        params.set_no_context(true);

        // Create state and run inference
        let mut state = self.ctx.create_state()
            .map_err(|e| SttEngineError::Whisper(e.to_string()))?;

        state.full(params, samples)
            .map_err(|e| SttEngineError::Transcription(e.to_string()))?;

        // Extract results
        let num_segments = state.full_n_segments()
            .map_err(|e| SttEngineError::Transcription(e.to_string()))?;

        let mut segments = Vec::with_capacity(num_segments as usize);
        let mut full_text = String::new();
        let mut overall_start = i64::MAX;
        let mut overall_end = i64::MIN;

        for i in 0..num_segments {
            let text = state.full_get_segment_text(i)
                .map_err(|e| SttEngineError::Transcription(e.to_string()))?;

            let start = state.full_get_segment_t0(i)
                .map_err(|e| SttEngineError::Transcription(e.to_string()))? as i64 * 10; // Convert to ms

            let end = state.full_get_segment_t1(i)
                .map_err(|e| SttEngineError::Transcription(e.to_string()))? as i64 * 10;

            overall_start = overall_start.min(start);
            overall_end = overall_end.max(end);

            if !full_text.is_empty() && !text.starts_with(' ') {
                full_text.push(' ');
            }
            full_text.push_str(text.trim());

            segments.push(TranscriptionSegment {
                text: text.trim().to_string(),
                start_ms: start,
                end_ms: end,
            });
        }

        debug!("Transcription complete: {} segments, {} chars", segments.len(), full_text.len());

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            start_ms: if overall_start == i64::MAX { 0 } else { overall_start },
            end_ms: if overall_end == i64::MIN { 0 } else { overall_end },
            segments,
        })
    }

    /// Get the language configured for transcription
    pub fn language(&self) -> &str {
        &self.config.language
    }

    /// Check if translation is enabled
    pub fn is_translating(&self) -> bool {
        self.config.translate
    }
}

// Safety: WhisperContext is thread-safe for inference
unsafe impl Send for SttEngine {}
unsafe impl Sync for SttEngine {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_result() {
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            start_ms: 0,
            end_ms: 1000,
            segments: vec![TranscriptionSegment {
                text: "Hello world".to_string(),
                start_ms: 0,
                end_ms: 1000,
            }],
        };

        assert_eq!(result.text, "Hello world");
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn test_engine_missing_model() {
        let config = SttConfig {
            model_path: "/nonexistent/model.bin".into(),
            ..Default::default()
        };

        let result = SttEngine::new(config);
        assert!(result.is_err());
    }
}
