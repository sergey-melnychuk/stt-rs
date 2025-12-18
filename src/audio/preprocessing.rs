//! Audio preprocessing module - resampling, filtering, normalization

use biquad::{Biquad, Coefficients, DirectForm1, ToHertz, Type, Q_BUTTERWORTH_F32};
use rubato::{FftFixedIn, Resampler};
use tracing::debug;

use crate::config::PreprocessingConfig;
use crate::error::{AudioError, Result};

/// Audio preprocessor for preparing audio for STT
pub struct AudioPreprocessor {
    config: PreprocessingConfig,
    resampler: Option<FftFixedIn<f32>>,
    high_pass_filter: Option<DirectForm1<f32>>,
    low_pass_filter: Option<DirectForm1<f32>>,
    source_sample_rate: u32,
    target_sample_rate: u32,
}

impl AudioPreprocessor {
    /// Create a new audio preprocessor
    pub fn new(
        config: PreprocessingConfig,
        source_sample_rate: u32,
        target_sample_rate: u32,
    ) -> Result<Self> {
        let mut preprocessor = Self {
            config,
            resampler: None,
            high_pass_filter: None,
            low_pass_filter: None,
            source_sample_rate,
            target_sample_rate,
        };

        preprocessor.init()?;
        Ok(preprocessor)
    }

    fn init(&mut self) -> Result<()> {
        // Initialize resampler if needed
        if self.config.enable_resampling && self.source_sample_rate != self.target_sample_rate {
            debug!(
                "Initializing resampler: {} Hz -> {} Hz",
                self.source_sample_rate, self.target_sample_rate
            );

            let resampler = FftFixedIn::<f32>::new(
                self.source_sample_rate as usize,
                self.target_sample_rate as usize,
                1024, // chunk size
                1,    // sub-chunks
                1,    // channels
            )
            .map_err(|e| AudioError::Resampling(e.to_string()))?;

            self.resampler = Some(resampler);
        }

        // Initialize filters
        if self.config.enable_filtering {
            // High-pass filter to remove DC offset and low rumble
            if self.config.high_pass_cutoff > 0.0 {
                let coeffs = Coefficients::<f32>::from_params(
                    Type::HighPass,
                    self.target_sample_rate.hz(),
                    self.config.high_pass_cutoff.hz(),
                    Q_BUTTERWORTH_F32,
                )
                .map_err(|e| AudioError::Buffer(format!("High-pass filter error: {:?}", e)))?;

                self.high_pass_filter = Some(DirectForm1::<f32>::new(coeffs));
                debug!("High-pass filter: {} Hz", self.config.high_pass_cutoff);
            }

            // Low-pass filter for radio voice band
            if self.config.low_pass_cutoff > 0.0 {
                let coeffs = Coefficients::<f32>::from_params(
                    Type::LowPass,
                    self.target_sample_rate.hz(),
                    self.config.low_pass_cutoff.hz(),
                    Q_BUTTERWORTH_F32,
                )
                .map_err(|e| AudioError::Buffer(format!("Low-pass filter error: {:?}", e)))?;

                self.low_pass_filter = Some(DirectForm1::<f32>::new(coeffs));
                debug!("Low-pass filter: {} Hz", self.config.low_pass_cutoff);
            }
        }

        Ok(())
    }

    /// Process a chunk of audio samples
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut output = samples.to_vec();

        // Resample if needed
        if self.resampler.is_some() {
            output = self.do_resample(&output)?;
        }

        // Apply high-pass filter
        if let Some(ref mut filter) = self.high_pass_filter {
            for sample in output.iter_mut() {
                *sample = filter.run(*sample);
            }
        }

        // Apply low-pass filter
        if let Some(ref mut filter) = self.low_pass_filter {
            for sample in output.iter_mut() {
                *sample = filter.run(*sample);
            }
        }

        // Normalize
        if self.config.enable_normalization {
            output = self.normalize(&output);
        }

        Ok(output)
    }

    fn do_resample(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let resampler = self.resampler.as_mut().unwrap();
        let input_frames_needed = resampler.input_frames_next();

        if samples.len() < input_frames_needed {
            // Pad with zeros if not enough samples
            let mut padded = samples.to_vec();
            padded.resize(input_frames_needed, 0.0);

            let result = resampler
                .process(&[padded], None)
                .map_err(|e| AudioError::Resampling(e.to_string()))?;

            Ok(result.into_iter().next().unwrap_or_default())
        } else {
            // Process in chunks
            let mut output = Vec::new();
            let mut remaining = samples;

            while remaining.len() >= input_frames_needed {
                let (chunk, rest) = remaining.split_at(input_frames_needed);
                remaining = rest;

                let result = resampler
                    .process(&[chunk.to_vec()], None)
                    .map_err(|e| AudioError::Resampling(e.to_string()))?;

                if let Some(resampled) = result.into_iter().next() {
                    output.extend(resampled);
                }
            }

            // Handle remaining samples
            if !remaining.is_empty() {
                let mut padded = remaining.to_vec();
                padded.resize(input_frames_needed, 0.0);

                let result = resampler
                    .process(&[padded], None)
                    .map_err(|e| AudioError::Resampling(e.to_string()))?;

                if let Some(resampled) = result.into_iter().next() {
                    // Only take proportional amount of output
                    let ratio = remaining.len() as f32 / input_frames_needed as f32;
                    let take = (resampled.len() as f32 * ratio) as usize;
                    output.extend(&resampled[..take.min(resampled.len())]);
                }
            }

            Ok(output)
        }
    }

    fn normalize(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Find peak amplitude
        let peak = samples
            .iter()
            .map(|s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        if peak < 1e-6 {
            // Audio is essentially silent
            return samples.to_vec();
        }

        // Normalize to peak = 0.95 (leave headroom)
        let scale = 0.95 / peak;
        samples.iter().map(|s| s * scale).collect()
    }

    /// Get the target sample rate
    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    /// Reset filter states
    pub fn reset(&mut self) {
        if let Some(ref mut filter) = self.high_pass_filter {
            filter.reset_state();
        }
        if let Some(ref mut filter) = self.low_pass_filter {
            filter.reset_state();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_creation() {
        let config = PreprocessingConfig::default();
        let preprocessor = AudioPreprocessor::new(config, 44100, 16000);
        assert!(preprocessor.is_ok());
    }

    #[test]
    fn test_normalization() {
        let config = PreprocessingConfig {
            enable_resampling: false,
            enable_filtering: false,
            enable_normalization: true,
            ..Default::default()
        };
        let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000).unwrap();

        let samples = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let processed = preprocessor.process(&samples).unwrap();

        // Peak should be close to 0.95
        let peak = processed.iter().map(|s| s.abs()).fold(0.0f32, |a, b| a.max(b));
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_filtering() {
        let config = PreprocessingConfig {
            enable_resampling: false,
            enable_filtering: true,
            enable_normalization: false,
            high_pass_cutoff: 300.0,
            low_pass_cutoff: 3400.0,
            ..Default::default()
        };
        let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000).unwrap();

        // Generate a simple sine wave
        let samples: Vec<f32> = (0..160)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 16000.0).sin())
            .collect();

        let processed = preprocessor.process(&samples);
        assert!(processed.is_ok());
        assert!(!processed.unwrap().is_empty());
    }
}
