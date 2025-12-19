//! Audio preprocessing module - resampling, filtering, normalization, noise reduction, AGC

use biquad::{Biquad, Coefficients, DirectForm1, ToHertz, Type, Q_BUTTERWORTH_F32};
use rubato::{FftFixedIn, Resampler};
use tracing::debug;

use crate::config::PreprocessingConfig;
use crate::error::{AudioError, Result};

/// Automatic Gain Control state
struct AgcState {
    /// Current gain level
    gain: f32,
    /// Target RMS level
    target_level: f32,
    /// Attack coefficient (for increasing gain)
    attack_coeff: f32,
    /// Release coefficient (for decreasing gain)
    release_coeff: f32,
    /// Minimum gain to prevent silence amplification
    min_gain: f32,
    /// Maximum gain to prevent clipping
    max_gain: f32,
}

impl AgcState {
    fn new(target_level: f32, attack_time: f32, release_time: f32, sample_rate: u32) -> Self {
        // Convert time constants to coefficients
        // coefficient = 1 - exp(-1 / (time * sample_rate))
        let attack_coeff = 1.0 - (-1.0 / (attack_time * sample_rate as f32)).exp();
        let release_coeff = 1.0 - (-1.0 / (release_time * sample_rate as f32)).exp();

        Self {
            gain: 1.0,
            target_level,
            attack_coeff,
            release_coeff,
            min_gain: 0.1,
            max_gain: 10.0,
        }
    }

    fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            // Calculate envelope (absolute value)
            let envelope = sample.abs();

            // Calculate desired gain
            let desired_gain = if envelope > 1e-6 {
                self.target_level / envelope
            } else {
                self.gain // Keep current gain for silence
            };

            // Clamp desired gain
            let desired_gain = desired_gain.clamp(self.min_gain, self.max_gain);

            // Smooth gain changes (attack/release)
            let coeff = if desired_gain < self.gain {
                self.attack_coeff // Reducing gain (attack)
            } else {
                self.release_coeff // Increasing gain (release)
            };

            self.gain += coeff * (desired_gain - self.gain);

            // Apply gain
            *sample *= self.gain;

            // Soft clipping to prevent harsh distortion
            *sample = soft_clip(*sample);
        }
    }

    fn reset(&mut self) {
        self.gain = 1.0;
    }
}

/// Soft clipping function to prevent harsh clipping
fn soft_clip(x: f32) -> f32 {
    if x.abs() <= 0.5 {
        x
    } else if x > 0.0 {
        0.5 + (1.0 - (-2.0 * (x - 0.5)).exp()) * 0.5
    } else {
        -0.5 - (1.0 - (-2.0 * (-x - 0.5)).exp()) * 0.5
    }
}

/// Simple noise reduction using spectral subtraction approximation
/// This is a time-domain approximation that works on short frames
struct NoiseReducer {
    /// Noise floor estimate
    noise_floor: f32,
    /// Smoothing coefficient for noise estimation
    noise_alpha: f32,
    /// Reduction strength (0.0 - 1.0)
    strength: f32,
    /// Minimum signal to preserve (prevents musical noise)
    min_signal: f32,
}

impl NoiseReducer {
    fn new(strength: f32) -> Self {
        Self {
            noise_floor: 0.0,
            noise_alpha: 0.001, // Very slow adaptation for noise floor
            strength: strength.clamp(0.0, 1.0),
            min_signal: 0.01,
        }
    }

    fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }

        // Calculate RMS energy of the frame
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Update noise floor estimate (only when signal is low)
        if rms < self.noise_floor * 2.0 || self.noise_floor < 1e-6 {
            self.noise_floor = self.noise_alpha * rms + (1.0 - self.noise_alpha) * self.noise_floor;
        }

        // Calculate noise reduction factor
        let noise_threshold = self.noise_floor * (1.0 + self.strength * 3.0);

        // Apply spectral subtraction approximation
        for sample in samples.iter_mut() {
            let abs_sample = sample.abs();
            if abs_sample < noise_threshold {
                // Reduce samples below threshold
                let reduction = 1.0 - self.strength * (1.0 - abs_sample / noise_threshold);
                *sample *= reduction.max(self.min_signal);
            }
        }
    }

    fn reset(&mut self) {
        self.noise_floor = 0.0;
    }
}

/// Audio preprocessor for preparing audio for STT
pub struct AudioPreprocessor {
    config: PreprocessingConfig,
    resampler: Option<FftFixedIn<f32>>,
    high_pass_filter: Option<DirectForm1<f32>>,
    low_pass_filter: Option<DirectForm1<f32>>,
    agc: Option<AgcState>,
    noise_reducer: Option<NoiseReducer>,
    source_sample_rate: u32,
    target_sample_rate: u32,
    resample_buffer: Vec<f32>,
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
            agc: None,
            noise_reducer: None,
            source_sample_rate,
            target_sample_rate,
            resample_buffer: Vec::new(),
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

        // Initialize AGC
        if self.config.enable_agc {
            self.agc = Some(AgcState::new(
                self.config.agc_target_level,
                self.config.agc_attack_time,
                self.config.agc_release_time,
                self.target_sample_rate,
            ));
            debug!(
                "AGC enabled: target={}, attack={}s, release={}s",
                self.config.agc_target_level,
                self.config.agc_attack_time,
                self.config.agc_release_time
            );
        }

        // Initialize noise reducer
        if self.config.enable_noise_reduction {
            self.noise_reducer = Some(NoiseReducer::new(self.config.noise_reduction_strength));
            debug!(
                "Noise reduction enabled: strength={}",
                self.config.noise_reduction_strength
            );
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

        // Apply noise reduction (before AGC)
        if let Some(ref mut reducer) = self.noise_reducer {
            reducer.process(&mut output);
        }

        // Apply AGC
        if let Some(ref mut agc) = self.agc {
            agc.process(&mut output);
        }

        // Normalize (after AGC, if both enabled AGC takes precedence)
        if self.config.enable_normalization && self.agc.is_none() {
            output = self.normalize(&output);
        }

        Ok(output)
    }

    fn do_resample(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        // Safe: we only call this when resampler.is_some()
        let resampler = self.resampler.as_mut().ok_or_else(|| {
            AudioError::Resampling("Resampler not initialized".to_string())
        })?;

        // Add new samples to buffer
        self.resample_buffer.extend_from_slice(samples);

        let input_frames_needed = resampler.input_frames_next();
        let mut output = Vec::new();

        // Process complete chunks only - no zero padding
        while self.resample_buffer.len() >= input_frames_needed {
            let chunk: Vec<f32> = self.resample_buffer.drain(..input_frames_needed).collect();

            let result = resampler
                .process(&[chunk], None)
                .map_err(|e| AudioError::Resampling(e.to_string()))?;

            if let Some(resampled) = result.into_iter().next() {
                output.extend(resampled);
            }
        }

        // Keep remaining samples in buffer for next call
        Ok(output)
    }

    fn normalize(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Find peak amplitude
        let peak = samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));

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

    /// Reset filter states and buffers
    pub fn reset(&mut self) {
        if let Some(ref mut filter) = self.high_pass_filter {
            filter.reset_state();
        }
        if let Some(ref mut filter) = self.low_pass_filter {
            filter.reset_state();
        }
        if let Some(ref mut agc) = self.agc {
            agc.reset();
        }
        if let Some(ref mut reducer) = self.noise_reducer {
            reducer.reset();
        }
        self.resample_buffer.clear();
    }

    /// Securely clear all internal buffers (for sensitive audio)
    pub fn secure_clear(&mut self) {
        // Zero out resample buffer
        for sample in self.resample_buffer.iter_mut() {
            *sample = 0.0;
        }
        self.resample_buffer.clear();

        // Reset all state
        self.reset();
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
            enable_agc: false,
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
            enable_agc: false,
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

    #[test]
    fn test_agc() {
        let config = PreprocessingConfig {
            enable_resampling: false,
            enable_filtering: false,
            enable_normalization: false,
            enable_agc: true,
            agc_target_level: 0.5,
            agc_attack_time: 0.001,
            agc_release_time: 0.01,
            ..Default::default()
        };
        let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000).unwrap();

        // Process quiet audio
        let quiet: Vec<f32> = (0..1600)
            .map(|i| 0.05 * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16000.0).sin())
            .collect();

        let processed = preprocessor.process(&quiet).unwrap();

        // AGC should have increased the level
        let input_rms = (quiet.iter().map(|s| s * s).sum::<f32>() / quiet.len() as f32).sqrt();
        let output_rms =
            (processed.iter().map(|s| s * s).sum::<f32>() / processed.len() as f32).sqrt();

        assert!(output_rms > input_rms, "AGC should amplify quiet audio");
    }

    #[test]
    fn test_noise_reduction() {
        let config = PreprocessingConfig {
            enable_resampling: false,
            enable_filtering: false,
            enable_normalization: false,
            enable_agc: false,
            enable_noise_reduction: true,
            noise_reduction_strength: 0.8,
            ..Default::default()
        };
        let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000).unwrap();

        // Feed some noise to establish noise floor
        let noise: Vec<f32> = (0..1600).map(|_| 0.01 * (rand_simple() - 0.5)).collect();
        let _ = preprocessor.process(&noise);

        // Now process slightly louder signal
        let signal: Vec<f32> = (0..1600).map(|_| 0.02 * (rand_simple() - 0.5)).collect();
        let processed = preprocessor.process(&signal).unwrap();

        // Noise reduction should reduce the amplitude
        let input_rms = (signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32).sqrt();
        let output_rms =
            (processed.iter().map(|s| s * s).sum::<f32>() / processed.len() as f32).sqrt();

        assert!(
            output_rms <= input_rms,
            "Noise reduction should reduce or maintain amplitude"
        );
    }

    #[test]
    fn test_soft_clip() {
        // Values below 0.5 should pass through
        assert!((soft_clip(0.3) - 0.3).abs() < 0.001);
        assert!((soft_clip(-0.3) - (-0.3)).abs() < 0.001);

        // Values above should be soft-clipped
        assert!(soft_clip(1.0) < 1.0);
        assert!(soft_clip(2.0) < 1.0);
        assert!(soft_clip(-1.0) > -1.0);
    }

    #[test]
    fn test_secure_clear() {
        let config = PreprocessingConfig::default();
        let mut preprocessor = AudioPreprocessor::new(config, 44100, 16000).unwrap();

        // Add some samples
        let samples: Vec<f32> = vec![0.5; 1000];
        let _ = preprocessor.process(&samples);

        // Secure clear
        preprocessor.secure_clear();

        // Buffer should be empty
        assert!(preprocessor.resample_buffer.is_empty());
    }

    // Simple deterministic pseudo-random for testing
    fn rand_simple() -> f32 {
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u32> = const { Cell::new(12345) };
        }
        SEED.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s.set(x);
            (x as f32) / (u32::MAX as f32)
        })
    }
}
