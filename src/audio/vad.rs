//! Voice Activity Detection module

use tracing::trace;

use crate::config::PreprocessingConfig;

/// Result of voice activity detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadResult {
    /// Speech detected
    Speech,
    /// Silence/noise detected
    Silence,
}

/// Voice Activity Detector using energy-based thresholding
pub struct VoiceActivityDetector {
    /// Energy threshold for speech detection
    threshold: f32,
    /// Sample rate for time calculations
    #[allow(dead_code)]
    sample_rate: u32,
    /// Minimum speech duration in samples
    min_speech_samples: usize,
    /// Minimum silence duration in samples
    min_silence_samples: usize,
    /// Current state
    current_state: VadResult,
    /// Counter for state persistence
    state_counter: usize,
    /// Running average of energy for adaptive threshold
    energy_avg: f32,
    /// Smoothing factor for energy average
    energy_alpha: f32,
}

impl VoiceActivityDetector {
    /// Create a new VAD instance
    pub fn new(config: &PreprocessingConfig, sample_rate: u32) -> Self {
        let min_speech_samples =
            (config.vad_min_speech_duration * sample_rate as f32) as usize;
        let min_silence_samples =
            (config.vad_min_silence_duration * sample_rate as f32) as usize;

        Self {
            threshold: config.vad_threshold,
            sample_rate,
            min_speech_samples,
            min_silence_samples,
            current_state: VadResult::Silence,
            state_counter: 0,
            energy_avg: 0.0,
            energy_alpha: 0.01, // Slow adaptation
        }
    }

    /// Process a frame of audio and return VAD result
    pub fn process(&mut self, samples: &[f32]) -> VadResult {
        let energy = self.calculate_energy(samples);

        // Update running average
        self.energy_avg = self.energy_alpha * energy + (1.0 - self.energy_alpha) * self.energy_avg;

        // Determine if this frame is speech based on energy
        // Use either absolute threshold OR relative to average (not both)
        let is_speech = energy > self.threshold || (self.energy_avg > 0.001 && energy > self.energy_avg * 2.0);

        // State machine with hysteresis
        match (self.current_state, is_speech) {
            (VadResult::Silence, true) => {
                self.state_counter += samples.len();
                if self.state_counter >= self.min_speech_samples {
                    self.current_state = VadResult::Speech;
                    self.state_counter = 0;
                    trace!("VAD: Silence -> Speech (energy: {:.4})", energy);
                }
            }
            (VadResult::Silence, false) => {
                self.state_counter = 0;
            }
            (VadResult::Speech, false) => {
                self.state_counter += samples.len();
                if self.state_counter >= self.min_silence_samples {
                    self.current_state = VadResult::Silence;
                    self.state_counter = 0;
                    trace!("VAD: Speech -> Silence (energy: {:.4})", energy);
                }
            }
            (VadResult::Speech, true) => {
                self.state_counter = 0;
            }
        }

        self.current_state
    }

    /// Calculate RMS energy of audio samples
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Get current VAD state
    pub fn current_state(&self) -> VadResult {
        self.current_state
    }

    /// Reset VAD state
    pub fn reset(&mut self) {
        self.current_state = VadResult::Silence;
        self.state_counter = 0;
        self.energy_avg = 0.0;
    }

    /// Check if currently in speech state
    pub fn is_speech(&self) -> bool {
        self.current_state == VadResult::Speech
    }

    /// Get the current energy average
    pub fn energy_average(&self) -> f32 {
        self.energy_avg
    }
}

/// Speech segment with timing information
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Audio samples
    pub samples: Vec<f32>,
}

/// Segmenter that accumulates audio and produces speech segments
pub struct SpeechSegmenter {
    vad: VoiceActivityDetector,
    sample_rate: u32,
    /// Buffer for current speech segment
    current_segment: Vec<f32>,
    /// Sample position for timing
    sample_position: usize,
    /// Start position of current segment
    segment_start: usize,
    /// Pre-roll samples to include before speech starts
    pre_roll_samples: usize,
    /// Pre-roll buffer
    pre_roll_buffer: Vec<f32>,
    /// Maximum segment duration in samples (forces segment completion)
    max_segment_samples: usize,
}

impl SpeechSegmenter {
    /// Create a new speech segmenter
    pub fn new(config: &PreprocessingConfig, sample_rate: u32) -> Self {
        let pre_roll_samples = (0.2 * sample_rate as f32) as usize; // 200ms pre-roll
        let max_segment_samples = (10.0 * sample_rate as f32) as usize; // 10 second max segment

        Self {
            vad: VoiceActivityDetector::new(config, sample_rate),
            sample_rate,
            current_segment: Vec::new(),
            sample_position: 0,
            segment_start: 0,
            pre_roll_samples,
            pre_roll_buffer: Vec::with_capacity(pre_roll_samples),
            max_segment_samples,
        }
    }

    /// Process audio samples and return completed segments
    pub fn process(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();

        // Process in smaller frames for VAD
        let frame_size = self.sample_rate as usize / 50; // 20ms frames

        for chunk in samples.chunks(frame_size) {
            let result = self.vad.process(chunk);

            match result {
                VadResult::Speech => {
                    if self.current_segment.is_empty() {
                        // Start of new segment - include pre-roll
                        self.segment_start = self.sample_position.saturating_sub(self.pre_roll_buffer.len());
                        self.current_segment.extend(&self.pre_roll_buffer);
                    }
                    self.current_segment.extend(chunk);

                    // Force segment completion if max duration exceeded
                    if self.current_segment.len() >= self.max_segment_samples {
                        trace!("VAD: Forcing segment completion (max duration reached)");
                        let segment = SpeechSegment {
                            start: self.segment_start as f32 / self.sample_rate as f32,
                            end: self.sample_position as f32 / self.sample_rate as f32,
                            samples: std::mem::take(&mut self.current_segment),
                        };
                        segments.push(segment);
                        // Start new segment immediately since speech is ongoing
                        self.segment_start = self.sample_position;
                    }
                }
                VadResult::Silence => {
                    if !self.current_segment.is_empty() {
                        // End of segment
                        let segment = SpeechSegment {
                            start: self.segment_start as f32 / self.sample_rate as f32,
                            end: self.sample_position as f32 / self.sample_rate as f32,
                            samples: std::mem::take(&mut self.current_segment),
                        };
                        segments.push(segment);
                    }
                }
            }

            // Update pre-roll buffer
            self.pre_roll_buffer.extend(chunk);
            if self.pre_roll_buffer.len() > self.pre_roll_samples {
                let excess = self.pre_roll_buffer.len() - self.pre_roll_samples;
                self.pre_roll_buffer.drain(0..excess);
            }

            self.sample_position += chunk.len();
        }

        segments
    }

    /// Flush any remaining segment
    pub fn flush(&mut self) -> Option<SpeechSegment> {
        if self.current_segment.is_empty() {
            return None;
        }

        let segment = SpeechSegment {
            start: self.segment_start as f32 / self.sample_rate as f32,
            end: self.sample_position as f32 / self.sample_rate as f32,
            samples: std::mem::take(&mut self.current_segment),
        };

        Some(segment)
    }

    /// Reset the segmenter state
    pub fn reset(&mut self) {
        self.vad.reset();
        self.current_segment.clear();
        self.sample_position = 0;
        self.segment_start = 0;
        self.pre_roll_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> PreprocessingConfig {
        PreprocessingConfig {
            vad_threshold: 0.05,
            vad_min_speech_duration: 0.1,
            vad_min_silence_duration: 0.2,
            ..Default::default()
        }
    }

    #[test]
    fn test_vad_silence() {
        let config = make_config();
        let mut vad = VoiceActivityDetector::new(&config, 16000);

        // Very quiet samples should be silence
        let silent = vec![0.001; 1600]; // 100ms of near-silence
        let result = vad.process(&silent);
        assert_eq!(result, VadResult::Silence);
    }

    #[test]
    fn test_vad_speech() {
        let config = make_config();
        let mut vad = VoiceActivityDetector::new(&config, 16000);

        // Loud samples should eventually trigger speech
        let loud: Vec<f32> = (0..3200)
            .map(|i| 0.5 * (i as f32 * 0.1).sin())
            .collect();

        // Process multiple frames to exceed min_speech_duration
        for chunk in loud.chunks(320) {
            vad.process(chunk);
        }

        assert_eq!(vad.current_state(), VadResult::Speech);
    }

    #[test]
    fn test_energy_calculation() {
        let config = make_config();
        let vad = VoiceActivityDetector::new(&config, 16000);

        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let energy = vad.calculate_energy(&samples);
        assert!((energy - 0.5).abs() < 0.01);
    }
}
