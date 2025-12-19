//! Integration tests for stt-rs

use stt_rs::{
    AudioPreprocessor, Config, PreprocessingConfig, RealtimeConfig,
    SpeechSegmenter, VoiceActivityDetector,
};

/// Generate synthetic audio that simulates speech
fn generate_speech(sample_rate: u32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies to simulate speech formants
            let f1 = 300.0; // First formant
            let f2 = 1000.0; // Second formant
            let f3 = 2500.0; // Third formant

            amplitude * (
                0.5 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * f2 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * f3 * t).sin()
            )
        })
        .collect()
}

/// Generate silence with minimal noise
fn generate_silence(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    vec![0.0001; num_samples]
}

#[test]
fn test_config_loading() {
    let config = Config::default();

    assert_eq!(config.audio.sample_rate, 16000);
    assert_eq!(config.realtime.target_sample_rate, 16000);
    assert!(config.preprocessing.enable_filtering);
    assert!(config.preprocessing.enable_agc);
}

#[test]
fn test_config_from_toml() {
    let toml_str = r#"
        [audio]
        sample_rate = 44100

        [preprocessing]
        enable_agc = false
        enable_noise_reduction = true
        noise_reduction_strength = 0.7

        [realtime]
        enable_degradation = false
        max_lag_seconds = 10.0
    "#;

    let config: Config = toml::from_str(toml_str).expect("Failed to parse TOML");

    assert_eq!(config.audio.sample_rate, 44100);
    assert!(!config.preprocessing.enable_agc);
    assert!(config.preprocessing.enable_noise_reduction);
    assert_eq!(config.preprocessing.noise_reduction_strength, 0.7);
    assert!(!config.realtime.enable_degradation);
    assert_eq!(config.realtime.max_lag_seconds, 10.0);
}

#[test]
fn test_full_preprocessing_pipeline() {
    // Test the complete preprocessing pipeline
    let config = PreprocessingConfig {
        enable_resampling: true,
        enable_filtering: true,
        enable_normalization: false,
        enable_agc: true,
        enable_noise_reduction: false,
        high_pass_cutoff: 300.0,
        low_pass_cutoff: 3400.0,
        ..Default::default()
    };

    let mut preprocessor = AudioPreprocessor::new(config, 44100, 16000)
        .expect("Failed to create preprocessor");

    // Generate 1 second of 44.1kHz audio
    let input = generate_speech(44100, 1.0, 0.3);

    // Process in chunks (simulating streaming)
    let mut output = Vec::new();
    for chunk in input.chunks(4410) { // 100ms chunks
        let processed = preprocessor.process(chunk)
            .expect("Processing failed");
        output.extend(processed);
    }

    // Output should be approximately 16000 samples (1 second at 16kHz)
    // Allow some tolerance due to resampling buffer
    assert!(output.len() > 14000 && output.len() < 18000,
        "Expected ~16000 samples, got {}", output.len());

    // All samples should be in valid range
    assert!(output.iter().all(|&s| s >= -1.5 && s <= 1.5),
        "Output contains out-of-range samples");
}

#[test]
fn test_vad_detects_speech() {
    let config = PreprocessingConfig::default();
    let sample_rate = 16000;

    let mut vad = VoiceActivityDetector::new(&config, sample_rate);

    // Feed silence first
    let silence = generate_silence(sample_rate, 0.5);
    for chunk in silence.chunks(320) {
        vad.process(chunk);
    }
    assert!(!vad.is_speech(), "VAD should detect silence");

    // Now feed speech
    let speech = generate_speech(sample_rate, 0.5, 0.3);
    for chunk in speech.chunks(320) {
        vad.process(chunk);
    }
    assert!(vad.is_speech(), "VAD should detect speech");
}

#[test]
fn test_speech_segmenter_produces_segments() {
    let config = PreprocessingConfig {
        vad_threshold: 0.01,
        vad_min_speech_duration: 0.1,
        vad_min_silence_duration: 0.2,
        vad_pre_roll: 0.1,
        vad_max_segment_duration: 5.0,
        ..Default::default()
    };
    let sample_rate = 16000;

    let mut segmenter = SpeechSegmenter::new(&config, sample_rate);

    // Generate mixed audio: silence, speech, silence, speech
    let mut audio = Vec::new();
    audio.extend(generate_silence(sample_rate, 0.3));
    audio.extend(generate_speech(sample_rate, 0.5, 0.3));
    audio.extend(generate_silence(sample_rate, 0.3));
    audio.extend(generate_speech(sample_rate, 0.8, 0.4));
    audio.extend(generate_silence(sample_rate, 0.3));

    // Process all audio
    let mut segments = segmenter.process(&audio);

    // Flush remaining
    if let Some(final_segment) = segmenter.flush() {
        segments.push(final_segment);
    }

    // Should have detected at least one segment
    assert!(!segments.is_empty(), "Should have detected speech segments");

    // Each segment should have valid timing
    for segment in &segments {
        assert!(segment.end > segment.start, "Segment end should be after start");
        assert!(!segment.samples.is_empty(), "Segment should have samples");
    }
}

#[test]
fn test_max_segment_duration() {
    let config = PreprocessingConfig {
        vad_threshold: 0.01,
        vad_min_speech_duration: 0.1,
        vad_min_silence_duration: 0.2,
        vad_max_segment_duration: 1.0, // Force split after 1 second
        ..Default::default()
    };
    let sample_rate = 16000;

    let mut segmenter = SpeechSegmenter::new(&config, sample_rate);

    // Generate 3 seconds of continuous speech
    let speech = generate_speech(sample_rate, 3.0, 0.3);

    let segments = segmenter.process(&speech);

    // Should have at least 2 segments due to max duration
    assert!(segments.len() >= 2,
        "Expected at least 2 segments due to max duration, got {}", segments.len());
}

#[test]
fn test_agc_amplifies_quiet_audio() {
    let config = PreprocessingConfig {
        enable_resampling: false,
        enable_filtering: false,
        enable_normalization: false,
        enable_agc: true,
        enable_noise_reduction: false,
        agc_target_level: 0.5,
        agc_attack_time: 0.001,
        agc_release_time: 0.01,
        ..Default::default()
    };

    let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000)
        .expect("Failed to create preprocessor");

    // Very quiet input
    let quiet_audio = generate_speech(16000, 0.5, 0.05);
    let input_rms = rms(&quiet_audio);

    let output = preprocessor.process(&quiet_audio)
        .expect("Processing failed");
    let output_rms = rms(&output);

    // AGC should have amplified the signal
    assert!(output_rms > input_rms * 1.5,
        "AGC should amplify quiet audio: input RMS={:.4}, output RMS={:.4}",
        input_rms, output_rms);
}

#[test]
fn test_noise_reduction_reduces_amplitude() {
    let config = PreprocessingConfig {
        enable_resampling: false,
        enable_filtering: false,
        enable_normalization: false,
        enable_agc: false,
        enable_noise_reduction: true,
        noise_reduction_strength: 0.8,
        ..Default::default()
    };

    let mut preprocessor = AudioPreprocessor::new(config, 16000, 16000)
        .expect("Failed to create preprocessor");

    // First, feed noise to establish floor
    let noise: Vec<f32> = (0..8000).map(|i| 0.01 * ((i * 17) as f32 % 1.0 - 0.5)).collect();
    let _ = preprocessor.process(&noise);

    // Now process slightly louder noise
    let input: Vec<f32> = (0..8000).map(|i| 0.02 * ((i * 23) as f32 % 1.0 - 0.5)).collect();
    let input_rms = rms(&input);

    let output = preprocessor.process(&input)
        .expect("Processing failed");
    let output_rms = rms(&output);

    // Noise reduction should reduce or maintain amplitude
    assert!(output_rms <= input_rms * 1.1,
        "Noise reduction should not increase amplitude significantly");
}

#[test]
fn test_secure_clear() {
    let config = PreprocessingConfig::default();
    let mut preprocessor = AudioPreprocessor::new(config, 44100, 16000)
        .expect("Failed to create preprocessor");

    // Process some audio
    let audio = generate_speech(44100, 1.0, 0.3);
    let _ = preprocessor.process(&audio);

    // Secure clear
    preprocessor.secure_clear();

    // Process more audio - should work normally
    let more_audio = generate_speech(44100, 0.5, 0.3);
    let result = preprocessor.process(&more_audio);
    assert!(result.is_ok(), "Should be able to process after secure_clear");
}

#[test]
fn test_realtime_config_defaults() {
    let config = RealtimeConfig::default();

    assert_eq!(config.target_sample_rate, 16000);
    assert!(config.enable_degradation);
    assert_eq!(config.max_lag_seconds, 5.0);
    assert_eq!(config.min_segment_seconds, 0.5);
}

/// Helper function to calculate RMS
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}
