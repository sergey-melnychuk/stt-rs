//! Benchmarks for Voice Activity Detection

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use stt_rs::{PreprocessingConfig, SpeechSegmenter, VoiceActivityDetector};

fn generate_speech_like_audio(sample_rate: u32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simulate speech with varying amplitude
            let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            amplitude * envelope * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
        })
        .collect()
}

fn generate_silence(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    vec![0.001; num_samples] // Very quiet noise floor
}

fn generate_mixed_audio(sample_rate: u32) -> Vec<f32> {
    // 5 seconds: silence, speech, silence, speech, silence
    let mut audio = Vec::new();
    audio.extend(generate_silence(sample_rate, 0.5));
    audio.extend(generate_speech_like_audio(sample_rate, 1.0, 0.3));
    audio.extend(generate_silence(sample_rate, 0.5));
    audio.extend(generate_speech_like_audio(sample_rate, 2.0, 0.4));
    audio.extend(generate_silence(sample_rate, 1.0));
    audio
}

fn bench_vad_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_detector");
    let sample_rate = 16000;

    let config = PreprocessingConfig::default();

    // Benchmark on continuous speech
    let speech = generate_speech_like_audio(sample_rate, 1.0, 0.3);
    group.bench_function("continuous_speech_1s", |b| {
        b.iter_with_setup(
            || VoiceActivityDetector::new(&config, sample_rate),
            |mut vad| {
                for chunk in speech.chunks(320) { // 20ms frames
                    black_box(vad.process(chunk));
                }
            },
        )
    });

    // Benchmark on silence
    let silence = generate_silence(sample_rate, 1.0);
    group.bench_function("silence_1s", |b| {
        b.iter_with_setup(
            || VoiceActivityDetector::new(&config, sample_rate),
            |mut vad| {
                for chunk in silence.chunks(320) {
                    black_box(vad.process(chunk));
                }
            },
        )
    });

    group.finish();
}

fn bench_speech_segmenter(c: &mut Criterion) {
    let mut group = c.benchmark_group("speech_segmenter");
    let sample_rate = 16000;

    let config = PreprocessingConfig::default();

    // Mixed audio with speech and silence
    let mixed = generate_mixed_audio(sample_rate);

    group.bench_function("mixed_5s", |b| {
        b.iter_with_setup(
            || SpeechSegmenter::new(&config, sample_rate),
            |mut segmenter| {
                let segments = segmenter.process(&mixed);
                if let Some(final_segment) = segmenter.flush() {
                    black_box(final_segment);
                }
                black_box(segments)
            },
        )
    });

    // Streaming simulation - process in small chunks
    group.bench_function("mixed_5s_streaming", |b| {
        b.iter_with_setup(
            || SpeechSegmenter::new(&config, sample_rate),
            |mut segmenter| {
                let mut all_segments = Vec::new();
                for chunk in mixed.chunks(512) { // ~32ms chunks
                    all_segments.extend(segmenter.process(chunk));
                }
                if let Some(final_segment) = segmenter.flush() {
                    all_segments.push(final_segment);
                }
                black_box(all_segments)
            },
        )
    });

    group.finish();
}

fn bench_vad_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("vad_thresholds");
    let sample_rate = 16000;
    let speech = generate_speech_like_audio(sample_rate, 1.0, 0.3);

    for threshold in [0.005, 0.01, 0.05, 0.1] {
        let config = PreprocessingConfig {
            vad_threshold: threshold,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("threshold", format!("{:.3}", threshold)),
            &speech,
            |b, speech| {
                b.iter_with_setup(
                    || VoiceActivityDetector::new(&config, sample_rate),
                    |mut vad| {
                        for chunk in speech.chunks(320) {
                            black_box(vad.process(chunk));
                        }
                    },
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vad_detector,
    bench_speech_segmenter,
    bench_vad_thresholds
);
criterion_main!(benches);
