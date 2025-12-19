//! Benchmarks for audio preprocessing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use stt_rs::{AudioPreprocessor, PreprocessingConfig};

fn generate_audio(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies to simulate speech
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
        })
        .collect()
}

fn bench_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling");

    for duration in [0.1, 0.5, 1.0] {
        let audio = generate_audio(44100, duration);
        let config = PreprocessingConfig {
            enable_resampling: true,
            enable_filtering: false,
            enable_normalization: false,
            enable_agc: false,
            enable_noise_reduction: false,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("44100_to_16000", format!("{:.1}s", duration)),
            &audio,
            |b, audio| {
                b.iter_with_setup(
                    || AudioPreprocessor::new(config.clone(), 44100, 16000).unwrap(),
                    |mut preprocessor| {
                        black_box(preprocessor.process(audio).unwrap())
                    },
                )
            },
        );
    }

    group.finish();
}

fn bench_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtering");

    let audio = generate_audio(16000, 1.0);

    // Bandpass filter (high + low pass)
    let config = PreprocessingConfig {
        enable_resampling: false,
        enable_filtering: true,
        enable_normalization: false,
        enable_agc: false,
        enable_noise_reduction: false,
        high_pass_cutoff: 300.0,
        low_pass_cutoff: 3400.0,
        ..Default::default()
    };

    group.bench_function("bandpass_1s", |b| {
        b.iter_with_setup(
            || AudioPreprocessor::new(config.clone(), 16000, 16000).unwrap(),
            |mut preprocessor| {
                black_box(preprocessor.process(&audio).unwrap())
            },
        )
    });

    group.finish();
}

fn bench_agc(c: &mut Criterion) {
    let mut group = c.benchmark_group("agc");

    let audio = generate_audio(16000, 1.0);

    let config = PreprocessingConfig {
        enable_resampling: false,
        enable_filtering: false,
        enable_normalization: false,
        enable_agc: true,
        enable_noise_reduction: false,
        agc_target_level: 0.5,
        agc_attack_time: 0.01,
        agc_release_time: 0.1,
        ..Default::default()
    };

    group.bench_function("agc_1s", |b| {
        b.iter_with_setup(
            || AudioPreprocessor::new(config.clone(), 16000, 16000).unwrap(),
            |mut preprocessor| {
                black_box(preprocessor.process(&audio).unwrap())
            },
        )
    });

    group.finish();
}

fn bench_noise_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_reduction");

    let audio = generate_audio(16000, 1.0);

    for strength in [0.3, 0.5, 0.8] {
        let config = PreprocessingConfig {
            enable_resampling: false,
            enable_filtering: false,
            enable_normalization: false,
            enable_agc: false,
            enable_noise_reduction: true,
            noise_reduction_strength: strength,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("strength", format!("{:.1}", strength)),
            &audio,
            |b, audio| {
                b.iter_with_setup(
                    || AudioPreprocessor::new(config.clone(), 16000, 16000).unwrap(),
                    |mut preprocessor| {
                        black_box(preprocessor.process(audio).unwrap())
                    },
                )
            },
        );
    }

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    let audio = generate_audio(44100, 1.0);

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

    group.bench_function("resample_filter_agc_1s", |b| {
        b.iter_with_setup(
            || AudioPreprocessor::new(config.clone(), 44100, 16000).unwrap(),
            |mut preprocessor| {
                black_box(preprocessor.process(&audio).unwrap())
            },
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_resampling,
    bench_filtering,
    bench_agc,
    bench_noise_reduction,
    bench_full_pipeline
);
criterion_main!(benches);
