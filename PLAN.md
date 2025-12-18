# Radio Speech-to-Text System - Project Plan

## Project Overview

Build a Rust-based prototype system that captures audio from military-grade radio streams and performs real-time speech-to-text transcription. The system must handle the unique challenges of radio audio including codec artifacts, low bandwidth, high noise floor, and compression artifacts.

## Core Objectives

1. Capture audio stream from radio output (analog or digital)
2. Preprocess audio to handle radio-specific artifacts
3. Perform accurate speech-to-text transcription
4. Output transcribed text in real-time or near-real-time
5. Support offline operation (no cloud dependencies for sensitive environments)

## Technical Architecture

### High-Level Pipeline

```
[Radio Audio Output]
    ↓
[Audio Capture] (cpal)
    ↓
[Audio Preprocessing]
    - Resampling (to 16kHz)
    - Noise reduction
    - Normalization
    - VAD (Voice Activity Detection)
    ↓
[Buffer Management]
    - Windowed audio chunks
    - Overlap handling
    ↓
[STT Engine] (Whisper.cpp)
    ↓
[Text Output]
    - Console/File/Stream
```

## Components Breakdown

### 1. Audio Capture Module (`src/audio/capture.rs`)

**Responsibilities:**
- Initialize audio input device
- Configure sample rate, channels, buffer size
- Capture continuous audio stream
- Handle audio device errors and reconnection

**Key APIs:**
- `cpal` for cross-platform audio I/O
- Support for default input device or specific device selection
- Configurable buffer sizes (128-2048 samples)

**Configuration:**
```rust
AudioConfig {
    sample_rate: 16000,  // 16kHz is standard for STT
    channels: 1,          // Mono
    buffer_size: 512,
}
```

### 2. Audio Preprocessing Module (`src/audio/preprocessing.rs`)

**Responsibilities:**
- Resample audio to target rate (16kHz for Whisper)
- Apply noise reduction/filtering
- Normalize audio levels
- Handle codec artifacts (MELP, CVSD, etc.)

**Signal Processing:**
- High-pass filter (remove DC offset and low-frequency rumble)
- Band-pass filter (300Hz - 3400Hz for radio voice band)
- AGC compensation
- Spectral subtraction for noise reduction (optional)

**Dependencies:**
- `rubato` - Sample rate conversion
- `biquad` - Digital filters
- Custom DSP utilities for radio-specific processing

### 3. Voice Activity Detection (`src/audio/vad.rs`)

**Responsibilities:**
- Detect speech vs silence/noise
- Segment audio into speech chunks
- Reduce processing of non-speech audio

**Algorithm Options:**
- Energy-based thresholding (simple, fast)
- WebRTC VAD (more robust)
- Spectral entropy analysis
- Zero-crossing rate

**Output:**
- Speech/non-speech classification per audio frame
- Segment boundaries for chunking

### 4. Buffer Management (`src/audio/buffer.rs`)

**Responsibilities:**
- Manage audio ringbuffer for continuous capture
- Create overlapping windows for STT processing
- Handle chunk boundaries to avoid cutting words

**Parameters:**
- Window size: 3-10 seconds of audio
- Overlap: 0.5-1 second
- Max buffer: 30 seconds (before dropping old data)

### 5. STT Engine Integration (`src/stt/engine.rs`)

**Primary Option: Whisper.cpp**
- Local, offline processing
- State-of-the-art accuracy
- Multiple model sizes (tiny, base, small, medium, large)

**Integration:**
- Use `whisper-rs` or direct FFI bindings to whisper.cpp
- Model loading and initialization
- Async processing of audio chunks
- Result handling and formatting

**Alternative Options (for comparison):**
- Vosk (lightweight, faster but less accurate)
- Coqui STT (DeepSpeech successor)

### 6. Output Module (`src/output/mod.rs`)

**Responsibilities:**
- Format transcription results
- Output to multiple sinks (console, file, network)
- Handle timestamps and speaker metadata
- Provide real-time streaming or batched output

**Output Formats:**
- Plain text
- JSON with timestamps
- SRT/VTT subtitles
- Structured logs

## Implementation Phases

### Phase 1: Basic Audio Capture (Week 1)
- [ ] Set up Rust project with cargo workspace
- [ ] Implement basic audio capture using cpal
- [ ] Write captured audio to WAV file for verification
- [ ] Test with various audio sources (microphone, line-in)
- [ ] Verify audio quality and format

**Deliverable:** Command-line tool that captures audio and saves to file

### Phase 2: Audio Preprocessing (Week 1-2)
- [ ] Implement resampling to 16kHz
- [ ] Add basic filtering (high-pass, band-pass)
- [ ] Implement audio normalization
- [ ] Add VAD using energy threshold
- [ ] Test preprocessing pipeline with radio recordings

**Deliverable:** Audio preprocessing pipeline with configurable parameters

### Phase 3: STT Integration (Week 2)
- [ ] Integrate whisper.cpp via Rust bindings
- [ ] Download and test Whisper models (start with `base.en`)
- [ ] Implement chunk-based processing
- [ ] Handle model initialization and inference
- [ ] Test transcription accuracy with clean audio

**Deliverable:** Working STT system with preprocessed audio input

### Phase 4: Buffer Management & Streaming (Week 3)
- [ ] Implement ringbuffer for continuous audio
- [ ] Add overlapping window logic
- [ ] Implement real-time processing pipeline
- [ ] Add async/parallel processing for performance
- [ ] Handle graceful degradation under load

**Deliverable:** Real-time streaming STT system

### Phase 5: Radio-Specific Optimization (Week 3-4)
- [ ] Test with actual military radio audio samples
- [ ] Tune preprocessing for codec artifacts
- [ ] Optimize VAD parameters for noisy radio
- [ ] Add adaptive filtering based on SNR
- [ ] Benchmark accuracy and latency

**Deliverable:** Production-ready system optimized for radio audio

### Phase 6: Output & UI (Week 4)
- [ ] Implement multiple output formats
- [ ] Add command-line interface with clap
- [ ] Implement configuration file support
- [ ] Add logging and diagnostics
- [ ] Create basic monitoring/dashboard (optional)

**Deliverable:** Complete system with user interface

## Key Technical Considerations

### Military Radio Characteristics

**Bandwidth Limitations:**
- Typical: 2.4 kHz - 4 kHz (vs 20 kHz for normal audio)
- MELP codec: 2.4 kHz bandwidth
- Impact: Reduced frequency range affects phoneme recognition

**Compression Artifacts:**
- MELP (Mixed Excitation Linear Prediction): 2.4 kbps
- CVSD (Continuous Variable Slope Delta): 16-64 kbps
- Impacts: Metallic sound, quantization noise

**Noise Floor:**
- RF interference
- Atmospheric noise (HF)
- Co-channel interference
- Typical SNR: 10-20 dB (vs 40+ dB for clean audio)

**AGC (Automatic Gain Control):**
- Clipping on loud signals
- Pumping/breathing effects
- Dynamic range compression

### Performance Requirements

**Latency:**
- Target: < 2 seconds end-to-end
- Audio capture: ~100ms
- Preprocessing: ~50ms
- STT inference: 1-5s (depends on model size)

**Accuracy:**
- Target: > 85% WER (Word Error Rate) on radio audio
- Baseline with Whisper on clean audio: ~95% WER
- Degradation expected due to radio artifacts

**Resource Usage:**
- CPU: Whisper.cpp is CPU-intensive
  - tiny model: ~1 core
  - base model: ~2 cores
  - small/medium: 4-8 cores
- Memory: 500MB - 2GB depending on model
- No GPU required (but can accelerate with CUDA/Metal)

### Security Considerations

**Offline Operation:**
- No cloud APIs for sensitive military communications
- All processing must be local
- Consider air-gapped deployment scenarios

**Data Handling:**
- Audio buffer security (no disk persistence by default)
- Secure deletion of temporary files
- Encrypted output option
- No telemetry or logging of sensitive data

## Dependencies

### Core Crates

```toml
[dependencies]
# Audio I/O
cpal = "0.15"
hound = "3.5"  # WAV encoding/decoding

# Signal processing
rubato = "0.15"  # Resampling
biquad = "0.4"   # Digital filters

# STT Engine
whisper-rs = "0.10"  # Or direct whisper.cpp bindings

# Async runtime
tokio = { version = "1", features = ["full"] }

# CLI and config
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Error handling
anyhow = "1.0"
thiserror = "1.0"
```

### System Requirements

**Build Dependencies:**
- Rust 1.75+
- C++ compiler (for whisper.cpp)
- CMake (for building whisper.cpp)
- pkg-config
- ALSA/PulseAudio dev libraries (Linux)

**Runtime Dependencies:**
- Whisper model files (download separately)
- Audio device drivers
- ~2GB RAM minimum

## Testing Strategy

### Unit Tests
- Audio processing functions (resampling, filtering)
- VAD algorithm accuracy
- Buffer management edge cases
- Configuration parsing

### Integration Tests
- End-to-end pipeline with test audio files
- Different audio formats and sample rates
- Error handling and recovery
- Performance benchmarks

### Test Data
- Clean speech recordings
- Simulated radio audio (with compression)
- Actual military radio recordings (if available)
- Various noise levels and conditions
- Edge cases: silence, very loud, clipping

### Benchmarks
- Processing latency (capture → output)
- CPU usage per model size
- Memory footprint
- Accuracy (WER) on test corpus

## Configuration Example

```toml
# config.toml
[audio]
sample_rate = 16000
channels = 1
buffer_size = 512
device = "default"  # or specific device name

[preprocessing]
enable_resampling = true
enable_filtering = true
high_pass_cutoff = 300
low_pass_cutoff = 3400
enable_normalization = true
enable_vad = true
vad_threshold = 0.05

[stt]
model_path = "./models/ggml-base.en.bin"
model_size = "base"
language = "en"
threads = 4

[output]
format = "json"  # json, text, srt
output_path = "./transcriptions"
enable_timestamps = true
enable_console = true
```

## Success Criteria

1. **Functional Requirements:**
   - Captures audio from radio input
   - Produces readable transcriptions
   - Operates in real-time or near-real-time
   - Handles continuous operation (hours)

2. **Performance Requirements:**
   - < 2 second latency
   - > 85% accuracy on radio audio
   - Stable under continuous operation
   - Low CPU usage (< 50% of available cores)

3. **Quality Requirements:**
   - Handles codec artifacts gracefully
   - Robust to noise and interference
   - Minimal false positives from non-speech
   - Clear error messages and diagnostics

## Future Enhancements

1. **Speaker Diarization:** Identify different speakers
2. **Keyword Spotting:** Real-time alerts for specific words/phrases
3. **Multi-language Support:** Beyond English
4. **Network Streaming:** Send transcriptions to remote server
5. **GPU Acceleration:** Faster inference with CUDA/Metal
6. **Custom Model Fine-tuning:** Train on radio-specific audio
7. **Integration with SDR:** Direct software-defined radio integration

## Resources

**Documentation:**
- Whisper: https://github.com/openai/whisper
- Whisper.cpp: https://github.com/ggerganov/whisper.cpp
- CPAL: https://docs.rs/cpal/

**Reference Audio Processing:**
- Digital Signal Processing basics
- Speech processing fundamentals
- Radio communication systems

**Military Radio Codecs:**
- MELP (MIL-STD-3005)
- CVSD specifications
- Radio channel characteristics

## Notes for Claude Code

- Start with Phase 1 and build incrementally
- Each phase should have working, testable output
- Prioritize correctness over optimization initially
- Use `cargo test` extensively during development
- Consider using `cargo-criterion` for benchmarking
- Document any assumptions about audio format or radio characteristics
- Keep security in mind - avoid unnecessary data persistence

