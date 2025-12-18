# STT-RS: Radio Speech-to-Text System

A Rust-based real-time speech-to-text system designed for radio audio streams. Uses OpenAI's Whisper model (via whisper.cpp) for accurate offline transcription.

## Features

- **Real-time transcription** from microphone or audio input
- **Offline operation** - no cloud dependencies
- **Radio-optimized preprocessing** - bandpass filtering (300-3400 Hz), noise handling
- **Voice Activity Detection (VAD)** - only transcribes speech segments
- **Multiple output formats** - Text, JSON, SRT, VTT subtitles
- **Configurable** via TOML config file or CLI arguments

## Requirements

- Rust 1.70+
- CMake (for building whisper.cpp)
- C++ compiler

### macOS
```bash
xcode-select --install
brew install cmake
```

### Linux
```bash
sudo apt install build-essential cmake pkg-config libasound2-dev
```

## Installation

```bash
# Clone and build
git clone <repository>
cd stt-rs
cargo build --release

# Download Whisper model (base.en recommended for speed/accuracy balance)
mkdir -p models
curl -L -o models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### Available Models

| Model | Size | Memory | Speed | Accuracy |
|-------|------|--------|-------|----------|
| tiny.en | 75 MB | ~400 MB | Fastest | Good |
| base.en | 142 MB | ~500 MB | Fast | Better |
| small.en | 466 MB | ~1 GB | Medium | Great |
| medium.en | 1.5 GB | ~2.5 GB | Slow | Excellent |

Download from: https://huggingface.co/ggerganov/whisper.cpp/tree/main

## Quick Start

### 1. List Audio Devices

```bash
./target/release/stt-rs devices
```

Output:
```
Available audio input devices:
  1. MacBook Pro Microphone
  2. External USB Microphone
```

### 2. Record Audio from Microphone

Record a test audio file from your microphone:

```bash
# Record 10 seconds of audio
./target/release/stt-rs record --output my_recording.wav --duration 10

# Record from specific device
./target/release/stt-rs record -o my_recording.wav -d 10 --device "USB Microphone"

# Record with verbose logging
./target/release/stt-rs record -o my_recording.wav -d 5 -v
```

### 3. Transcribe Audio File

```bash
# Basic transcription
./target/release/stt-rs transcribe my_recording.wav --model models/ggml-base.en.bin

# With JSON output
./target/release/stt-rs transcribe my_recording.wav -m models/ggml-base.en.bin --format json

# Generate SRT subtitles
./target/release/stt-rs transcribe my_recording.wav -m models/ggml-base.en.bin --format srt
```

### 4. Real-Time Transcription

```bash
# Start live transcription from default microphone
./target/release/stt-rs run --model models/ggml-base.en.bin

# With verbose output
./target/release/stt-rs run -m models/ggml-base.en.bin -v

# Save to file while displaying on console
./target/release/stt-rs run -m models/ggml-base.en.bin --output transcript.txt

# Use specific audio device
./target/release/stt-rs run -m models/ggml-base.en.bin --device "MacBook Pro Microphone"
```

Press `Ctrl+C` to stop real-time transcription.

## Example Workflow: Record and Transcribe

```bash
# Step 1: Check available audio devices
./target/release/stt-rs devices

# Step 2: Record 10 seconds of speech
./target/release/stt-rs record -o speech.wav -d 10 -v

# Step 3: Transcribe the recording
./target/release/stt-rs transcribe speech.wav -m models/ggml-base.en.bin

# Step 4: Generate subtitles
./target/release/stt-rs transcribe speech.wav -m models/ggml-base.en.bin -f srt > speech.srt
```

## CLI Reference

```
stt-rs [OPTIONS] <COMMAND>

Commands:
  run         Start real-time transcription
  devices     List available audio input devices
  record      Record audio to a WAV file
  transcribe  Transcribe a WAV file
  help        Print help for commands

Global Options:
  -c, --config <FILE>   Load configuration from TOML file
  -v, --verbose         Increase verbosity (-v, -vv, -vvv)
  -h, --help            Print help
```

### run
```
stt-rs run [OPTIONS]

Options:
  -d, --device <NAME>   Audio input device name
  -m, --model <PATH>    Path to Whisper model file
  -o, --output <PATH>   Output file path
  -f, --format <FMT>    Output format: text, json, srt, vtt
      --no-console      Disable console output
```

### record
```
stt-rs record [OPTIONS]

Options:
  -o, --output <PATH>   Output WAV file [default: recording.wav]
  -d, --duration <SEC>  Recording duration in seconds [default: 10]
  -D, --device <NAME>   Audio input device name
```

### transcribe
```
stt-rs transcribe <INPUT> [OPTIONS]

Arguments:
  <INPUT>               Input WAV file path

Options:
  -m, --model <PATH>    Path to Whisper model file
  -f, --format <FMT>    Output format: text, json, srt, vtt
```

## Configuration File

Create `config.toml` for persistent settings (see `config.example.toml`):

```toml
[audio]
sample_rate = 16000
channels = 1
buffer_size = 512
# device = "MacBook Pro Microphone"

[preprocessing]
enable_resampling = true
enable_filtering = true
high_pass_cutoff = 300.0    # Remove low rumble
low_pass_cutoff = 3400.0    # Radio voice band
enable_normalization = true
enable_vad = true
vad_threshold = 0.05

[stt]
model_path = "./models/ggml-base.en.bin"
language = "en"
threads = 4

[output]
format = "text"
enable_timestamps = true
enable_console = true
```

Use with:
```bash
./target/release/stt-rs run --config config.toml
```

## Output Formats

### Text (default)
```
[00:00.000 --> 00:02.500] Hello, this is a test.
[00:02.500 --> 00:05.120] The quick brown fox jumps over the lazy dog.
```

### JSON
```json
{
  "text": "Hello, this is a test.",
  "start_ms": 0,
  "end_ms": 2500,
  "segments": [
    {"text": "Hello, this is a test.", "start_ms": 0, "end_ms": 2500}
  ]
}
```

### SRT (Subtitles)
```
1
00:00:00,000 --> 00:00:02,500
Hello, this is a test.

2
00:00:02,500 --> 00:00:05,120
The quick brown fox jumps over the lazy dog.
```

### VTT (Web Subtitles)
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, this is a test.
```

## Architecture

```
┌─────────────────┐
│  Audio Input    │  (microphone, line-in, radio)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Capture  │  cpal - cross-platform audio I/O
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  Resample → Filter → Normalize
│                 │  rubato + biquad
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      VAD        │  Voice Activity Detection
│                 │  Energy-based thresholding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Whisper STT    │  whisper-rs (whisper.cpp bindings)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Output      │  Text / JSON / SRT / VTT
└─────────────────┘
```

## Project Structure

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library exports
├── config.rs            # Configuration structs
├── error.rs             # Error types
├── audio/
│   ├── capture.rs       # Audio capture (cpal)
│   ├── preprocessing.rs # Resampling, filtering
│   ├── vad.rs           # Voice activity detection
│   └── buffer.rs        # Ring buffer management
├── stt/
│   └── engine.rs        # Whisper integration
└── output/
    ├── mod.rs           # Output writer
    └── formats.rs       # Format implementations
```

## Performance

Tested on MacBook Pro M1:

| Model | Load Time | Transcription (5s audio) | Memory |
|-------|-----------|--------------------------|--------|
| base.en | ~150ms | ~500ms | ~500 MB |

## Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture
```

## Troubleshooting

### No audio devices found
- Check microphone permissions in System Preferences/Settings
- Ensure audio device is connected and recognized by OS

### Model loading fails
- Verify model file exists and is not corrupted
- Check file permissions
- Re-download the model file

### Poor transcription quality
- Use a larger model (small.en or medium.en)
- Ensure clear audio input with minimal background noise
- Check that preprocessing filters aren't too aggressive

### High CPU usage
- Reduce `threads` in config (default: 4)
- Use a smaller model (tiny.en or base.en)

## License

MIT

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Whisper inference in C++
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for whisper.cpp
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O
