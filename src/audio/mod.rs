//! Audio processing modules

pub mod buffer;
pub mod capture;
pub mod preprocessing;
pub mod vad;

pub use buffer::{AudioBuffer, AudioWindow, WindowedBuffer};
pub use capture::{AudioCapture, AudioSample};
pub use preprocessing::AudioPreprocessor;
pub use vad::{SpeechSegment, SpeechSegmenter, VadResult, VoiceActivityDetector};
