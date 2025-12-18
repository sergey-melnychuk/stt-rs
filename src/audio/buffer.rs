//! Audio buffer management for continuous capture

use parking_lot::Mutex;
use std::sync::Arc;

/// Thread-safe ring buffer for audio samples
pub struct AudioBuffer {
    inner: Arc<Mutex<RingBuffer>>,
    sample_rate: u32,
}

struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    len: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with specified duration in seconds
    pub fn new(sample_rate: u32, duration_secs: f32) -> Self {
        let capacity = (sample_rate as f32 * duration_secs) as usize;

        Self {
            inner: Arc::new(Mutex::new(RingBuffer {
                data: vec![0.0; capacity],
                write_pos: 0,
                read_pos: 0,
                len: 0,
            })),
            sample_rate,
        }
    }

    /// Write samples to the buffer
    pub fn write(&self, samples: &[f32]) {
        let mut buffer = self.inner.lock();
        let capacity = buffer.data.len();

        for &sample in samples {
            let write_pos = buffer.write_pos;
            buffer.data[write_pos] = sample;
            buffer.write_pos = (write_pos + 1) % capacity;

            if buffer.len < capacity {
                buffer.len += 1;
            } else {
                // Overwriting old data - advance read position
                buffer.read_pos = (buffer.read_pos + 1) % capacity;
            }
        }
    }

    /// Read all available samples without removing them
    pub fn peek_all(&self) -> Vec<f32> {
        let buffer = self.inner.lock();
        let capacity = buffer.data.len();
        let mut result = Vec::with_capacity(buffer.len);

        let mut pos = buffer.read_pos;
        for _ in 0..buffer.len {
            result.push(buffer.data[pos]);
            pos = (pos + 1) % capacity;
        }

        result
    }

    /// Read and consume samples from the buffer
    pub fn read(&self, max_samples: usize) -> Vec<f32> {
        let mut buffer = self.inner.lock();
        let capacity = buffer.data.len();
        let to_read = max_samples.min(buffer.len);
        let mut result = Vec::with_capacity(to_read);

        for _ in 0..to_read {
            result.push(buffer.data[buffer.read_pos]);
            buffer.read_pos = (buffer.read_pos + 1) % capacity;
            buffer.len -= 1;
        }

        result
    }

    /// Get a window of audio with specified duration and overlap
    pub fn get_window(&self, window_secs: f32, overlap_secs: f32) -> Option<AudioWindow> {
        let buffer = self.inner.lock();
        let window_samples = (self.sample_rate as f32 * window_secs) as usize;

        if buffer.len < window_samples {
            return None;
        }

        let capacity = buffer.data.len();
        let mut samples = Vec::with_capacity(window_samples);

        let mut pos = buffer.read_pos;
        for _ in 0..window_samples {
            samples.push(buffer.data[pos]);
            pos = (pos + 1) % capacity;
        }

        let overlap_samples = (self.sample_rate as f32 * overlap_secs) as usize;

        Some(AudioWindow {
            samples,
            overlap_samples,
        })
    }

    /// Advance the read position after processing a window
    pub fn advance(&self, samples: usize) {
        let mut buffer = self.inner.lock();
        let capacity = buffer.data.len();
        let to_advance = samples.min(buffer.len);

        buffer.read_pos = (buffer.read_pos + to_advance) % capacity;
        buffer.len -= to_advance;
    }

    /// Get the number of samples currently in the buffer
    pub fn len(&self) -> usize {
        self.inner.lock().len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the duration of audio in the buffer (seconds)
    pub fn duration(&self) -> f32 {
        self.len() as f32 / self.sample_rate as f32
    }

    /// Clear the buffer
    pub fn clear(&self) {
        let mut buffer = self.inner.lock();
        buffer.read_pos = 0;
        buffer.write_pos = 0;
        buffer.len = 0;
    }

    /// Clone the buffer handle
    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            sample_rate: self.sample_rate,
        }
    }
}

/// A window of audio data for processing
#[derive(Debug, Clone)]
pub struct AudioWindow {
    /// Audio samples in the window
    pub samples: Vec<f32>,
    /// Number of samples that overlap with previous window
    pub overlap_samples: usize,
}

impl AudioWindow {
    /// Get the non-overlapping portion of the window
    pub fn new_samples(&self) -> &[f32] {
        &self.samples[self.overlap_samples..]
    }

    /// Get duration in seconds (assuming 16kHz sample rate)
    pub fn duration(&self, sample_rate: u32) -> f32 {
        self.samples.len() as f32 / sample_rate as f32
    }
}

/// Continuous window provider for streaming audio processing
pub struct WindowedBuffer {
    buffer: AudioBuffer,
    window_samples: usize,
    hop_samples: usize,
}

impl WindowedBuffer {
    /// Create a new windowed buffer
    pub fn new(
        sample_rate: u32,
        max_duration_secs: f32,
        window_secs: f32,
        overlap_secs: f32,
    ) -> Self {
        let window_samples = (sample_rate as f32 * window_secs) as usize;
        let hop_samples = (sample_rate as f32 * (window_secs - overlap_secs)) as usize;

        Self {
            buffer: AudioBuffer::new(sample_rate, max_duration_secs),
            window_samples,
            hop_samples,
        }
    }

    /// Write samples to the buffer
    pub fn write(&self, samples: &[f32]) {
        self.buffer.write(samples);
    }

    /// Get the next complete window if available
    pub fn next_window(&self) -> Option<Vec<f32>> {
        if self.buffer.len() < self.window_samples {
            return None;
        }

        let samples = self.buffer.peek_all();
        let window = samples[..self.window_samples].to_vec();

        // Advance by hop size
        self.buffer.advance(self.hop_samples);

        Some(window)
    }

    /// Get all remaining samples (for final processing)
    pub fn flush(&self) -> Vec<f32> {
        self.buffer.read(self.buffer.len())
    }

    /// Check if a window is ready
    pub fn has_window(&self) -> bool {
        self.buffer.len() >= self.window_samples
    }

    /// Get current buffer duration
    pub fn duration(&self) -> f32 {
        self.buffer.duration()
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer = AudioBuffer::new(16000, 1.0);

        // Write some samples
        buffer.write(&[1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3);

        // Read them back
        let samples = buffer.read(3);
        assert_eq!(samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        // Small buffer that can only hold 10 samples
        let buffer = AudioBuffer::new(10, 1.0);

        // Write more than capacity
        buffer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        buffer.write(&[6.0, 7.0, 8.0, 9.0, 10.0]);
        buffer.write(&[11.0, 12.0]); // Overflow

        assert_eq!(buffer.len(), 10);

        // Should have dropped oldest samples
        let samples = buffer.read(10);
        assert_eq!(samples, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_windowed_buffer() {
        // 1 second window, 0.5 second overlap, at 100 Hz for easy math
        let buffer = WindowedBuffer::new(100, 5.0, 1.0, 0.5);

        // Write 1.5 seconds of audio
        let samples: Vec<f32> = (0..150).map(|i| i as f32).collect();
        buffer.write(&samples);

        // Should have one window ready
        let window = buffer.next_window();
        assert!(window.is_some());
        let window = window.unwrap();
        assert_eq!(window.len(), 100); // 1 second at 100 Hz
        assert_eq!(window[0], 0.0);
        assert_eq!(window[99], 99.0);

        // After advancing by hop (50 samples), should have another window
        let window2 = buffer.next_window();
        assert!(window2.is_some());
        let window2 = window2.unwrap();
        assert_eq!(window2[0], 50.0); // Starts at sample 50
    }

    #[test]
    fn test_duration() {
        let buffer = AudioBuffer::new(16000, 2.0);
        buffer.write(&vec![0.0; 8000]); // 0.5 seconds
        assert!((buffer.duration() - 0.5).abs() < 0.001);
    }
}
