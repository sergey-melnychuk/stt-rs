//! Audio capture module using cpal

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleRate, Stream, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::config::AudioConfig;
use crate::error::{AudioError, Result};

/// Audio sample type alias
pub type AudioSample = f32;

/// Audio capture handle
pub struct AudioCapture {
    config: AudioConfig,
    host: Host,
    device: Option<Device>,
    stream: Option<Stream>,
    sample_sender: Sender<Vec<AudioSample>>,
    sample_receiver: Receiver<Vec<AudioSample>>,
    is_running: Arc<AtomicBool>,
    actual_sample_rate: u32,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new(config: AudioConfig) -> Result<Self> {
        let host = cpal::default_host();
        let (sender, receiver) = bounded(100); // Buffer up to 100 chunks

        Ok(Self {
            config,
            host,
            device: None,
            stream: None,
            sample_sender: sender,
            sample_receiver: receiver,
            is_running: Arc::new(AtomicBool::new(false)),
            actual_sample_rate: 0,
        })
    }

    /// List available audio input devices
    pub fn list_devices(&self) -> Result<Vec<String>> {
        let devices = self
            .host
            .input_devices()
            .map_err(|e| AudioError::DeviceConfig(e.to_string()))?;

        let mut names = Vec::new();
        for device in devices {
            if let Ok(name) = device.name() {
                names.push(name);
            }
        }
        Ok(names)
    }

    /// Initialize the audio capture device
    pub fn init(&mut self) -> Result<()> {
        let device = if let Some(ref device_name) = self.config.device {
            self.find_device_by_name(device_name)?
        } else {
            self.host
                .default_input_device()
                .ok_or(AudioError::NoInputDevice)?
        };

        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        info!("Using audio input device: {}", device_name);

        // Get supported configurations
        let supported_configs = device
            .supported_input_configs()
            .map_err(|e| AudioError::DeviceConfig(e.to_string()))?;

        // Find best matching configuration
        let mut best_config = None;
        for cfg in supported_configs {
            debug!(
                "Supported config: channels={}, sample_rate={:?}-{:?}",
                cfg.channels(),
                cfg.min_sample_rate(),
                cfg.max_sample_rate()
            );

            // Prefer matching channel count
            if cfg.channels() == self.config.channels {
                let target_rate = SampleRate(self.config.sample_rate);
                if cfg.min_sample_rate() <= target_rate && target_rate <= cfg.max_sample_rate() {
                    best_config = Some(cfg.with_sample_rate(target_rate));
                } else {
                    // Use max available sample rate (we'll resample later)
                    best_config = Some(cfg.with_max_sample_rate());
                }
                break;
            }
            if best_config.is_none() {
                best_config = Some(cfg.with_max_sample_rate());
            }
        }

        let supported_config = best_config.ok_or_else(|| {
            AudioError::DeviceConfig("No suitable audio configuration found".to_string())
        })?;

        self.actual_sample_rate = supported_config.sample_rate().0;
        info!(
            "Audio config: {} channels @ {} Hz (target: {} Hz)",
            supported_config.channels(),
            self.actual_sample_rate,
            self.config.sample_rate
        );

        self.device = Some(device);
        Ok(())
    }

    /// Get the actual sample rate of the capture device
    pub fn actual_sample_rate(&self) -> u32 {
        self.actual_sample_rate
    }

    /// Start capturing audio
    pub fn start(&mut self) -> Result<()> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| AudioError::DeviceConfig("Device not initialized".to_string()))?;

        let config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.actual_sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size),
        };

        let sender = self.sample_sender.clone();
        let is_running = self.is_running.clone();
        let channels = self.config.channels as usize;

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !is_running.load(Ordering::Relaxed) {
                        return;
                    }

                    // Convert to mono if stereo
                    let samples: Vec<f32> = if channels > 1 {
                        data.chunks(channels)
                            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                            .collect()
                    } else {
                        data.to_vec()
                    };

                    if sender.try_send(samples).is_err() {
                        warn!("Audio buffer overflow - dropping samples");
                    }
                },
                move |err| {
                    error!("Audio stream error: {}", err);
                },
                None,
            )
            .map_err(|e| AudioError::StreamBuild(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::StreamPlay(e.to_string()))?;

        self.is_running.store(true, Ordering::Relaxed);
        self.stream = Some(stream);

        info!("Audio capture started");
        Ok(())
    }

    /// Stop capturing audio
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);
        self.stream = None;
        info!("Audio capture stopped");
    }

    /// Check if capture is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Get the sample receiver channel
    pub fn receiver(&self) -> Receiver<Vec<AudioSample>> {
        self.sample_receiver.clone()
    }

    /// Try to receive audio samples (non-blocking)
    pub fn try_receive(&self) -> Option<Vec<AudioSample>> {
        self.sample_receiver.try_recv().ok()
    }

    /// Receive audio samples (blocking with timeout)
    pub fn receive_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Option<Vec<AudioSample>> {
        self.sample_receiver.recv_timeout(timeout).ok()
    }

    fn find_device_by_name(&self, name: &str) -> Result<Device> {
        let devices = self
            .host
            .input_devices()
            .map_err(|e| AudioError::DeviceConfig(e.to_string()))?;

        for device in devices {
            if let Ok(device_name) = device.name() {
                if device_name.contains(name) {
                    return Ok(device);
                }
            }
        }

        Err(AudioError::DeviceNotFound(name.to_string()).into())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_capture_creation() {
        let config = AudioConfig::default();
        let capture = AudioCapture::new(config);
        assert!(capture.is_ok());
    }

    #[test]
    fn test_list_devices() {
        let config = AudioConfig::default();
        let capture = AudioCapture::new(config).unwrap();
        let devices = capture.list_devices();
        // Just verify it doesn't panic - actual devices depend on system
        assert!(devices.is_ok());
    }
}
