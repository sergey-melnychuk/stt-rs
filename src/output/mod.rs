//! Output formatting and writing modules

pub mod formats;

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

use crate::config::{OutputConfig, OutputFormat};
use crate::stt::TranscriptionResult;

pub use formats::{format_json, format_srt, format_text, format_vtt};

/// Output writer that handles multiple destinations
pub struct OutputWriter {
    config: OutputConfig,
    file: Option<File>,
    sequence_number: u32,
}

impl OutputWriter {
    /// Create a new output writer
    pub fn new(config: OutputConfig) -> io::Result<Self> {
        let file = if let Some(ref path) = config.output_path {
            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?,
            )
        } else {
            None
        };

        Ok(Self {
            config,
            file,
            sequence_number: 0,
        })
    }

    /// Write a transcription result
    pub fn write(&mut self, result: &TranscriptionResult, offset_ms: i64) -> io::Result<()> {
        if result.text.trim().is_empty() {
            return Ok(());
        }

        self.sequence_number += 1;

        let formatted = self.format(result, offset_ms);

        // Write to console if enabled
        if self.config.enable_console {
            self.write_console(&formatted)?;
        }

        // Write to file if configured
        if let Some(ref mut file) = self.file {
            writeln!(file, "{}", formatted)?;
            file.flush()?;
        }

        Ok(())
    }

    /// Format the transcription according to configured format
    fn format(&self, result: &TranscriptionResult, offset_ms: i64) -> String {
        match self.config.format {
            OutputFormat::Text => {
                if self.config.enable_timestamps {
                    format_text(result, offset_ms)
                } else {
                    result.text.clone()
                }
            }
            OutputFormat::Json => format_json(result, offset_ms),
            OutputFormat::Srt => format_srt(result, offset_ms, self.sequence_number),
            OutputFormat::Vtt => format_vtt(result, offset_ms, self.sequence_number == 1),
        }
    }

    fn write_console(&self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{}", text)?;
        stdout.flush()
    }

    /// Write a header (for formats that need it)
    pub fn write_header(&mut self) -> io::Result<()> {
        if self.config.format == OutputFormat::Vtt {
            let header = "WEBVTT\n";

            if self.config.enable_console {
                print!("{}", header);
            }

            if let Some(ref mut file) = self.file {
                write!(file, "{}", header)?;
            }
        }
        Ok(())
    }

    /// Flush any buffered output
    pub fn flush(&mut self) -> io::Result<()> {
        if let Some(ref mut file) = self.file {
            file.flush()?;
        }
        Ok(())
    }

    /// Get the output file path if configured
    pub fn output_path(&self) -> Option<&PathBuf> {
        self.config.output_path.as_ref()
    }
}

/// Simple console output for real-time display
pub struct ConsoleOutput {
    show_timestamps: bool,
    last_text: String,
}

impl ConsoleOutput {
    pub fn new(show_timestamps: bool) -> Self {
        Self {
            show_timestamps,
            last_text: String::new(),
        }
    }

    /// Print transcription result
    pub fn print(&mut self, result: &TranscriptionResult, offset_ms: i64) {
        if result.text.trim().is_empty() || result.text == self.last_text {
            return;
        }

        self.last_text = result.text.clone();

        if self.show_timestamps {
            let time = format_timestamp(offset_ms + result.start_ms);
            println!("[{}] {}", time, result.text);
        } else {
            println!("{}", result.text);
        }
    }

    /// Print partial/streaming result (updates in place)
    pub fn print_partial(&self, text: &str) {
        print!("\r{}\x1b[K", text);
        let _ = io::stdout().flush();
    }

    /// Clear the current line
    pub fn clear_line(&self) {
        print!("\r\x1b[K");
        let _ = io::stdout().flush();
    }
}

/// Format milliseconds as HH:MM:SS.mmm
pub fn format_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = ms % 1000;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
    } else {
        format!("{:02}:{:02}.{:03}", minutes, seconds, millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(0), "00:00.000");
        assert_eq!(format_timestamp(1500), "00:01.500");
        assert_eq!(format_timestamp(61000), "01:01.000");
        assert_eq!(format_timestamp(3661500), "01:01:01.500");
    }
}
