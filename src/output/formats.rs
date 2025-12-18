//! Output format implementations

use serde::Serialize;

use crate::stt::TranscriptionResult;
use super::format_timestamp;

/// JSON output structure
#[derive(Debug, Serialize)]
struct JsonOutput {
    text: String,
    start_ms: i64,
    end_ms: i64,
    segments: Vec<JsonSegment>,
}

#[derive(Debug, Serialize)]
struct JsonSegment {
    text: String,
    start_ms: i64,
    end_ms: i64,
}

/// Format as plain text with timestamps
pub fn format_text(result: &TranscriptionResult, offset_ms: i64) -> String {
    let start = format_timestamp(offset_ms + result.start_ms);
    let end = format_timestamp(offset_ms + result.end_ms);
    format!("[{} --> {}] {}", start, end, result.text)
}

/// Format as JSON
pub fn format_json(result: &TranscriptionResult, offset_ms: i64) -> String {
    let output = JsonOutput {
        text: result.text.clone(),
        start_ms: offset_ms + result.start_ms,
        end_ms: offset_ms + result.end_ms,
        segments: result
            .segments
            .iter()
            .map(|s| JsonSegment {
                text: s.text.clone(),
                start_ms: offset_ms + s.start_ms,
                end_ms: offset_ms + s.end_ms,
            })
            .collect(),
    };

    serde_json::to_string(&output).unwrap_or_else(|_| format!("{{\"text\": \"{}\"}}", result.text))
}

/// Format as SRT subtitle
pub fn format_srt(result: &TranscriptionResult, offset_ms: i64, sequence: u32) -> String {
    let start = format_srt_timestamp(offset_ms + result.start_ms);
    let end = format_srt_timestamp(offset_ms + result.end_ms);

    format!("{}\n{} --> {}\n{}\n", sequence, start, end, result.text)
}

/// Format as WebVTT subtitle
pub fn format_vtt(result: &TranscriptionResult, offset_ms: i64, include_header: bool) -> String {
    let start = format_vtt_timestamp(offset_ms + result.start_ms);
    let end = format_vtt_timestamp(offset_ms + result.end_ms);

    if include_header {
        format!("WEBVTT\n\n{} --> {}\n{}\n", start, end, result.text)
    } else {
        format!("{} --> {}\n{}\n", start, end, result.text)
    }
}

/// Format timestamp for SRT (HH:MM:SS,mmm)
fn format_srt_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Format timestamp for VTT (HH:MM:SS.mmm)
fn format_vtt_timestamp(ms: i64) -> String {
    let total_seconds = ms / 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::TranscriptionSegment;

    fn make_result() -> TranscriptionResult {
        TranscriptionResult {
            text: "Hello world".to_string(),
            start_ms: 1000,
            end_ms: 2500,
            segments: vec![TranscriptionSegment {
                text: "Hello world".to_string(),
                start_ms: 1000,
                end_ms: 2500,
            }],
        }
    }

    #[test]
    fn test_format_text() {
        let result = make_result();
        let formatted = format_text(&result, 0);
        assert!(formatted.contains("[00:01.000 --> 00:02.500]"));
        assert!(formatted.contains("Hello world"));
    }

    #[test]
    fn test_format_json() {
        let result = make_result();
        let formatted = format_json(&result, 0);
        assert!(formatted.contains("\"text\":\"Hello world\""));
        assert!(formatted.contains("\"start_ms\":1000"));
    }

    #[test]
    fn test_format_srt() {
        let result = make_result();
        let formatted = format_srt(&result, 0, 1);
        assert!(formatted.contains("1\n"));
        assert!(formatted.contains("00:00:01,000 --> 00:00:02,500"));
        assert!(formatted.contains("Hello world"));
    }

    #[test]
    fn test_format_vtt() {
        let result = make_result();
        let formatted = format_vtt(&result, 0, true);
        assert!(formatted.starts_with("WEBVTT"));
        assert!(formatted.contains("00:00:01.000 --> 00:00:02.500"));
    }

    #[test]
    fn test_format_srt_timestamp() {
        assert_eq!(format_srt_timestamp(0), "00:00:00,000");
        assert_eq!(format_srt_timestamp(1500), "00:00:01,500");
        assert_eq!(format_srt_timestamp(3661500), "01:01:01,500");
    }
}
