import os
import math
import json
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# 1. Environment and Path Setup
load_dotenv()

# Ensure FFmpeg/FFprobe binaries are available in PATH
extra_paths = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/miniconda3/envs/bootcamp-env/bin"
]
os.environ["PATH"] += ":" + ":".join(extra_paths)

ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")
if ffmpeg_path and ffprobe_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure working directories exist
os.makedirs("chunks", exist_ok=True)
os.makedirs("exports", exist_ok=True)


# 2. Transcription Functions
def transcribe_basic(audio_file_path, prompt=None):
    """Performs standard Whisper transcription (guided if prompt provided)."""
    with open(audio_file_path, "rb") as f:
        params = {"model": "whisper-1", "file": f}
        if prompt:
            params["prompt"] = prompt
        response = client.audio.transcriptions.create(**params)
    return response.text


def split_and_transcribe_audio(file_path, chunk_length_ms=60000, prompt=None):
    """
    Chunks audio into slices, transcribes with segment timestamps,
    and calculates true global offsets across all chunks.
    """
    audio = AudioSegment.from_file(file_path)
    total_duration_ms = len(audio)
    num_chunks = math.ceil(total_duration_ms / chunk_length_ms)

    combined_segments = []
    full_transcript_text = []

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_duration_ms)
        chunk_offset_sec = start_ms / 1000.0

        chunk = audio[start_ms:end_ms]
        chunk_path = f"chunks/chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")

        with open(chunk_path, "rb") as f:
            params = {
                "model": "whisper-1",
                "file": f,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"]
            }
            if prompt:
                params["prompt"] = prompt
            response = client.audio.transcriptions.create(**params)

        full_transcript_text.append(response.text)

        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                seg_dict = seg if isinstance(seg, dict) else seg.model_dump()
                combined_segments.append({
                    "start": round(seg_dict["start"] + chunk_offset_sec, 2),
                    "end": round(seg_dict["end"] + chunk_offset_sec, 2),
                    "text": seg_dict["text"].strip()
                })

    return {
        "text": " ".join(full_transcript_text),
        "segments": combined_segments
    }


# 3. Export Helpers
def format_timestamp_srt(seconds):
    """Converts seconds to SRT format: HH:MM:SS,mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_text(seconds):
    """Converts seconds to [HH:MM:SS] format"""
    total_seconds = int(seconds)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"[{hours:02d}:{mins:02d}:{secs:02d}]"


def export_transcriptions(result, base_path="exports/transcription"):
    """Exports results to JSON, TXT, and SRT."""
    # JSON
    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    # TXT
    with open(f"{base_path}.txt", "w", encoding="utf-8") as f:
        f.write("--- TRANSCRIPTION WITH TIMESTAMPS ---\n\n")
        for seg in result["segments"]:
            f.write(f"{format_timestamp_text(seg['start'])} - {format_timestamp_text(seg['end'])} {seg['text']}\n")

    # SRT
    with open(f"{base_path}.srt", "w", encoding="utf-8") as f:
        for idx, seg in enumerate(result["segments"], start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


if __name__ == "__main__":
    audio_file = "audio/audioforw1d5lab.mp3"
    print("Executing full transcription and export pipeline...")
    pipeline_result = split_and_transcribe_audio(audio_file, chunk_length_ms=60000)
    export_transcriptions(pipeline_result)
    print("Pipeline finished successfully. Output files saved in exports/")