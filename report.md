# Whisper Speech-to-Text Implementation Report

## 1. Executive Summary
This project evaluated OpenAI's Whisper API (`whisper-1`) for long-form speech transcription, chunking pipelines, and multi-format exports (JSON, human-readable TXT, and SRT subtitles). A modular pipeline was established using Python, `pydub`, and `ffmpeg` to overcome API file limits and deliver timestamped transcripts.

---

## 2. Prompted vs. Unguided Transcription Analysis
Whisper was evaluated with both unprompted (zero-shot) and guided (prompt-informed) configurations:
* **Unguided Transcription:** Transcribes phonetically and relies entirely on acoustic modeling and general language statistics. While conversational speech is recognized with high fidelity, domain-specific names and technical terminology are frequently approximated or spelled with standard phonetics.
* **Guided Transcription:** By supplying a contextual prompt containing entities (e.g., proper nouns, acronyms, speaker names), the model shifts its prior distribution. This noticeably improved spelling consistency and technical vocabulary precision without altering the speech flow.
* **Key Takeaway:** Prompts in Whisper should be treated as dynamic glossaries or style guides rather than instructional system prompts.

---

## 3. Benefits of Audio Chunking
The Whisper API maintains an upload ceiling of 25 MB per request:
* **Bypassing Payload Limits:** Chunking allows arbitrary lengths (such as 2+ hour recordings) to be processed cleanly.
* **Fault Tolerance & Granularity:** Splitting files into manageable segments (60–600 seconds) prevents complete re-runs if an individual request encounters a network failure or rate limit.
* **Timestamp Normalization:** Segment-level timestamps returned by `verbose_json` represent offsets relative to each chunk. Adding `chunk_start_offset` dynamically reconstructs continuous global timestamps across the entire source recording.

---

## 4. Challenges & Engineering Solutions
* **Path Resolution for Audio Binaries:** On macOS Apple Silicon and Conda environments, `pydub` was unable to resolve `ffprobe` by default. This was solved by configuring system PATH overrides and assigning explicit paths to `AudioSegment.converter` and `AudioSegment.ffprobe`.
* **Empty File Downloads:** Browser handling initially produced empty or preview-locked files. Using direct stream requests (`curl`) verified via byte-size assertions resolved corrupted file inputs.
* **Timestamp Continuity:** Splicing required strict math calculations to ensure subtitle (`SRT`) millisecond formatting and block numbers matched video playback standards.

---

## 5. Recommendations for Production Optimization
1. **Silence-Aware Slicing:** Replace fixed duration chunking with silence detection (`pydub.silence.split_on_silence`) to prevent slicing mid-word across boundary points.
2. **Dynamic Context Windows:** Carry over the trailing 200–300 characters of the preceding chunk's transcription into the `prompt` parameter of the subsequent chunk to preserve conversational continuity.
3. **Async Batch Processing:** Implement asynchronous worker pools (`asyncio` / concurrent tasks) with rate-limit retries to transcribe non-dependent chunks in parallel.