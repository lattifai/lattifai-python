# Gemini Transcript Reader/Writer (YouTube)

## Transcript Format 示例

```
## [HH:MM:SS] Section Title

**Speaker Name:** Dialogue text [HH:MM:SS]

[Event description] [HH:MM:SS]

Continuation text [HH:MM:SS]
```

### Example

```
## [00:00:00] Introduction

[Music starts] [00:00:08]

**Mira Murati:** Hi everyone. [00:00:13]

Thank you for being here today. [00:00:19]

## [00:53:00] Announcing GPT-4o

**Mira Murati:** We are launching our new flagship model. [00:00:57]
```

## Usage

### 1. Read Transcript with Metadata

```python
from lattifai.io import GeminiReader

segments = GeminiReader.read(
	'transcript.txt',
	include_events=True,
	include_sections=True,
)
for seg in segments:
	print(seg.segment_type, seg.speaker, seg.section, seg.timestamp, seg.text)
```

### 2. Extract Dialogue for Alignment

```python
from lattifai.io import GeminiReader
supervisions = GeminiReader.extract_for_alignment(
	'transcript.txt',
	merge_consecutive=False,
	min_duration=0.1,
)
```

### 3. Perform Forced Alignment

```python
from lattifai import align
aligned = align(audio='video.wav', supervisions=supervisions, language='en')
```

### 4. Update Original Transcript

```python
from lattifai.io import GeminiWriter
GeminiWriter.update_timestamps(
	original_transcript='transcript.txt',
	aligned_supervisions=aligned,
	output_path='transcript_aligned.txt'
)
```

### 5. Write Simplified Aligned Transcript

```python
GeminiWriter.write_aligned_transcript(
	aligned_supervisions=aligned,
	output_path='transcript_simple.txt',
	include_word_timestamps=True,
)
```

## API Reference

### GeminiReader

`read(path, include_events=False, include_sections=False)` → List[TranscriptSegment]
`extract_for_alignment(path, merge_consecutive=True, min_duration=0.1)` → List[Supervision]

### GeminiWriter

`update_timestamps(original_transcript, aligned_supervisions, output_path, timestamp_mapping=None)` → Path
`write_aligned_transcript(aligned_supervisions, output_path, include_word_timestamps=False)` → Path

### TranscriptSegment

* `text`, `timestamp`, `speaker`, `section`, `segment_type`, `line_number`
