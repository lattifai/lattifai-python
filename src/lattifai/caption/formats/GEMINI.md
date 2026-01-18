# Gemini Transcript Reader/Writer (YouTube)

## Transcript Format

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
from lattifai.caption import GeminiReader

segments = GeminiReader.read(
	'videoid_Gemini.md',
	include_events=True,
	include_sections=True,
)
for seg in segments:
	print(seg.segment_type, seg.speaker, seg.section, seg.timestamp, seg.text)
```

### 2. Extract Dialogue for Alignment

```python
from lattifai.caption import GeminiReader
supervisions = GeminiReader.extract_for_alignment(
	'videoid_Gemini.md',
	merge_consecutive=False,
	min_duration=0.1,
)
```


## API Reference

### GeminiReader

`read(path, include_events=False, include_sections=False)` → List[GeminiSegment]
`extract_for_alignment(path, merge_consecutive=True, min_duration=0.1)` → List[Supervision]

### GeminiWriter

`update_timestamps(original_transcript, aligned_supervisions, output_path, timestamp_mapping=None)` → Path
`write_aligned_transcript(aligned_supervisions, output_path, include_word_timestamps=False)` → Path

### GeminiSegment

* `text`, `timestamp`, `speaker`, `section`, `segment_type`, `line_number`
