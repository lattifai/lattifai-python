# Benchmark Evaluation

Evaluate subtitle alignment quality using DER, JER, WER, SCA, and SCER metrics.

## Metrics

**DER (Diarization Error Rate)** - Speaker diarization quality
- Formula: `(False Alarm + Missed Speech + Speaker Error) / Total Speech Time`
- Lower is better, < 20% is good

**JER (Jaccard Error Rate)** - Temporal overlap of speaker segments
- Formula: `1 - (Intersection / Union)`
- Lower is better

**WER (Word Error Rate)** - Transcription accuracy
- Formula: `(Substitutions + Deletions + Insertions) / Total Words`
- Lower is better, < 10% is good

**SCA (Speaker Count Accuracy)** - Speaker counting accuracy
- Formula: `Correct Predictions / Total Samples`
- Higher is better

**SCER (Speaker Counting Error Rate)** - Relative error in speaker count
- Formula: `|Hypothesis Count - Reference Count| / Reference Count`
- Lower is better

## Benchmarks

| Model | DER ↓ | JER ↓ | WER ↓ | SCA ↑ | SCER ↓ |
|--------|--------|--------|--------|--------|--------|
| Ground Truth | 0.0000 (0.00%) | 0.0000 (0.00%) | 0.0000 (0.00%) | 1.0000 (100.00%) | 0.0000 (0.00%) |
| Gemini 2.5 Pro | 0.6303 (63.03%) | 0.6532 (65.32%) | 0.1851 (18.51%) | 1.0000 (100.00%) | 0.0000 (0.00%) |
| Gemini 2.5 Pro + LattifAI | 0.2280 (22.80%) | 0.3226 (32.26%) | 0.1867 (18.67%) | 1.0000 (100.00%) | 0.0000 (0.00%) |
| Gemini 3 Pro Preview | 0.6433 (64.33%) | 0.6637 (66.37%) | 0.0861 (8.61%) | 1.0000 (100.00%) | 0.0000 (0.00%) |
| Gemini 3 Pro Preview + LattifAI | 0.2177 (21.77%) | 0.3601 (36.01%) | 0.0876 (8.76%) | 1.0000 (100.00%) | 0.0000 (0.00%) |

**Command to reproduce:**
```bash
# Ground Truth vs Ground Truth
python eval.py -r data/Introducing_GPT-4o.ass -hyp data/Introducing_GPT-4o.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Ground Truth"

# Gemini 2.5 Pro
python eval.py -r data/Introducing_GPT-4o.ass -hyp data/Introducing_GPT-4o_Gemini.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 2.5 Pro"

# Gemini 2.5 Pro + LattifAI alignment
lai alignment youtube \
    https://www.youtube.com/watch\?v\=DQacCB9tDaw \
    media.output_dir=~/Downloads/lattifai_youtube \
    subtitle.use_transcription=true \
    transcription.model_name=gemini-2.5-pro \
    subtitle.split_sentence=true subtitle.normalize_text=true subtitle.include_speaker_in_text=false \
    subtitle.input_path=./data/Introducing_GPT-4o_Gemini.md \
    subtitle.output_path=./data/Introducing_GPT-4o_Gemini_LattifAI.ass transcription.gemini_api_key="YOUR_GEMINI_API_KEY"

python eval.py -r data/Introducing_GPT-4o.ass -hyp data/Introducing_GPT-4o_Gemini_LattifAI.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 2.5 Pro + LattifAI"

# Gemini 3 Pro + LattifAI alignment
lai alignment youtube \
    https://www.youtube.com/watch\?v\=DQacCB9tDaw \
    media.output_dir=~/Downloads/lattifai_youtube_Gemini3 \
    subtitle.split_sentence=true subtitle.normalize_text=true subtitle.include_speaker_in_text=false \
    subtitle.use_transcription=true \
    transcription.model_name=gemini-3-pro-preview \
    subtitle.output_path=./data/Introducing_GPT-4o_Gemini3_LattifAI.ass transcription.gemini_api_key="YOUR_GEMINI_API_KEY"


python eval.py -r data/Introducing_GPT-4o.ass -hyp data/Introducing_GPT-4o_Gemini3.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Pro Preview"
python eval.py -r data/Introducing_GPT-4o.ass -hyp data/Introducing_GPT-4o_Gemini3_LattifAI.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Pro Preview + LattifAI"
```

## Installation

```bash
pip install pysubs2 pyannote.core pyannote.metrics jiwer
```

These dependencies are included with the `lattifai-python` package.

## Usage

```bash
# With model name
python eval.py -r reference.ass -hyp hypothesis.ass -n "MyModel"
```

## Output Format

Default output is a markdown table:
```
| Model | DER ↓ | JER ↓ | WER ↓ | SCA ↑ | SCER ↓ |
|-------|-------|-------|-------|-------|-------|
| -     | 0.1523 (15.23%) | 0.2045 (20.45%) | 0.1012 (10.12%) | 0.8500 (85.00%) | 0.1200 (12.00%) |
```

## Supported Formats

Formats with speaker name field: `.ass`, `.ssa`

## Notes

- DER/JER require speaker labels in subtitle files (Name field)
- WER only requires text transcription
- SCA/SCER require speaker labels
- Collar adds temporal forgiveness zones
- Lower is better for DER/JER/WER/SCER, higher is better for SCA

## References

- pyannote.metrics: https://pyannote.github.io/pyannote-metrics/
- OpenBench: https://github.com/argmaxinc/OpenBench
- jiwer: https://github.com/jitsi/jiwer
