# Benchmark Evaluation

Evaluate subtitle alignment quality using DER, JER, WER, SCA, and SCER metrics.

## 📊 Interactive Visualization

**[View Interactive Benchmark Results](https://lattifai.github.io/lattifai-python/benchmark/)**

Open `index.html` in your browser for an interactive visualization with:
- Click-to-highlight metric comparison
- Detailed metric explanations
- Key performance insights
- Responsive design for all devices

```bash
# Open directly in browser (may have CORS issues with file://)
open benchmark/index.html

# Or start a local HTTP server (recommended)
cd benchmark
python -m http.server 8002
# Then open: http://localhost:8002/index.html

# Alternative with Python 2
python -m SimpleHTTPServer 8002
```

## Metrics

**DER (Diarization Error Rate)** - Speaker diarization quality
- Formula: $\text{DER} = \frac{\text{false alarm} + \text{missed detection} + \text{confusion}}{\text{total}}$
- Components measured in seconds: false alarm, missed detection, confusion, correct, total
- Lower is better, < 20% is good

**JER (Jaccard Error Rate)** - Temporal overlap of speaker segments
- Formula: $\text{JER} = 1 - \frac{\text{Intersection}}{\text{Union}}$
- Lower is better

**WER (Word Error Rate)** - Transcription accuracy
- Formula: $\text{WER} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Words}}$
- Lower is better, < 10% is good

**SCA (Speaker Count Accuracy)** - Speaker counting accuracy
- Formula: $\text{SCA} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$
- Higher is better

**SCER (Speaker Counting Error Rate)** - Relative error in speaker count
- Formula: $\text{SCER} = \frac{|\text{Hypothesis Count} - \text{Reference Count}|}{\text{Reference Count}}$
- Lower is better

## Benchmarks

| Model | DER ↓ | JER ↓ | WER ↓ | SCA ↑ | SCER ↓ |
|--------|--------|--------|--------|--------|--------|
| Ground Truth | 0.0000 (0.00%) | 0.0000 (0.00%) | 0.0000 (0.00%) | 1.0000 (100.00%) | 0.0000 (0.00%) |
| Gemini 2.5 Pro | 0.3610 (36.10%) | 0.3917 (39.17%) | 0.1511 (15.11%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |
| Gemini 2.5 Pro + LattifAI | 0.2252 (22.52%) | 0.3160 (31.60%) | 0.1511 (15.11%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |
| Gemini 3 Pro Preview | 0.5343 (53.43%) | 0.5679 (56.79%) | 0.0494 ( 4.94%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |
| Gemini 3 Pro Preview + LattifAI | 0.2265 (22.65%) | 0.3656 (36.56%) | 0.0494 ( 4.94%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |
| Gemini 3 Flash Preview | 0.3044 (30.44%) | 0.3042 (30.42%) | 0.0454 ( 4.54%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |
| Gemini 3 Flash Preview + LattifAI | 0.1142 (11.42%) | 0.1363 (13.63%) | 0.0454 ( 4.54%) | 1.0000 (100.00%) | 0.0000 ( 0.00%) |

**Command to reproduce:**
```bash
cd benchmark

# set .env

# Ground Truth vs Ground Truth
python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_GroundTruth.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Ground Truth"

# Gemini 2.5 Pro + LattifAI alignment
# Note: caption.split_sentence=true may cause slight WER increase due to text normalization
# (e.g., "ChatGPT" may be split into "Chat GPT")
lai alignment youtube -Y \
    https://www.youtube.com/watch\?v\=DQacCB9tDaw \
    media.output_dir=~/Downloads/lattifai_youtube client.profile=true \
    caption.include_speaker_in_text=false caption.split_sentence=true \
    caption.input_path=./data/Introducing_GPT-4o_Gemini.md \
    caption.output_path=./data/Introducing_GPT-4o_Gemini_LattifAI.ass \
    transcription.model_name=gemini-2.5-pro \
    transcription.gemini_api_key="YOUR_GEMINI_API_KEY"

# Gemini 2.5 Pro
lai caption convert -Y ./data/Introducing_GPT-4o_Gemini.md ./data/Introducing_GPT-4o_Gemini.ass include_speaker_in_text=false

python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 2.5 Pro"
python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini_LattifAI.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 2.5 Pro + LattifAI"

# Gemini 3 Pro + LattifAI alignment
lai alignment youtube -Y \
    https://www.youtube.com/watch\?v\=DQacCB9tDaw \
    media.output_dir=~/Downloads/lattifai_youtube_Gemini3 client.profile=true \
    caption.include_speaker_in_text=false caption.split_sentence=true \
    caption.input_path=./data/Introducing_GPT-4o_Gemini3.md \
    caption.output_path=./data/Introducing_GPT-4o_Gemini3_LattifAI.ass \
    transcription.model_name=gemini-3-pro-preview \
    transcription.gemini_api_key="YOUR_GEMINI_API_KEY"

lai caption convert -Y ./data/Introducing_GPT-4o_Gemini3.md ./data/Introducing_GPT-4o_Gemini3.ass include_speaker_in_text=false

python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini3.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Pro Preview"
python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini3_LattifAI.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Pro Preview + LattifAI"


# Gemini 3 Flash + LattifAI alignment
lai alignment youtube -Y \
    https://www.youtube.com/watch\?v\=DQacCB9tDaw \
    media.output_dir=~/Downloads/lattifai_youtube_Gemini3 client.profile=true \
    caption.include_speaker_in_text=false caption.split_sentence=true \
    caption.input_path=./data/Introducing_GPT-4o_Gemini3_Flash.md \
    caption.output_path=./data/Introducing_GPT-4o_Gemini3_Flash_LattifAI.ass \
    transcription.model_name=gemini-3-flash-preview \
    transcription.gemini_api_key="YOUR_GEMINI_API_KEY"
lai caption convert -Y ./data/Introducing_GPT-4o_Gemini3_Flash.md ./data/Introducing_GPT-4o_Gemini3_Flash.ass include_speaker_in_text=false

python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini3_Flash.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Flash Preview"
python eval.py -r data/Introducing_GPT-4o_GroundTruth.ass -hyp data/Introducing_GPT-4o_Gemini3_Flash_LattifAI.ass \
  --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 3 Flash Preview + LattifAI"

# python eval.py -r data/Introducing_GPT-4o_Gemini.ass -hyp data/Introducing_GPT-4o_Gemini3.ass \
#   --metrics der jer wer sca scer --collar 0.0 --model-name "Gemini 2.5 vs 3"
```

## Installation

```bash
pip install pysubs2 pyannote.core pyannote.metrics jiwer
pip install whisper-normalizer
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
