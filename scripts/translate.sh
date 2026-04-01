#!/usr/bin/env bash
# End-to-end test: YouTube download → caption convert (with reference alignment) → translate
# Video: Terence Tao – How the world's top mathematician uses AI
# URL: https://www.youtube.com/watch?v=Q8Fkpi18QXU
#
# Usage:
#   ./scripts/translate.sh                              # run all steps
#   ./scripts/translate.sh --step 2                     # run only step 2
#   ./scripts/translate.sh --step 2 4                   # run steps 2 and 4
#   ./scripts/translate.sh --step 3-5                   # run steps 3, 4, 5
#   ./scripts/translate.sh --outdir /tmp/output           # custom output directory
#   ./scripts/translate.sh --model gemini-2.5-flash     # use a specific model
#   ./scripts/translate.sh --model gemini-2.5-flash --step 4
set -euo pipefail

cd "$(dirname "$0")/.."

VIDEO_ID="Q8Fkpi18QXU"
VIDEO_URL="https://www.youtube.com/watch?v=${VIDEO_ID}"
DEBUG_DIR=./debug/youtube
OUT_DIR=./debug/youtube/translate

# ── Parse arguments ───────────────────────────────────
MODEL="gemini-3-flash-preview"
CUSTOM_OUTDIR=""
STEPS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir|-o)
            CUSTOM_OUTDIR="$2"; shift 2 ;;
        --model|-m)
            MODEL="$2"; shift 2 ;;
        --step)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                if [[ "$1" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do
                        STEPS+=("$i")
                    done
                else
                    STEPS+=("$1")
                fi
                shift
            done
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
# Apply custom output directory if provided
if [[ -n "$CUSTOM_OUTDIR" ]]; then
    OUT_DIR="$CUSTOM_OUTDIR"
fi

mkdir -p "$OUT_DIR"

# Build model argument (empty = omit, let default_factory handle config.toml)
MODEL_ARG=()
if [[ -n "$MODEL" ]]; then
    MODEL_ARG=(translation.llm.model_name="$MODEL")
fi

echo "Model:  ${MODEL:-<config.toml>}"
echo "Output: $OUT_DIR"

should_run() {
    [[ ${#STEPS[@]} -eq 0 ]] && return 0  # no --step = run all
    local s
    for s in "${STEPS[@]}"; do
        [[ "$s" == "$1" ]] && return 0
    done
    return 1
}

# ────────────────────────────────────────────────────────
# Step 1: Download media + caption (skip if already exist)
# ────────────────────────────────────────────────────────
if should_run 1; then
echo "═══ Step 1: Download media & captions ═══"

if [[ ! -f "$DEBUG_DIR/${VIDEO_ID}.mp4" ]]; then
    lai youtube download -Y "$VIDEO_URL" \
        only=media \
        media.output_dir="$DEBUG_DIR" \
        media.output_format=mp4
else
    echo "  ✓ Media already exists: $DEBUG_DIR/${VIDEO_ID}.mp4"
fi

if [[ ! -f "$DEBUG_DIR/${VIDEO_ID}.en.vtt" ]]; then
    lai youtube download -Y "$VIDEO_URL" \
        only=caption \
        media.output_dir="$DEBUG_DIR"
else
    echo "  ✓ Caption already exists: $DEBUG_DIR/${VIDEO_ID}.en.vtt"
fi
fi

# ────────────────────────────────────────────────────────
# Step 2: Convert transcript.md → SRT with reference alignment
#   - transcript.md has speaker labels + chapter timestamps
#   - en.vtt has accurate timestamps from YouTube ASR
#   - Result: SRT with accurate timing + speaker labels
# ────────────────────────────────────────────────────────
if should_run 2; then
echo ""
echo "═══ Step 2: Caption convert with reference alignment ═══"

lai caption convert -Y \
    "$DEBUG_DIR/${VIDEO_ID}.transcript.md" \
    "$OUT_DIR/${VIDEO_ID}.aligned.srt" \
    reference="$DEBUG_DIR/${VIDEO_ID}.en.vtt" \
    include_speaker_in_text=true

echo "  ✓ Aligned SRT: $OUT_DIR/${VIDEO_ID}.aligned.srt"
echo "  Preview:"
head -12 "$OUT_DIR/${VIDEO_ID}.aligned.srt"
fi

# ────────────────────────────────────────────────────────
# Step 3: Forced alignment + sentence splitting
#   - aligned.srt has text-matched timestamps (coarse)
#   - Lattice-1 model produces word-level precise timing
#   - split_sentence breaks long segments into natural sentences
# ────────────────────────────────────────────────────────
if should_run 3; then
echo ""
echo "═══ Step 3: Forced alignment & sentence splitting ═══"

lai alignment align -Y \
    "$DEBUG_DIR/${VIDEO_ID}.mp4" \
    "$OUT_DIR/${VIDEO_ID}.aligned.srt" \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.json" \
    caption.split_sentence=true \
    caption.word_level=true

echo "  ✓ Lattice-aligned JSON: $OUT_DIR/${VIDEO_ID}.LattifAI.json"

# Also export SRT for translate input
lai caption convert -Y \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.json" \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.srt" include_speaker_in_text=true

echo "  ✓ Lattice-aligned SRT: $OUT_DIR/${VIDEO_ID}.LattifAI.srt"
echo "  Preview:"
head -20 "$OUT_DIR/${VIDEO_ID}.LattifAI.srt"
fi

# ────────────────────────────────────────────────────────
# Step 4: Translate (quick mode → ASS bilingual output)
# ────────────────────────────────────────────────────────
if should_run 4; then
echo ""
echo "═══ Step 4: Translate to Chinese (quick mode → ASS) ═══"

lai translate caption -Y \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.srt" \
    "$OUT_DIR/${VIDEO_ID}.zh-quick.ass" \
    translation.mode=quick \
    translation.target_lang=zh \
    translation.bilingual=true \
    ${MODEL_ARG[@]+"${MODEL_ARG[@]}"}

echo "  ✓ Quick translation: $OUT_DIR/${VIDEO_ID}.zh-quick.ass"
echo "  Preview:"
head -30 "$OUT_DIR/${VIDEO_ID}.zh-quick.ass"
fi

# ────────────────────────────────────────────────────────
# Step 5: Translate (normal mode → ASS + artifacts)
# ────────────────────────────────────────────────────────
if should_run 5; then
echo ""
echo "═══ Step 5: Translate to Chinese (normal mode → ASS + artifacts) ═══"

lai translate caption -Y \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.srt" \
    "$OUT_DIR/${VIDEO_ID}.zh-normal.ass" \
    translation.mode=normal \
    translation.target_lang=zh \
    translation.bilingual=true \
    translation.save_artifacts=true \
    translation.artifacts_dir="$OUT_DIR/normal_artifacts" \
    translation.ask_refine_after_normal=false \
    ${MODEL_ARG[@]+"${MODEL_ARG[@]}"}

echo "  ✓ Normal translation: $OUT_DIR/${VIDEO_ID}.zh-normal.ass"
echo "  Artifacts:"
ls -la "$OUT_DIR/normal_artifacts/" 2>/dev/null || echo "  (no artifacts saved)"
fi

# ────────────────────────────────────────────────────────
# Step 6 (optional): Translate refined mode
# ────────────────────────────────────────────────────────
if should_run 6; then
echo ""
echo "═══ Step 6: Translate to Chinese (refined mode → ASS) ═══"

lai translate caption -Y \
    "$OUT_DIR/${VIDEO_ID}.LattifAI.srt" \
    "$OUT_DIR/${VIDEO_ID}.zh-refined.ass" \
    translation.mode=refined \
    translation.target_lang=zh \
    translation.bilingual=true \
    translation.save_artifacts=true \
    translation.artifacts_dir="$OUT_DIR/refined_artifacts" \
    translation.ask_refine_after_normal=false \
    ${MODEL_ARG[@]+"${MODEL_ARG[@]}"}

echo "  ✓ Refined translation: $OUT_DIR/${VIDEO_ID}.zh-refined.ass"
fi

# ────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────
echo ""
echo "═══ Summary ═══"
echo "  Input:   $DEBUG_DIR/${VIDEO_ID}.transcript.md (speaker labels + timestamps)"
echo "  Ref:     $DEBUG_DIR/${VIDEO_ID}.en.vtt (YouTube ASR timing)"
echo "  Aligned: $OUT_DIR/${VIDEO_ID}.aligned.srt (text-matched timestamps)"
echo "  Lattice: $OUT_DIR/${VIDEO_ID}.LattifAI.srt (forced alignment + sentence split)"
echo "  Quick:   $OUT_DIR/${VIDEO_ID}.zh-quick.ass (bilingual ASS)"
echo "  Normal:  $OUT_DIR/${VIDEO_ID}.zh-normal.ass (bilingual ASS)"
echo ""
echo "  All outputs in: $OUT_DIR/"
ls -lh "$OUT_DIR/"*.{srt,ass,json} 2>/dev/null
