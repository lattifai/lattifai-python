"""Evaluation metrics for subtitle alignment quality: DER, JER, WER, and SCA."""

from pathlib import Path
from typing import List, Union

import jiwer
import pysubs2
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from speaker_count_metrics import SpeakerCountAccuracy, SpeakerCountingErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()


def subtitle_to_annotation(subtitle: pysubs2.SSAFile, uri: str = "default") -> Annotation:
    """Convert subtitle to pyannote Annotation for diarization metrics."""
    annotation = Annotation(uri=uri)

    speaker = None
    for event in subtitle.events:
        segment = Segment(event.start / 1000.0, event.end / 1000.0)
        if event.name:
            event.name = event.name.rstrip(":").lstrip(">").strip()
            speaker = event.name

        annotation[segment] = event.name or speaker

    return annotation


def subtitle_to_text(
    subtitle: pysubs2.SSAFile,
) -> str:
    """Convert subtitle to text string for WER calculation."""
    text = " ".join([english_normalizer(event.text.replace("...", " ").strip()) for event in subtitle.events])
    return text


def evaluate_alignment(
    reference_file: Union[str, Path],
    hypothesis_file: Union[str, Path],
    metrics: List[str] = ["der", "jer", "wer", "sca", "scer"],
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict:
    """Evaluate alignment quality using specified metrics.

    Args:
        reference_file: Path to reference subtitle file
        hypothesis_file: Path to hypothesis subtitle file
        metrics: List of metrics to compute (der, jer, wer, sca, scer)
        collar: Collar size in seconds for diarization metrics
        skip_overlap: Skip overlapping speech regions for DER

    Returns:
        Dictionary mapping metric names to values
    """
    reference = pysubs2.load(reference_file)
    hypothesis = pysubs2.load(hypothesis_file)

    ref_ann = subtitle_to_annotation(reference)
    hyp_ann = subtitle_to_annotation(hypothesis)
    ref_text = subtitle_to_text(reference)
    hyp_text = subtitle_to_text(hypothesis)

    if False:
        with open(hypothesis_file[:-4] + ".txt", "w") as f:
            words = hyp_text.split()
            for word in words:
                f.write(word + "\n")

    # Perform detailed text alignment analysis
    if False:  # Enable for debugging alignment issues
        from kaldialign import align as kaldi_align

        ali = kaldi_align(ref_text, hyp_text, "*", sclite_mode=True)

        print("\n=== Text Alignment Diff ===\n")

        # Split alignment into segments by spaces
        segments = []
        current_segment = []

        for ref_token, hyp_token in ali:
            if ref_token == " " and hyp_token == " ":
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append((ref_token, hyp_token))

        if current_segment:
            segments.append(current_segment)

        # Print diff for each segment with context
        context_size = 5  # Number of words before and after
        for i, segment in enumerate(segments):
            ref_word = "".join(r for r, h in segment)
            hyp_word = "".join(h for r, h in segment)

            if ref_word != hyp_word:
                # Get context words
                start_idx = max(0, i - context_size)
                end_idx = min(len(segments), i + context_size + 1)

                # Build context - need to get both ref and hyp for accurate display
                context_before_ref = []
                context_before_hyp = []
                for j in range(start_idx, i):
                    ref_w = "".join(r for r, h in segments[j] if r != "*")
                    hyp_w = "".join(h for r, h in segments[j] if h != "*")
                    context_before_ref.append(ref_w)
                    context_before_hyp.append(hyp_w)

                context_after_ref = []
                context_after_hyp = []
                for j in range(i + 1, end_idx):
                    ref_w = "".join(r for r, h in segments[j] if r != "*")
                    hyp_w = "".join(h for r, h in segments[j] if h != "*")
                    context_after_ref.append(ref_w)
                    context_after_hyp.append(hyp_w)

                # Print with context
                ref_before_str = " ".join(context_before_ref)
                ref_after_str = " ".join(context_after_ref)
                hyp_before_str = " ".join(context_before_hyp)
                hyp_after_str = " ".join(context_after_hyp)

                print(f"REF: ...{ref_before_str} [{ref_word}] {ref_after_str}...")
                print(f"HYP: ...{hyp_before_str} [{hyp_word}] {hyp_after_str}...")
                print("---\n")

        print("\n" + "=" * 50 + "\n")

    results = {}

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower == "der":
            der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
            results["der"] = der_metric(ref_ann, hyp_ann, detailed=True, uem=None)
        elif metric_lower == "jer":
            jer_metric = JaccardErrorRate(collar=collar)
            results["jer"] = jer_metric(ref_ann, hyp_ann)
        elif metric_lower == "wer":
            results["wer"] = jiwer.wer(ref_text, hyp_text)
        elif metric_lower == "sca":
            sca_metric = SpeakerCountAccuracy()
            results["sca"] = sca_metric(ref_ann, hyp_ann)
        elif metric_lower == "scer":
            scer_metric = SpeakerCountingErrorRate()
            results["scer"] = scer_metric(ref_ann, hyp_ann)
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported: der, jer, wer, sca, scer")

    return results


def main():
    """CLI for evaluation metrics."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate subtitle alignment quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py -r ref.ass -hyp hyp.ass
  python eval.py -r ref.ass -hyp hyp.ass -m wer
  python eval.py -r ref.ass -hyp hyp.ass -m der jer sca -c 0.25
  python eval.py -r ref.ass -hyp hyp.ass -f json
        """,
    )

    parser.add_argument("--reference", "-r", required=True, help="Reference subtitle file")
    parser.add_argument("--hypothesis", "-hyp", required=True, help="Hypothesis subtitle file")
    parser.add_argument("--model-name", "--model_name", "-n", default="", help="Model name to display in results")
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["der", "jer", "wer", "sca", "scer"],
        choices=["der", "jer", "wer", "sca", "scer"],
        help="Metrics to compute",
    )
    parser.add_argument("--collar", "-c", type=float, default=0.0, help="Collar size in seconds")
    parser.add_argument("--skip-overlap", action="store_true", help="Skip overlapping speech for DER")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.hypothesis).exists():
        print(f"Error: Hypothesis file not found: {args.hypothesis}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reference: {args.reference}", file=sys.stderr)
        print(f"Hypothesis: {args.hypothesis}", file=sys.stderr)
        print(f"Metrics: {', '.join(args.metrics)}", file=sys.stderr)
        print(f"Collar: {args.collar}s\n", file=sys.stderr)

    results = evaluate_alignment(
        reference_file=args.reference,
        hypothesis_file=args.hypothesis,
        metrics=args.metrics,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        # Extract detailed DER if present
        for metric, value in results.items():
            if not isinstance(value, float):
                assert metric == "der", f"Detailed output only supported for DER, got: {metric}"
                if args.verbose:
                    print("Detailed DER results:")
                    for key, val in value.items():
                        print(f"  {key:24s}: {val:12.4f}")
                    print()
                value = value["diarization error rate"]
                results[metric] = value

        # Display in markdown-friendly format
        metric_names = ["Model"]
        metric_values = [args.model_name if args.model_name else "-"]
        for metric, value in results.items():
            arrow = "↓" if metric.lower() in ["der", "jer", "wer", "scer"] else "↑"
            metric_names.append(f"{metric.upper()} {arrow}")
            metric_values.append(f"{value:.4f} ({value:.2%})")

        print("| " + " | ".join(metric_names) + " |")
        print("|" + "|".join(["--------"] * len(metric_names)) + "|")
        print("| " + " | ".join(metric_values) + " |")


if __name__ == "__main__":
    main()
