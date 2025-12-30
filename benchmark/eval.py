"""Evaluation metrics for caption alignment quality: DER, JER, WER, and SCA."""

from pathlib import Path
from typing import List, Union

import jiwer
import pysubs2
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from speaker_count_metrics import SpeakerCountAccuracy, SpeakerCountingErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()


def caption_to_annotation(caption: pysubs2.SSAFile, uri: str = "default") -> Annotation:
    """Convert caption to pyannote Annotation for diarization metrics."""
    annotation = Annotation(uri=uri)

    speaker = None
    for event in caption.events:
        segment = Segment(event.start / 1000.0, event.end / 1000.0)
        if event.name:
            event.name = event.name.rstrip(":").lstrip(">").strip()
            speaker = event.name

        annotation[segment] = event.name or speaker

    return annotation


def caption_to_text(
    caption: pysubs2.SSAFile,
) -> str:
    """Convert caption to text string for WER calculation."""
    text = " ".join(
        [
            english_normalizer(event.text.replace("...", " ").strip()).replace("chatgpt", "chat gpt")
            for event in caption.events
        ]
    )
    return text


def evaluate_alignment(
    reference_file: Union[str, Path],
    hypothesis_file: Union[str, Path],
    metrics: List[str] = ["der", "jer", "wer", "sca", "scer"],
    collar: float = 0.0,
    skip_overlap: bool = False,
    verbose: bool = False,
) -> dict:
    """Evaluate alignment quality using specified metrics.

    Args:
        reference_file: Path to reference caption file
        hypothesis_file: Path to hypothesis caption file
        metrics: List of metrics to compute (der, jer, wer, sca, scer)
        collar: Collar size in seconds for diarization metrics
        skip_overlap: Skip overlapping speech regions for DER

    Returns:
        Dictionary mapping metric names to values
    """
    reference = pysubs2.load(reference_file)
    hypothesis = pysubs2.load(hypothesis_file)

    ref_ann = caption_to_annotation(reference)
    hyp_ann = caption_to_annotation(hypothesis)
    ref_text = caption_to_text(reference)
    hyp_text = caption_to_text(hypothesis)

    if False:
        with open(hypothesis_file[:-4] + ".txt", "w") as f:
            words = hyp_text.split()
            for word in words:
                f.write(word + "\n")

    # Perform detailed text alignment analysis
    if verbose:  # Enable for debugging alignment issues
        from kaldialign import align as kaldi_align

        ref_timelines = [(event.start / 1000.0, event.end / 1000.0) for event in reference.events]
        hyp_timelines = [(event.start / 1000.0, event.end / 1000.0) for event in hypothesis.events]
        ref_sentences = [event.text for event in reference.events]
        hyp_sentences = [event.text for event in hypothesis.events]

        sent_symbol = "❅"
        eps_symbol = "-"
        alignments = kaldi_align(
            sent_symbol.join(ref_sentences), sent_symbol.join(hyp_sentences), eps_symbol, sclite_mode=True
        )

        idx = 0
        rstart, hstart = 0, 0
        rend, hend = 0, 0
        for k, ali in enumerate(alignments):
            ref_sym, hyp_sym = ali
            if ref_sym == sent_symbol:
                rend += 1
            if hyp_sym == sent_symbol:
                hend += 1

            if ref_sym == sent_symbol and hyp_sym == sent_symbol:
                isdiff = any(_ali[0].lower() != _ali[1].lower() for _ali in alignments[idx:k])
                if isdiff:
                    # fmt: off
                    print(f"[{ref_timelines[rstart][0]:.2f}, {ref_timelines[rend - 1][1]:.2f}] REF: {''.join(_ali[0] for _ali in alignments[idx:k])}")  # noqa: E501
                    print(f"[{hyp_timelines[hstart][0]:.2f}, {hyp_timelines[hend - 1][1]:.2f}] HYP: {''.join(_ali[1] for _ali in alignments[idx:k])}\n")  # noqa: E501
                    # fmt: on

                idx = k + 1
                rstart = rend
                hstart = hend
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
        description="Evaluate caption alignment quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py -r ref.ass -hyp hyp.ass
  python eval.py -r ref.ass -hyp hyp.ass -m wer
  python eval.py -r ref.ass -hyp hyp.ass -m der jer sca -c 0.25
  python eval.py -r ref.ass -hyp hyp.ass -f json
        """,
    )

    parser.add_argument("--reference", "-r", required=True, help="Reference caption file")
    parser.add_argument("--hypothesis", "-hyp", required=True, help="Hypothesis caption file")
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
        verbose=args.verbose,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        # Extract detailed DER if present
        for metric, value in results.items():
            if not isinstance(value, float):
                assert metric == "der", f"Detailed output only supported for DER, got: {metric}"

                model_display = args.model_name if args.model_name else "-"
                print("\nDetailed DER Components:")

                # Build header and values with custom order
                sorted_items = sorted(value.items(), key=lambda x: x[0])

                # Define the desired column order
                column_order = [
                    "diarization error rate",
                    "false alarm",
                    "missed detection",
                    "confusion",
                    "correct",
                    "total",
                ]

                # Reorder items according to column_order
                ordered_items = []
                value_dict = dict(sorted_items)
                for key in column_order:
                    if key in value_dict:
                        ordered_items.append((key, value_dict[key]))

                header = ["Model"] + [
                    "DER" if key == "diarization error rate" else f"{key} (s)" for key, _ in ordered_items
                ]
                values = [model_display] + [f"{val:.4f}" for _, val in ordered_items]

                # Print table
                print("Metric Details:")
                print("| " + " | ".join(header) + " |")
                print("|" + "|".join(["--------"] * len(header)) + "|")
                print("| " + " | ".join(values) + " |")
                print()

                value = value["diarization error rate"]
                results[metric] = value

        # Display in markdown-friendly format
        metric_names = ["Model"]
        metric_values = [args.model_name if args.model_name else "-"]
        for metric, value in results.items():
            arrow = "↓" if metric.lower() in ["der", "jer", "wer", "scer"] else "↑"
            metric_names.append(f"{metric.upper()} {arrow}")
            metric_values.append(f"{value:.4f} ({value * 100:5.2f}%)")

        print("| " + " | ".join(metric_names) + " |")
        print("|" + "|".join(["--------"] * len(metric_names)) + "|")
        print("| " + " | ".join(metric_values) + " |")


if __name__ == "__main__":
    main()
