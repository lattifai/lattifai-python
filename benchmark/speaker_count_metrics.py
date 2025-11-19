# modified from openbench/metric/speaker_count_metrics.py

from dataclasses import dataclass

from pyannote.core import Annotation
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents


@dataclass
class SpeakerCounts:
    reference: int
    hypothesis: int

    # Returning properties as floats for aggregation purposes done internally by pyannote BaseMetric
    @property
    def absolute_error(self) -> float:
        return float(abs(self.hypothesis - self.reference))

    @property
    def is_correct(self) -> float:
        return float(self.reference == self.hypothesis)


class BaseSpeakerCountMetric(BaseMetric):
    """Base class for speaker counting metrics."""

    def __init__(self, **kwargs):
        super().__init__(instantaneous=True)

    def _supports_paired_evaluation(self) -> bool:
        return True

    def _get_speaker_counts(self, reference: Annotation, hypothesis: Annotation) -> SpeakerCounts:
        return SpeakerCounts(reference=len(reference.labels()), hypothesis=len(hypothesis.labels()))


class SpeakerCountingErrorRate(BaseSpeakerCountMetric):
    """Speaker Counting Error Rate (SCER)

    Measures the relative error in predicted number of speakers.

    SCER = |hypothesis_count - reference_count| / reference_count

    A value of 0.0 indicates perfect prediction, while higher values indicate
    larger relative errors in speaker count prediction.

    The global value for this metric is computed by the aggregation of its components.
    I.e. the global SCER is -> |sum(hypothesis_count) - sum(reference_count)| / sum(reference_count)
    """

    @classmethod
    def metric_name(cls) -> str:
        return "speaker counting error rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["reference_speaker_count", "hypothesis_speaker_count"]

    def compute_components(self, reference: Annotation, hypothesis: Annotation, **kwargs) -> Details:
        counts = self._get_speaker_counts(reference, hypothesis)
        return {
            "reference_speaker_count": counts.reference,
            "hypothesis_speaker_count": counts.hypothesis,
        }

    def compute_metric(self, detail: Details) -> float:
        reference_count = detail["reference_speaker_count"]
        if reference_count == 0:
            return 0.0
        return abs(detail["hypothesis_speaker_count"] - reference_count) / reference_count


class SpeakerCountMeanAbsoluteError(BaseSpeakerCountMetric):
    """Speaker Count Mean Absolute Error (SCMAE)

    Measures the absolute difference between predicted and actual speaker counts.

    SCMAE = |reference_count - hypothesis_count| / 1

    A value of 0 indicates perfect prediction, while higher values indicate
    larger absolute errors in speaker count prediction.

    The global value for this metric is computed by the aggregation of its components.
    I.e. the global SCMAE is -> sum(|hypothesis_count - reference_count|) / total_samples
    """

    @classmethod
    def metric_name(cls) -> str:
        return "speaker count mean absolute error"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["absolute_error", "total_samples"]

    def compute_components(self, reference: Annotation, hypothesis: Annotation, **kwargs) -> Details:
        counts = self._get_speaker_counts(reference, hypothesis)
        return {"absolute_error": counts.absolute_error, "total_samples": 1}

    def compute_metric(self, detail: Details) -> float:
        if detail["total_samples"] == 0:
            return 0.0
        return detail["absolute_error"] / detail["total_samples"]


class SpeakerCountAccuracy(BaseSpeakerCountMetric):
    """Speaker Count Accuracy (SCA)

    Measures the proportion of exactly correct speaker count predictions.

    SCA = (number of correct predictions) / (total number of samples)

    Returns a value between 0.0 and 1.0, where 1.0 indicates perfect accuracy.

    The global value for this metric is computed by the aggregation of its components.
    I.e. the global SCA is -> sum(correct_predictions) / total_samples
    """

    @classmethod
    def metric_name(cls) -> str:
        return "speaker count accuracy"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return ["correct_predictions", "total_samples"]

    def compute_components(self, reference: Annotation, hypothesis: Annotation, **kwargs) -> Details:
        counts = self._get_speaker_counts(reference, hypothesis)
        return {
            "correct_predictions": int(counts.is_correct),
            "total_samples": 1,
        }

    def compute_metric(self, detail: Details) -> float:
        if detail["total_samples"] == 0:
            return 0.0
        return detail["correct_predictions"] / detail["total_samples"]
