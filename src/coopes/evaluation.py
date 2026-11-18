"""Calibration utilities, prediction grading, and test metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .data import LabeledQuestion, Question
from .inference import InferenceEngine
from .prompts import build_prompt, parse_answer

INVALID_PREDICTION_LABEL = ""


@dataclass(frozen=True)
class Prediction:
    """A parsed response for one question."""

    question_id: str
    raw_text: str
    predicted_label: str | None
    cache_hit: bool


@dataclass(frozen=True)
class GradedPrediction:
    """A prediction joined with its gold label and correctness."""

    prediction: Prediction
    gold_label: str
    is_correct: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics for one frozen example selection."""

    question_count: int
    accuracy: float
    weighted_f1: float
    parse_failure_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    """Per-question rows and aggregate metrics for one trial."""

    rows: tuple[GradedPrediction, ...]
    metrics: EvaluationMetrics


def predict_questions(
    questions: Sequence[Question],
    examples: Sequence[LabeledQuestion],
    engine: InferenceEngine,
) -> tuple[Prediction, ...]:
    """Build prompts, run inference, and parse labels in order."""

    if not questions:
        return ()
    requests = [build_prompt(question, examples) for question in questions]
    results = engine.generate_batch(requests)
    if len(results) != len(questions):
        raise ValueError("inference returned an unexpected response count")
    predictions: list[Prediction] = []
    for question, result in zip(questions, results, strict=True):
        label = parse_answer(
            result.text,
            tuple(choice.label for choice in question.choices),
        )
        predictions.append(
            Prediction(
                question_id=question.question_id,
                raw_text=result.text,
                predicted_label=label,
                cache_hit=result.cache_hit,
            )
        )
    return tuple(predictions)


def grade_predictions(
    questions: Sequence[LabeledQuestion],
    predictions: Sequence[Prediction],
) -> tuple[GradedPrediction, ...]:
    """Join unique predictions to questions and mark correctness."""

    question_ids = [item.question.question_id for item in questions]
    prediction_ids = [item.question_id for item in predictions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question IDs must be unique")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction IDs must be unique")
    if set(question_ids) != set(prediction_ids):
        raise ValueError("question and prediction IDs must match exactly")

    by_id = {item.question_id: item for item in predictions}
    graded: list[GradedPrediction] = []
    for question in questions:
        prediction = by_id[question.question.question_id]
        graded.append(
            GradedPrediction(
                prediction=prediction,
                gold_label=question.answer_label,
                is_correct=(
                    prediction.predicted_label == question.answer_label
                ),
            )
        )
    return tuple(graded)


def make_utility(
    candidates: Sequence[LabeledQuestion],
    calibration: Sequence[LabeledQuestion],
    engine: InferenceEngine,
) -> Callable[[tuple[str, ...]], float]:
    """Create a cached calibration-set accuracy utility callback."""

    if not calibration:
        raise ValueError("calibration set must not be empty")
    candidate_ids = [item.question.question_id for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    calibration_ids = [item.question.question_id for item in calibration]
    if len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError("calibration IDs must be unique")
    by_id = {item.question.question_id: item for item in candidates}
    utility_cache: dict[tuple[str, ...], float] = {}
    calibration_questions = tuple(item.question for item in calibration)

    def utility(selected_ids: tuple[str, ...]) -> float:
        if not isinstance(selected_ids, tuple):
            raise ValueError("utility input must be a tuple of candidate IDs")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("utility input contains duplicate candidate IDs")
        unknown = [item for item in selected_ids if item not in by_id]
        if unknown:
            raise ValueError(f"unknown utility candidate IDs: {unknown}")
        key = tuple(sorted(selected_ids))
        if key in utility_cache:
            return utility_cache[key]
        examples = tuple(by_id[item] for item in key)
        predictions = predict_questions(
            calibration_questions,
            examples,
            engine,
        )
        graded = grade_predictions(calibration, predictions)
        value = sum(row.is_correct for row in graded) / len(graded)
        utility_cache[key] = value
        return value

    return utility


def compute_metrics(
    rows: Sequence[GradedPrediction],
) -> EvaluationMetrics:
    """Compute accuracy, weighted F1, and parse-failure rate."""

    if not rows:
        raise ValueError("metrics require at least one graded prediction")
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:  # pragma: no cover - core dependency
        raise RuntimeError("compute_metrics requires scikit-learn") from exc

    gold = [row.gold_label for row in rows]
    predicted = [
        row.prediction.predicted_label
        if row.prediction.predicted_label is not None
        else INVALID_PREDICTION_LABEL
        for row in rows
    ]
    correct = sum(row.is_correct for row in rows)
    parse_failures = sum(
        row.prediction.predicted_label is None for row in rows
    )
    weighted_f1 = f1_score(
        gold,
        predicted,
        average="weighted",
        zero_division=0,
    )
    return EvaluationMetrics(
        question_count=len(rows),
        accuracy=correct / len(rows),
        weighted_f1=float(weighted_f1),
        parse_failure_rate=parse_failures / len(rows),
    )


def evaluate_trial(
    examples: Sequence[LabeledQuestion],
    test_questions: Sequence[LabeledQuestion],
    engine: InferenceEngine,
) -> EvaluationResult:
    """Evaluate one frozen selection on a non-empty test set."""

    if not test_questions:
        raise ValueError("test questions must not be empty")
    predictions = predict_questions(
        tuple(item.question for item in test_questions),
        examples,
        engine,
    )
    rows = grade_predictions(test_questions, predictions)
    return EvaluationResult(rows=rows, metrics=compute_metrics(rows))
