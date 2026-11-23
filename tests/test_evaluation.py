from __future__ import annotations

import pytest

from coopes.data import Choice, LabeledQuestion, Question
from coopes.evaluation import (
    GradedPrediction,
    Prediction,
    compute_metrics,
    grade_predictions,
    make_utility,
    predict_questions,
)
from coopes.inference import GenerationResult


def _labeled(question_id: str, answer: str = "A") -> LabeledQuestion:
    return LabeledQuestion(
        question=Question(
            question_id=question_id,
            stem=f"Stem {question_id}",
            choices=(Choice("A", "first"), Choice("B", "second")),
            fact=None,
        ),
        answer_label=answer,
    )


class FakeEngine:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate_batch(self, requests):
        selected = self.responses[self.calls : self.calls + len(requests)]
        self.calls += len(requests)
        return tuple(GenerationResult(text, False) for text in selected)


def test_predict_and_grade_questions() -> None:
    question = _labeled("arc_challenge:1", "A")
    engine = FakeEngine(["A"])
    predictions = predict_questions([question.question], [], engine)
    graded = grade_predictions([question], predictions)
    assert graded[0].is_correct is True


def test_grade_rejects_id_mismatch() -> None:
    question = _labeled("arc_challenge:1")
    prediction = Prediction("arc_challenge:2", "A", "A", False)
    with pytest.raises(ValueError):
        grade_predictions([question], [prediction])


def test_utility_caches_repeated_subsets() -> None:
    candidates = (_labeled("arc_challenge:a"), _labeled("arc_challenge:b"))
    calibration = (_labeled("arc_challenge:c"),)
    engine = FakeEngine(["A", "A"])
    utility = make_utility(candidates, calibration, engine)
    assert utility(()) == 1.0
    assert utility(()) == 1.0
    assert engine.calls == 1
    assert utility(("arc_challenge:a",)) == 1.0
    assert engine.calls == 2


def test_compute_metrics_handles_parse_failures() -> None:
    correct = GradedPrediction(
        Prediction("q1", "A", "A", False),
        "A",
        True,
    )
    failed = GradedPrediction(
        Prediction("q2", "not a label", None, False),
        "B",
        False,
    )
    metrics = compute_metrics([correct, failed])
    assert metrics.question_count == 2
    assert metrics.accuracy == 0.5
    assert metrics.parse_failure_rate == 0.5
    assert 0.0 <= metrics.weighted_f1 <= 1.0
