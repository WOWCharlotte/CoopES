from __future__ import annotations

import random

import pytest

from coopes.data import normalize_question, sample_questions


def _openbookqa_row(question_id: str = "1") -> dict[str, object]:
    return {
        "id": question_id,
        "question_stem": "Which object floats?",
        "choices": {
            "label": ["A", "B"],
            "text": ["Steel", "Cork"],
        },
        "answerKey": "B",
        "fact1": "Cork is less dense than water.",
    }


def _arc_row(question_id: str = "2") -> dict[str, object]:
    return {
        "id": question_id,
        "question": "Which material conducts?",
        "choices": {
            "label": ["1", "2"],
            "text": ["Copper", "Rubber"],
        },
        "answerKey": "1",
    }


def test_normalize_openbookqa_preserves_choices_and_fact() -> None:
    normalized = normalize_question(_openbookqa_row(), "openbookqa")
    assert normalized.question.question_id == "openbookqa:1"
    assert normalized.question.fact == "Cork is less dense than water."
    assert normalized.answer_label == "B"
    assert [(c.label, c.text) for c in normalized.question.choices] == [
        ("A", "Steel"),
        ("B", "Cork"),
    ]


def test_normalize_arc_uses_question_and_no_fact() -> None:
    normalized = normalize_question(_arc_row(), "arc_challenge")
    assert normalized.question.question_id == "arc_challenge:2"
    assert normalized.question.fact is None
    assert normalized.question.stem == "Which material conducts?"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answerKey", "C"),
        ("answerKey", ""),
        ("question_stem", ""),
    ],
)
def test_normalize_rejects_invalid_rows(field: str, value: object) -> None:
    row = _openbookqa_row()
    row[field] = value
    with pytest.raises(ValueError):
        normalize_question(row, "openbookqa")


def test_sample_questions_is_stable_and_sorted() -> None:
    questions = [
        normalize_question(_arc_row(str(index)), "arc_challenge")
        for index in range(10, 0, -1)
    ]
    first = sample_questions(questions, 4, random.Random(7))
    second = sample_questions(questions, 4, random.Random(7))
    assert first == second
    assert [q.question.question_id for q in first] == sorted(
        q.question.question_id for q in first
    )


def test_sample_questions_rejects_invalid_count() -> None:
    questions = [normalize_question(_arc_row("1"), "arc_challenge")]
    with pytest.raises(ValueError):
        sample_questions(questions, 0, random.Random(1))
    with pytest.raises(ValueError):
        sample_questions(questions, 2, random.Random(1))
