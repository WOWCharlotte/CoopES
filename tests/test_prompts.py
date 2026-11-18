from __future__ import annotations

from coopes.data import Choice, LabeledQuestion, Question
from coopes.prompts import build_prompt, parse_answer


def _labeled(
    question_id: str,
    fact: str | None = None,
    answer: str = "A",
) -> LabeledQuestion:
    return LabeledQuestion(
        question=Question(
            question_id=question_id,
            stem=f"Stem {question_id}",
            choices=(Choice("A", "first"), Choice("B", "second")),
            fact=fact,
        ),
        answer_label=answer,
    )


def test_build_prompt_sorts_examples_and_keeps_target_last() -> None:
    target = _labeled("arc_challenge:target").question
    examples = [
        _labeled("arc_challenge:z", answer="B"),
        _labeled("arc_challenge:a"),
    ]
    request = build_prompt(target, examples)
    assert request.example_ids == (
        "arc_challenge:a",
        "arc_challenge:z",
    )
    assert [message.role for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert request.messages[4].content == "B"
    assert "Stem arc_challenge:target" in request.messages[-1].content


def test_openbookqa_prompt_contains_fact() -> None:
    example = _labeled("openbookqa:1", fact="A useful fact.")
    request = build_prompt(_labeled("openbookqa:2").question, [example])
    assert request.messages[1].content.startswith("Fact: A useful fact.")


def test_zero_shot_prompt_has_only_system_and_question() -> None:
    request = build_prompt(_labeled("arc_challenge:1").question, [])
    assert [message.role for message in request.messages] == [
        "system",
        "user",
    ]


def test_parse_answer_accepts_only_unambiguous_forms() -> None:
    labels = ("A", "B")
    assert parse_answer("A", labels) == "A"
    assert parse_answer(" a ", labels) == "A"
    assert parse_answer("(B)", labels) == "B"
    assert parse_answer("Answer: b", labels) == "B"
    assert parse_answer("A because it is first", labels) is None
    assert parse_answer("A B", labels) is None
    assert parse_answer("", labels) is None


def test_parse_answer_prefers_exact_case_match() -> None:
    assert parse_answer("a", ("A", "a")) == "a"
    assert parse_answer("A", ("A", "a")) == "A"
