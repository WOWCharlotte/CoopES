"""Prompt construction and answer-label parsing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .data import LabeledQuestion, Question

SYSTEM_PROMPT = (
    "You answer multiple-choice science questions.\n"
    "Use the provided fact when one is present.\n"
    "Choose exactly one option for the current question.\n"
    "Return only its label exactly as shown in the options, without any "
    "explanation."
)
OPENBOOKQA_USER_TEMPLATE = (
    "Fact: {fact}\n\nQuestion: {stem}\nOptions:\n{option_lines}"
)
ARC_USER_TEMPLATE = "Question: {stem}\nOptions:\n{option_lines}"
PROMPT_TEMPLATE_VERSION = "coopes-mcq-v1"

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class PromptRequest:
    question_id: str
    example_ids: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    template_version: str


def _render_question(question: Question) -> str:
    """Render a question with its original choice order."""

    option_lines = "\n".join(
        f"{choice.label}. {choice.text}" for choice in question.choices
    )
    if question.fact is None:
        return ARC_USER_TEMPLATE.format(
            stem=question.stem,
            option_lines=option_lines,
        )
    return OPENBOOKQA_USER_TEMPLATE.format(
        fact=question.fact,
        stem=question.stem,
        option_lines=option_lines,
    )


def build_prompt(
    question: Question,
    examples: Sequence[LabeledQuestion],
) -> PromptRequest:
    """Build one deterministic chat request for the target question."""

    ordered = tuple(
        sorted(examples, key=lambda item: item.question.question_id)
    )
    example_ids = tuple(item.question.question_id for item in ordered)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("prompt examples must have unique question IDs")
    if question.question_id in example_ids:
        raise ValueError("the target question cannot also be an example")

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT)
    ]
    for example in ordered:
        messages.append(
            ChatMessage(
                role="user",
                content=_render_question(example.question),
            )
        )
        messages.append(
            ChatMessage(role="assistant", content=example.answer_label)
        )
    messages.append(
        ChatMessage(role="user", content=_render_question(question))
    )
    return PromptRequest(
        question_id=question.question_id,
        example_ids=example_ids,
        messages=tuple(messages),
        template_version=PROMPT_TEMPLATE_VERSION,
    )


def _unique_folded_label(
    value: str,
    labels_by_fold: dict[str, str],
) -> str | None:
    folded = value.casefold()
    return labels_by_fold.get(folded)


def parse_answer(
    text: str,
    valid_labels: Sequence[str],
) -> str | None:
    """Parse exactly one valid label from a generated answer.

    Exact labels are preferred. Case-insensitive matching is accepted
    only for an unambiguous original label. Wrapped labels such as
    ``(A)`` and explicit ``Answer: A`` forms are accepted on one line.
    Ambiguity or additional explanation returns ``None``.
    """

    if not isinstance(text, str):
        return None
    labels = tuple(valid_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("valid labels must be non-empty and unique")
    stripped = text.strip()
    if not stripped:
        return None
    if stripped in labels:
        return stripped

    labels_by_fold = {label.casefold(): label for label in labels}
    if len(labels_by_fold) == len(labels):
        folded = _unique_folded_label(stripped, labels_by_fold)
        if folded is not None:
            return folded

    if "\n" in stripped or "\r" in stripped:
        return None

    candidate = stripped
    answer_match = re.fullmatch(r"(?i)\s*answer\s*:\s*(.+?)\s*", candidate)
    if answer_match is not None:
        candidate = answer_match.group(1).strip()
    wrapped_match = re.fullmatch(r"\(\s*(.+?)\s*\)", candidate)
    if wrapped_match is not None:
        candidate = wrapped_match.group(1).strip()

    if candidate in labels:
        return candidate
    if len(labels_by_fold) == len(labels):
        return _unique_folded_label(candidate, labels_by_fold)
    return None
