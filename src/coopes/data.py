"""Dataset loading, normalization, and split construction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

DatasetName = Literal["openbookqa", "arc_challenge"]

DATASET_SPECS: Mapping[DatasetName, tuple[str, str]] = {
    "openbookqa": ("allenai/openbookqa", "additional"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
}


@dataclass(frozen=True)
class DataConfig:
    """Dataset and deterministic sample-size configuration."""

    dataset: DatasetName
    revision: str
    candidate_count: int
    calibration_count: int
    test_count: int
    split_seed: int


@dataclass(frozen=True)
class Choice:
    """An answer choice preserving its original label and order."""

    label: str
    text: str


@dataclass(frozen=True)
class Question:
    """An unlabeled multiple-choice question."""

    question_id: str
    stem: str
    choices: tuple[Choice, ...]
    fact: str | None


@dataclass(frozen=True)
class LabeledQuestion:
    """A question paired with its gold answer label."""

    question: Question
    answer_label: str


@dataclass(frozen=True)
class DatasetSplits:
    """The isolated candidate, calibration, and test collections."""

    candidates: tuple[LabeledQuestion, ...]
    calibration: tuple[LabeledQuestion, ...]
    test: tuple[LabeledQuestion, ...]


def _required_text(
    row: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing or empty {key!r}")
    return value.strip()


def _choice_fields(
    row: Mapping[str, object],
    context: str,
) -> tuple[list[object], list[object]]:
    choices = row.get("choices")
    if not isinstance(choices, Mapping):
        raise ValueError(f"{context}: 'choices' must be a mapping")
    labels = choices.get("label")
    texts = choices.get("text")
    if not isinstance(labels, list) or not isinstance(texts, list):
        raise ValueError(
            f"{context}: choices must contain list-valued label and text"
        )
    if not labels or len(labels) != len(texts):
        raise ValueError(
            f"{context}: choices must be non-empty and equally sized"
        )
    return labels, texts


def normalize_question(
    row: Mapping[str, object],
    dataset: DatasetName,
) -> LabeledQuestion:
    """Normalize one dataset row into the common domain model.

    Bad rows raise ``ValueError`` with the dataset and question ID when
    the latter is available. Labels retain their original spelling and
    order.
    """

    if dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {dataset!r}")
    raw_id = row.get("id")
    context_id = raw_id if raw_id is not None else "<unknown-id>"
    context = f"dataset={dataset}, question_id={context_id}"
    question_id = _required_text(row, "id", context)
    stem_key = "question_stem" if dataset == "openbookqa" else "question"
    stem = _required_text(row, stem_key, context)
    labels, texts = _choice_fields(row, context)

    choices: list[Choice] = []
    for index, (label, text) in enumerate(zip(labels, texts, strict=True)):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{context}: choice {index} has an empty label")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context}: choice {index} has empty text")
        choices.append(Choice(label.strip(), text.strip()))
    if len({choice.label for choice in choices}) != len(choices):
        raise ValueError(f"{context}: choice labels must be unique")

    answer_label = _required_text(row, "answerKey", context)
    if answer_label not in {choice.label for choice in choices}:
        raise ValueError(
            f"{context}: answerKey {answer_label!r} is not a choice label"
        )

    fact: str | None = None
    if dataset == "openbookqa":
        fact = _required_text(row, "fact1", context)

    return LabeledQuestion(
        question=Question(
            question_id=f"{dataset}:{question_id}",
            stem=stem,
            choices=tuple(choices),
            fact=fact,
        ),
        answer_label=answer_label,
    )


def sample_questions(
    questions: Sequence[LabeledQuestion],
    count: int,
    rng: random.Random,
) -> tuple[LabeledQuestion, ...]:
    """Sample ``count`` questions without replacement using ``rng``."""

    ordered = tuple(
        sorted(questions, key=lambda item: item.question.question_id)
    )
    if not 1 <= count <= len(ordered):
        raise ValueError(
            f"sample count {count} must be between 1 and {len(ordered)}"
        )
    if count == len(ordered):
        return ordered
    sampled = rng.sample(ordered, count)
    return tuple(sorted(sampled, key=lambda item: item.question.question_id))


def _load_official_split(
    dataset: DatasetName,
    revision: str,
    official_split: str,
) -> tuple[LabeledQuestion, ...]:
    try:
        from datasets import load_dataset
    # This import is optional and may be unavailable in a minimal
    # installation.
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "loading datasets requires the optional 'data' dependencies"
        ) from exc

    repo_id, config_name = DATASET_SPECS[dataset]
    try:
        rows = load_dataset(
            repo_id,
            config_name,
            split=official_split,
            revision=revision,
        )
    except Exception as exc:
        raise ValueError(
            f"failed to load dataset={dataset}, split={official_split}, "
            f"revision={revision}: {exc}"
        ) from exc

    result: list[LabeledQuestion] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(
                f"dataset={dataset}, split={official_split}: "
                "row is not a mapping"
            )
        normalized = normalize_question(row, dataset)
        question_id = normalized.question.question_id
        if question_id in seen:
            raise ValueError(
                f"dataset={dataset}, split={official_split}: duplicate "
                f"question ID {question_id!r}"
            )
        seen.add(question_id)
        result.append(normalized)
    return tuple(result)


def load_splits(config: DataConfig) -> DatasetSplits:
    """Load splits and construct deterministic isolated samples."""

    if config.dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {config.dataset!r}")
    if not config.revision.strip():
        raise ValueError("data revision must not be empty")
    counts = (
        config.candidate_count,
        config.calibration_count,
        config.test_count,
    )
    if any(count <= 0 for count in counts):
        raise ValueError("all split counts must be positive")

    split_names = ("train", "validation", "test")
    loaded = {
        split: _load_official_split(config.dataset, config.revision, split)
        for split in split_names
    }
    all_ids = [
        item.question.question_id
        for split in split_names
        for item in loaded[split]
    ]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("the same question ID appears in multiple splits")

    candidates = sample_questions(
        loaded["train"],
        config.candidate_count,
        random.Random(f"{config.split_seed}:train"),
    )
    calibration = sample_questions(
        loaded["validation"],
        config.calibration_count,
        random.Random(f"{config.split_seed}:validation"),
    )
    test = sample_questions(
        loaded["test"],
        config.test_count,
        random.Random(f"{config.split_seed}:test"),
    )

    split_ids = [
        {item.question.question_id for item in collection}
        for collection in (candidates, calibration, test)
    ]
    if any(
        split_ids[left] & split_ids[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError(
            "candidate, calibration, and test IDs must be disjoint"
        )
    return DatasetSplits(candidates, calibration, test)
