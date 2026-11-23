"""Marginal-contribution estimation, ranking, and SHAP checks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal[
    "zero_shot",
    "coopes_uniform",
    "coopes_weighted",
]
ShapMethod = Literal["exact", "permutation"]
UtilityFunction = Callable[[tuple[str, ...]], float]

SHAP_EXACT_MAX_CANDIDATES = 12


@dataclass(frozen=True)
class SelectionConfig:
    """Parameterization of cooperative selection."""

    shot_count: int
    num_samples: int
    sample_seed: int
    epsilon: float


@dataclass(frozen=True)
class CandidateScore:
    """Sample-averaged contribution for one candidate."""

    candidate_id: str
    score: float | None
    sample_count: int


@dataclass(frozen=True)
class SelectionTrial:
    """One frozen example selection to evaluate on the test set."""

    method: Method
    trial_index: int
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class SelectionPlan:
    """All trials and diagnostics produced by selection."""

    trials: tuple[SelectionTrial, ...]
    scores: dict[Method, tuple[CandidateScore, ...]]
    missing_by_method: dict[Method, tuple[str, ...]]


def _unique_ids(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if any(not isinstance(item, str) or not item for item in candidate_ids):
        raise ValueError("candidate IDs must be non-empty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    return tuple(candidate_ids)


def _validate_config(config: SelectionConfig) -> None:
    if config.shot_count <= 0:
        raise ValueError("shot_count must be positive")
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not math.isfinite(config.epsilon) or config.epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")


def _utility_value(
    utility: UtilityFunction,
    candidate_ids: tuple[str, ...],
) -> float:
    value = utility(candidate_ids)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("utility must return a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("utility must return a finite numeric value")
    return result


def draw_prefix(
    candidate_ids: Sequence[str],
    shot_count: int,
    rng: np.random.Generator,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Draw a distinct, ordered candidate prefix without replacement."""

    candidates = list(candidate_ids)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    if not 0 < shot_count <= len(candidates):
        raise ValueError("shot_count must be between 1 and candidate count")
    if not candidates:
        raise ValueError("candidate_ids must not be empty")

    remaining = candidates.copy()
    selected: list[str] = []
    for _ in range(shot_count):
        if weights is None:
            index = int(rng.integers(0, len(remaining)))
        else:
            if any(item not in weights for item in remaining):
                raise ValueError("weights must cover every candidate")
            values = np.asarray(
                [weights[item] for item in remaining],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("weights must be finite and positive")
            probabilities = values / values.sum()
            index = int(rng.choice(len(remaining), p=probabilities))
        selected.append(str(remaining.pop(index)))
    return tuple(selected)


def estimate_marginal_contributions(
    candidate_ids: Sequence[str],
    utility: UtilityFunction,
    config: SelectionConfig,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[CandidateScore, ...]:
    """Estimate prefix marginal contributions by sampling."""

    candidates = _unique_ids(candidate_ids)
    _validate_config(config)
    if config.shot_count > len(candidates):
        raise ValueError("shot_count cannot exceed candidate count")

    ordered_ids = tuple(sorted(candidates))
    if weights is not None:
        if any(item not in weights for item in ordered_ids):
            raise ValueError("weights must cover every candidate")
        for item in ordered_ids:
            value = weights[item]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("weights must be finite and positive")

    rng = np.random.default_rng(config.sample_seed)
    gain_by_id = {item: 0.0 for item in ordered_ids}
    count_by_id = {item: 0 for item in ordered_ids}
    baseline = _utility_value(utility, ())

    for _ in range(config.num_samples):
        prefix = draw_prefix(
            ordered_ids,
            config.shot_count,
            rng,
            weights=weights,
        )
        current: list[str] = []
        previous_value = baseline
        for candidate_id in prefix:
            current.append(candidate_id)
            current_key = tuple(sorted(current))
            next_value = _utility_value(utility, current_key)
            gain_by_id[candidate_id] += next_value - previous_value
            count_by_id[candidate_id] += 1
            previous_value = next_value

    return tuple(
        CandidateScore(
            candidate_id=item,
            score=(
                gain_by_id[item] / count_by_id[item]
                if count_by_id[item]
                else None
            ),
            sample_count=count_by_id[item],
        )
        for item in ordered_ids
    )


def rank_candidates(
    scores: Sequence[CandidateScore],
    shot_count: int,
) -> tuple[str, ...]:
    """Rank observed scores and return the top ``shot_count`` IDs."""

    if shot_count <= 0:
        raise ValueError("shot_count must be positive")
    if len(scores) < shot_count:
        raise ValueError("not enough candidate scores to select a prefix")
    ids = [score.candidate_id for score in scores]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate scores must have unique IDs")
    if any(score.score is None for score in scores):
        raise ValueError("cannot rank candidates with missing scores")
    if any(not math.isfinite(float(score.score)) for score in scores):
        raise ValueError("candidate scores must be finite")
    ordered = sorted(
        scores,
        key=lambda item: (-float(item.score), item.candidate_id),
    )
    return tuple(item.candidate_id for item in ordered[:shot_count])


def _missing_candidates(
    scores: Sequence[CandidateScore],
) -> tuple[str, ...]:
    return tuple(score.candidate_id for score in scores if score.score is None)


def build_selection_plan(
    candidate_ids: Sequence[str],
    utility: UtilityFunction,
    config: SelectionConfig,
) -> SelectionPlan:
    """Construct zero-shot and both CoopES selections."""

    candidates = _unique_ids(candidate_ids)
    _validate_config(config)
    if config.shot_count > len(candidates):
        raise ValueError("shot_count cannot exceed candidate count")
    ordered_ids = tuple(sorted(candidates))

    trials: list[SelectionTrial] = [
        SelectionTrial(method="zero_shot", trial_index=1, candidate_ids=())
    ]
    weights = {
        item: config.epsilon + _utility_value(utility, (item,))
        for item in ordered_ids
    }

    uniform_scores = estimate_marginal_contributions(
        ordered_ids,
        utility,
        config,
    )
    weighted_scores = estimate_marginal_contributions(
        ordered_ids,
        utility,
        config,
        weights=weights,
    )

    scores: dict[Method, tuple[CandidateScore, ...]] = {
        "coopes_uniform": uniform_scores,
        "coopes_weighted": weighted_scores,
    }
    missing_by_method: dict[Method, tuple[str, ...]] = {}
    for method, method_scores in scores.items():
        missing = _missing_candidates(method_scores)
        if missing:
            missing_by_method[method] = missing
            continue
        trials.append(
            SelectionTrial(
                method=method,
                trial_index=1,
                candidate_ids=rank_candidates(
                    method_scores, config.shot_count
                ),
            )
        )
    return SelectionPlan(
        trials=tuple(trials),
        scores=scores,
        missing_by_method=missing_by_method,
    )


def compute_shapley(
    candidate_ids: Sequence[str],
    utility: UtilityFunction,
    *,
    method: ShapMethod = "exact",
    max_evals: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Compute standard Shapley values for a small candidate pool."""

    candidates = tuple(sorted(_unique_ids(candidate_ids)))
    if method not in ("exact", "permutation"):
        raise ValueError("method must be 'exact' or 'permutation'")
    if max_evals is not None and max_evals <= 0:
        raise ValueError("max_evals must be positive")
    required_exact = 2 ** len(candidates)
    if method == "exact":
        if len(candidates) > SHAP_EXACT_MAX_CANDIDATES:
            raise ValueError(
                "exact SHAP is limited to "
                f"{SHAP_EXACT_MAX_CANDIDATES} candidates"
            )
        if max_evals is not None and max_evals < required_exact:
            raise ValueError(
                f"exact SHAP requires at least {required_exact} evaluations"
            )
    else:
        minimum = 2 * len(candidates) + 1
        if max_evals is None or max_evals < minimum:
            raise ValueError(
                f"permutation SHAP requires max_evals >= {minimum}"
            )

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "compute_shapley requires the optional 'shap' dependencies"
        ) from exc

    def evaluate_masks(
        masks: NDArray[np.int8],
    ) -> NDArray[np.float64]:
        values = np.asarray(masks)
        if values.ndim != 2 or values.shape[1] != len(candidates):
            raise ValueError("SHAP masks have an unexpected shape")
        output = np.empty(values.shape[0], dtype=np.float64)
        for row_index, row in enumerate(values):
            selected = tuple(
                candidate
                for candidate, bit in zip(candidates, row, strict=True)
                if int(bit) == 1
            )
            output[row_index] = _utility_value(utility, selected)
        return output

    background = np.zeros((1, len(candidates)), dtype=np.int8)
    instance = np.ones((1, len(candidates)), dtype=np.int8)
    masker = shap.maskers.Independent(background)
    if method == "exact":
        explainer = shap.ExactExplainer(evaluate_masks, masker)
        explanation = explainer(instance)
    else:
        explainer = shap.PermutationExplainer(
            evaluate_masks,
            masker,
            max_evals=max_evals,
            seed=seed,
        )
        explanation = explainer(instance)

    raw_values = getattr(explanation, "values", explanation)
    array = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    if array.size != len(candidates):
        raise RuntimeError(
            "SHAP returned an unexpected number of feature values"
        )
    return {
        candidate: float(value)
        for candidate, value in zip(candidates, array, strict=True)
    }
