from __future__ import annotations

import numpy as np
import pytest

from coopes.selection import (
    CandidateScore,
    SelectionConfig,
    build_selection_plan,
    draw_prefix,
    estimate_marginal_contributions,
    rank_candidates,
)


def _config(**overrides: object) -> SelectionConfig:
    values = {
        "shot_count": 2,
        "num_samples": 8,
        "sample_seed": 42,
        "epsilon": 0.05,
    }
    values.update(overrides)
    return SelectionConfig(**values)  # type: ignore[arg-type]


def test_draw_prefix_is_distinct_and_reproducible() -> None:
    ids = ("a", "b", "c", "d")
    first = draw_prefix(ids, 3, np.random.default_rng(11))
    second = draw_prefix(ids, 3, np.random.default_rng(11))
    assert first == second
    assert len(set(first)) == 3
    assert set(first) <= set(ids)


def test_draw_prefix_supports_weights() -> None:
    prefix = draw_prefix(
        ("a", "b"),
        2,
        np.random.default_rng(1),
        weights={"a": 1.0, "b": 2.0},
    )
    assert set(prefix) == {"a", "b"}


def test_marginal_contributions_count_every_prefix_position() -> None:
    ids = ("a", "b", "c", "d")

    def utility(selected: tuple[str, ...]) -> float:
        return len(selected) / len(ids)

    scores = estimate_marginal_contributions(ids, utility, _config())
    assert len(scores) == len(ids)
    assert sum(score.sample_count for score in scores) == 16
    assert all(score.score == pytest.approx(0.25) for score in scores)


def test_rank_candidates_breaks_ties_by_id() -> None:
    scores = [
        CandidateScore("b", 0.5, 1),
        CandidateScore("a", 0.5, 1),
        CandidateScore("c", 0.9, 1),
    ]
    assert rank_candidates(scores, 2) == ("c", "a")


def test_build_selection_plan_contains_only_requested_methods() -> None:
    ids = ("a", "b", "c", "d")
    utility_by_id = {"a": 0.1, "b": 0.8, "c": 0.4, "d": 0.2}

    def utility(selected: tuple[str, ...]) -> float:
        return sum(utility_by_id[item] for item in selected) / 10

    plan = build_selection_plan(ids, utility, _config())
    assert [trial.method for trial in plan.trials] == [
        "zero_shot",
        "coopes_uniform",
        "coopes_weighted",
    ]
    assert plan.trials[0].candidate_ids == ()
    assert all(trial.candidate_ids == ("b", "c") for trial in plan.trials[1:])
    assert set(plan.scores) == {"coopes_uniform", "coopes_weighted"}
    assert not plan.missing_by_method


def test_compute_shapley_matches_additive_utility() -> None:
    pytest.importorskip("shap")
    from coopes.selection import compute_shapley

    values = {"a": 0.1, "b": 0.2, "c": 0.3}

    def utility(selected: tuple[str, ...]) -> float:
        return sum(values[item] for item in selected)

    result = compute_shapley(
        tuple(values),
        utility,
        method="exact",
        max_evals=8,
    )
    assert result == pytest.approx(values)
