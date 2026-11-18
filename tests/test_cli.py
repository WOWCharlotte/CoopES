from __future__ import annotations

import json
from pathlib import Path

import pytest

from coopes.cli import (
    TrialReport,
    load_config,
    prepare_run,
    write_evaluation_artifacts,
    write_selection_artifacts,
)
from coopes.data import DatasetSplits
from coopes.evaluation import EvaluationMetrics, EvaluationResult
from coopes.inference import (
    InferenceMetadata,
    InferenceStats,
    TransformersConfig,
)
from coopes.selection import CandidateScore, SelectionPlan, SelectionTrial


def test_load_config_validates_all_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    ollama = load_config(root / "configs" / "smoke.toml")
    transformers = load_config(root / "configs" / "smoke-transformers.toml")
    assert ollama.selection.shot_count == 2
    assert isinstance(transformers.inference, TransformersConfig)
    assert transformers.inference.batch_size == 4


def test_prepare_run_rejects_completed_run(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"
    )
    splits = DatasetSplits((), (), ())
    prepare_run(tmp_path, config, splits)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path, config, splits)


def _plan() -> SelectionPlan:
    scores = (
        CandidateScore("a", 0.2, 8),
        CandidateScore("b", 0.1, 8),
    )
    return SelectionPlan(
        trials=(
            SelectionTrial("zero_shot", 1, ()),
            SelectionTrial("coopes_uniform", 1, ("a", "b")),
        ),
        scores={"coopes_uniform": scores},
        missing_by_method={},
    )


def test_selection_artifacts_are_written(tmp_path: Path) -> None:
    write_selection_artifacts(tmp_path, _plan())
    assert "candidate_id,score,sample_count" in (
        tmp_path / "candidate_scores.csv"
    ).read_text(encoding="utf-8")
    payload = json.loads(
        (tmp_path / "selections.json").read_text(encoding="utf-8")
    )
    assert payload["trials"][1]["candidate_ids"] == ["a", "b"]


def test_evaluation_artifacts_mean_random_trials(
    tmp_path: Path,
) -> None:
    metrics_a = EvaluationMetrics(2, 0.5, 0.5, 0.0)
    metrics_b = EvaluationMetrics(2, 1.0, 1.0, 0.0)
    reports = [
        TrialReport(
            SelectionTrial("random", 1, ("a", "b")),
            EvaluationResult((), metrics_a),
            1.0,
        ),
        TrialReport(
            SelectionTrial("random", 2, ("a", "b")),
            EvaluationResult((), metrics_b),
            1.0,
        ),
    ]
    metadata = InferenceMetadata(
        backend="ollama",
        model_id="m",
        model_version="v",
        template_digest="t",
        runtime_versions={},
        settings={},
    )
    stats = InferenceStats(0, 2, 4, 0)
    summary = write_evaluation_artifacts(
        tmp_path,
        reports,
        metadata,
        stats,
        3.0,
    )
    assert summary.method_metrics["random"].mean_accuracy == 0.75
    payload = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert payload["method_metrics"]["random"]["trial_count"] == 2
