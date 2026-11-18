from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from coopes import cli
from coopes.cli import RunConfig, run_experiment
from coopes.data import (
    Choice,
    DataConfig,
    DatasetSplits,
    LabeledQuestion,
    Question,
)
from coopes.inference import (
    DecodeConfig,
    GenerationResult,
    InferenceMetadata,
    InferenceStats,
    OllamaConfig,
)
from coopes.selection import SelectionConfig


def _labeled(question_id: str) -> LabeledQuestion:
    return LabeledQuestion(
        question=Question(
            question_id=question_id,
            stem=f"Stem {question_id}",
            choices=(Choice("A", "one"), Choice("B", "two")),
            fact=None,
        ),
        answer_label="A",
    )


class FakeEngine:
    metadata = InferenceMetadata(
        backend="ollama",
        model_id="fake",
        model_version="v1",
        template_digest="template",
        runtime_versions={},
        settings={},
    )

    def generate_batch(self, requests):
        return tuple(GenerationResult("A", False) for _ in requests)

    def snapshot_stats(self) -> InferenceStats:
        return InferenceStats(0, 1, 1, 0)


@contextmanager
def _fake_open_engine(config, cache_path):
    cache_path.touch()
    yield FakeEngine()


def test_run_experiment_writes_complete_artifact_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    splits = DatasetSplits(
        candidates=(
            _labeled("arc_challenge:c1"),
            _labeled("arc_challenge:c2"),
            _labeled("arc_challenge:c3"),
        ),
        calibration=(
            _labeled("arc_challenge:v1"),
            _labeled("arc_challenge:v2"),
        ),
        test=(
            _labeled("arc_challenge:t1"),
            _labeled("arc_challenge:t2"),
        ),
    )
    config = RunConfig(
        data=DataConfig(
            dataset="arc_challenge",
            revision="rev",
            candidate_count=3,
            calibration_count=2,
            test_count=2,
            split_seed=42,
        ),
        inference=OllamaConfig(
            base_url="http://localhost:11434",
            model_tag="fake",
            model_digest="v1",
            timeout_seconds=1,
            max_attempts=1,
            decode=DecodeConfig(42, 8, 1024),
        ),
        selection=SelectionConfig(
            shot_count=2,
            num_samples=4,
            sample_seed=42,
            random_seed=42,
            random_trials=2,
            epsilon=0.05,
        ),
    )
    monkeypatch.setattr(cli, "load_splits", lambda _: splits)
    monkeypatch.setattr(cli, "open_engine", _fake_open_engine)

    summary = run_experiment(config, tmp_path)
    assert summary.method_metrics["coopes_uniform"].mean_accuracy == 1.0
    assert summary.method_metrics["random"].trial_count == 2
    expected = {
        "cache.sqlite3",
        "config.json",
        "candidate_scores.csv",
        "inference_metadata.json",
        "metrics.csv",
        "predictions.jsonl",
        "selections.json",
        "splits.json",
        "summary.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
