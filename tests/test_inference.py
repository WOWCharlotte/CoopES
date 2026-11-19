from __future__ import annotations

import sqlite3

import pytest

from coopes.data import Choice, Question
from coopes.inference import (
    InferenceEngine,
    InferenceError,
    InferenceMetadata,
    TransientInferenceError,
    _build_cache_key,
    _create_cache,
)
from coopes.prompts import build_prompt


def _request(question_id: str):
    question = Question(
        question_id=question_id,
        stem="Stem",
        choices=(Choice("A", "one"), Choice("B", "two")),
        fact=None,
    )
    return build_prompt(question, [])


def _metadata() -> InferenceMetadata:
    return InferenceMetadata(
        backend="ollama",
        model_id="test-model",
        model_version="digest",
        template_digest="template",
        runtime_versions={"python": "3.11"},
        settings={},
    )


def _engine(
    connection: sqlite3.Connection,
    generator,
    *,
    batch_limit: int = 1,
    max_attempts: int = 1,
) -> InferenceEngine:
    _create_cache(connection)
    return InferenceEngine(
        metadata=_metadata(),
        connection=connection,
        batch_limit=batch_limit,
        max_attempts=max_attempts,
        cache_namespace="namespace",
        generate_uncached=generator,
    )


def test_cache_key_is_stable_and_content_sensitive() -> None:
    first = _build_cache_key("ns", _request("q1"))
    second = _build_cache_key("ns", _request("q1"))
    other = _build_cache_key("ns", _request("q2"))
    assert first == second
    assert first != other


def test_generate_batch_caches_and_reports_hits() -> None:
    connection = sqlite3.connect(":memory:")
    calls = 0

    def generate(requests):
        nonlocal calls
        calls += 1
        return tuple("A" for _ in requests)

    engine = _engine(connection, generate)
    request = _request("q1")
    first = engine.generate_batch([request, request])
    second = engine.generate_batch([request])
    assert calls == 1
    assert first[0].text == "A"
    assert first[0].cache_hit is False
    assert first[1].cache_hit is False
    assert second[0].cache_hit is True
    stats = engine.snapshot_stats()
    assert stats.backend_calls == 1
    assert stats.submitted_inputs == 1
    assert stats.cache_hits == 1


def test_successful_earlier_batches_survive_later_failure() -> None:
    connection = sqlite3.connect(":memory:")
    failure = {"enabled": True}

    def generate(requests):
        if failure["enabled"] and requests[0].question_id.endswith("q2"):
            raise InferenceError("boom")
        return ("A",)

    engine = _engine(connection, generate, batch_limit=1)
    requests = [_request("q1"), _request("q2")]
    with pytest.raises(InferenceError):
        engine.generate_batch(requests)
    failure["enabled"] = False
    retried = _engine(connection, generate, batch_limit=1)
    result = retried.generate_batch([requests[0]])
    assert result[0].cache_hit is True


def test_transient_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    connection = sqlite3.connect(":memory:")
    attempts = 0

    def generate(requests):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientInferenceError("temporary")
        return ("B",)

    engine = _engine(connection, generate, max_attempts=3)
    result = engine.generate_batch([_request("q1")])
    assert result[0].text == "B"
    stats = engine.snapshot_stats()
    assert stats.backend_calls == 3
    assert stats.retry_attempts == 2


def test_ollama_backend_sends_deterministic_options() -> None:
    from coopes.inference import DecodeConfig, OllamaConfig, _generate_ollama

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": "A"}}

    class Client:
        def __init__(self) -> None:
            self.body = None

        def post(self, url, json):
            self.url = url
            self.body = json
            return Response()

    client = Client()
    config = OllamaConfig(
        base_url="http://localhost:11434",
        model_tag="model",
        model_digest="digest",
        timeout_seconds=1.0,
        max_attempts=1,
        decode=DecodeConfig(seed=7, max_new_tokens=8, context_window=1024),
    )
    result = _generate_ollama([_request("q1")], client, config)
    assert result == ("A",)
    assert client.url.endswith("/api/chat")
    assert client.body["stream"] is False
    assert client.body["options"] == {
        "temperature": 0,
        "seed": 7,
        "num_predict": 8,
        "num_ctx": 1024,
    }


def test_transformers_backend_strips_prompt_tokens() -> None:
    import torch

    from coopes.inference import (
        DecodeConfig,
        TransformersConfig,
        _generate_transformers,
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "right"
        chat_template = "fake-template"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, texts, **kwargs):
            width = max(len(text) for text in texts)
            ids = torch.ones((len(texts), width), dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }

        def batch_decode(self, tokens, **kwargs):
            return ["A", "B"]

    class Model:
        device = None

        def generate(self, input_ids, attention_mask, **kwargs):
            extra = torch.tensor([[5], [6]], dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)

    config = TransformersConfig(
        model_id="model",
        revision="rev",
        device="cpu",
        dtype="float32",
        batch_size=2,
        decode=DecodeConfig(seed=1, max_new_tokens=2, context_window=16),
    )
    result = _generate_transformers(
        [_request("q1"), _request("q2")],
        Model(),
        Tokenizer(),
        config,
    )
    assert result == ("A", "B")


def test_transformers_backend_rejects_short_context() -> None:
    import torch

    from coopes.inference import (
        DecodeConfig,
        TransformersConfig,
        _generate_transformers,
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        chat_template = "fake-template"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, texts, **kwargs):
            ids = torch.ones((1, 8), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Model:
        device = None

    config = TransformersConfig(
        model_id="model",
        revision="rev",
        device="cpu",
        dtype="float32",
        batch_size=1,
        decode=DecodeConfig(seed=1, max_new_tokens=2, context_window=9),
    )
    with pytest.raises(InferenceError):
        _generate_transformers(
            [_request("q1")],
            Model(),
            Tokenizer(),
            config,
        )
