"""Inference backends, persistent caching, retries, and metadata."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
)

from .prompts import (
    ARC_USER_TEMPLATE,
    OPENBOOKQA_USER_TEMPLATE,
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    PromptRequest,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

BackendName = Literal["ollama", "transformers"]
DTypeName = Literal["float32", "float16", "bfloat16"]

LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


@dataclass(frozen=True)
class DecodeConfig:
    """Backend-independent decoding limits."""

    seed: int
    max_new_tokens: int
    context_window: int


@dataclass(frozen=True)
class OllamaConfig:
    """Configuration for the Ollama chat backend."""

    base_url: str
    model_tag: str
    model_digest: str
    timeout_seconds: float
    max_attempts: int
    decode: DecodeConfig


@dataclass(frozen=True)
class TransformersConfig:
    """Local Transformers causal-language-model configuration."""

    model_id: str
    revision: str
    device: str
    dtype: DTypeName
    batch_size: int
    decode: DecodeConfig


InferenceConfig = OllamaConfig | TransformersConfig


@dataclass(frozen=True)
class InferenceMetadata:
    """Metadata describing the active inference setup."""

    backend: BackendName
    model_id: str
    model_version: str
    template_digest: str
    runtime_versions: Mapping[str, str]
    settings: Mapping[str, str | int | float]


@dataclass(frozen=True)
class GenerationResult:
    """A model response and its persistent-cache status."""

    text: str
    cache_hit: bool


@dataclass(frozen=True)
class InferenceStats:
    """Process-local inference counters."""

    cache_hits: int
    backend_calls: int
    submitted_inputs: int
    retry_attempts: int


class InferenceError(RuntimeError):
    """A generation or cache operation could not be completed."""


class TransientInferenceError(InferenceError):
    """A retryable backend failure."""


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _template_digest() -> str:
    payload = "\n".join(
        (
            PROMPT_TEMPLATE_VERSION,
            SYSTEM_PROMPT,
            OPENBOOKQA_USER_TEMPLATE,
            ARC_USER_TEMPLATE,
            LLAMA3_CHAT_TEMPLATE,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _cache_namespace(config: InferenceConfig) -> str:
    payload: dict[str, object] = {
        "decode": asdict(config.decode),
        "template_digest": _template_digest(),
    }
    if isinstance(config, OllamaConfig):
        payload.update(
            {
                "backend": "ollama",
                "model": config.model_tag,
                "model_version": config.model_digest,
            }
        )
    else:
        payload.update(
            {
                "backend": "transformers",
                "model": config.model_id,
                "model_version": config.revision,
                "dtype": config.dtype,
            }
        )
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _build_cache_key(namespace: str, request: PromptRequest) -> str:
    """Return a stable SHA-256 cache key for a namespace and request."""

    payload = {
        "namespace": namespace,
        "request": {
            "question_id": request.question_id,
            "example_ids": list(request.example_ids),
            "messages": [asdict(message) for message in request.messages],
            "template_version": request.template_version,
        },
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _generate_ollama(
    requests: Sequence[PromptRequest],
    client: object,
    config: OllamaConfig,
) -> tuple[str, ...]:
    """Generate exactly one response through the Ollama chat API."""

    if len(requests) != 1:
        raise InferenceError("the Ollama backend accepts one request per call")
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InferenceError(
            "the Ollama backend requires the optional 'ollama' dependencies"
        ) from exc

    request = requests[0]
    body = {
        "model": config.model_tag,
        "messages": [asdict(message) for message in request.messages],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": config.decode.seed,
            "num_predict": config.decode.max_new_tokens,
            "num_ctx": config.decode.context_window,
        },
    }
    try:
        response = client.post(  # type: ignore[attr-defined]
            f"{config.base_url.rstrip('/')}/api/chat",
            json=body,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise TransientInferenceError(
            f"Ollama request failed for question {request.question_id}: {exc}"
        ) from exc

    if response.status_code == 408 or response.status_code == 429:
        raise TransientInferenceError(
            "Ollama returned a retryable status "
            f"{response.status_code} for question {request.question_id}"
        )
    if response.status_code >= 500:
        raise TransientInferenceError(
            "Ollama returned a retryable status "
            f"{response.status_code} for question {request.question_id}"
        )
    if response.status_code >= 400:
        raise InferenceError(
            "Ollama returned status "
            f"{response.status_code} for question {request.question_id}: "
            f"{response.text[:500]}"
        )

    try:
        payload = response.json()
        content = payload["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise InferenceError(
            "Ollama returned an invalid response for question "
            f"{request.question_id}"
        ) from exc
    if not isinstance(content, str):
        raise InferenceError(
            "Ollama returned non-text content for question "
            f"{request.question_id}"
        )
    return (content,)


def _generate_transformers(
    requests: Sequence[PromptRequest],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: TransformersConfig,
) -> tuple[str, ...]:
    """Generate a batch with a local Transformers model."""

    if not requests:
        return ()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InferenceError(
            "Transformers inference requires the optional 'transformers' "
            "dependencies"
        ) from exc

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise InferenceError(
                "tokenizer has neither a padding token nor an end token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        template_kwargs: dict[str, str] = {}
        if not getattr(tokenizer, "chat_template", None):
            marker_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
            if marker_id == tokenizer.unk_token_id:
                raise InferenceError(
                    "tokenizer has no chat template and cannot use the "
                    "Llama 3 fallback"
                )
            template_kwargs["chat_template"] = LLAMA3_CHAT_TEMPLATE
        texts = [
            tokenizer.apply_chat_template(
                [asdict(message) for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for request in requests
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_width = int(encoded["input_ids"].shape[1])
        if input_width + config.decode.max_new_tokens > (
            config.decode.context_window
        ):
            raise InferenceError(
                "prompt length exceeds the configured context window: "
                f"{input_width} + {config.decode.max_new_tokens} > "
                f"{config.decode.context_window}"
            )
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {
                key: value.to(model_device) for key, value in encoded.items()
            }
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=config.decode.max_new_tokens,
                max_length=None,
                pad_token_id=tokenizer.pad_token_id,
            )
    except InferenceError:
        raise
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "context length" in message:
            raise InferenceError(
                "Transformers generation failed because of memory or "
                f"context limits: {exc}"
            ) from exc
        raise InferenceError(f"Transformers generation failed: {exc}") from exc
    except Exception as exc:
        raise InferenceError(f"Transformers generation failed: {exc}") from exc

    if generated.shape[0] != len(requests):
        raise InferenceError("Transformers returned an invalid batch size")
    new_tokens = generated[:, input_width:]
    decoded = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != len(requests):
        raise InferenceError("Transformers returned an invalid response count")
    return tuple(str(text) for text in decoded)


class InferenceEngine:
    """Cached inference facade shared by calibration and evaluation."""

    def __init__(
        self,
        metadata: InferenceMetadata,
        connection: sqlite3.Connection,
        batch_limit: int,
        max_attempts: int,
        cache_namespace: str,
        generate_uncached: Callable[
            [Sequence[PromptRequest]], tuple[str, ...]
        ],
    ) -> None:
        if batch_limit <= 0:
            raise ValueError("batch_limit must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.metadata = metadata
        self._connection = connection
        self._batch_limit = batch_limit
        self._max_attempts = max_attempts
        self._cache_namespace = cache_namespace
        self._generate_uncached = generate_uncached
        self._cache_hits = 0
        self._backend_calls = 0
        self._submitted_inputs = 0
        self._retry_attempts = 0

    def _generate_missing_batch(
        self,
        requests: Sequence[PromptRequest],
    ) -> tuple[str, ...]:
        """Generate a batch with bounded transient retries."""

        if not requests:
            return ()
        for attempt in range(1, self._max_attempts + 1):
            self._backend_calls += 1
            self._submitted_inputs += len(requests)
            try:
                responses = self._generate_uncached(requests)
            except TransientInferenceError as exc:
                if attempt >= self._max_attempts:
                    raise InferenceError(
                        "inference retries exhausted for question IDs "
                        f"{[request.question_id for request in requests]}: "
                        f"{exc}"
                    ) from exc
                self._retry_attempts += 1
                time.sleep(min(0.2 * (2 ** (attempt - 1)), 2.0))
                continue
            if len(responses) != len(requests):
                raise InferenceError(
                    "backend returned a different number of responses than "
                    "requests"
                )
            if any(not isinstance(text, str) for text in responses):
                raise InferenceError("backend returned a non-string response")
            return tuple(responses)
        raise InferenceError("inference attempt loop ended unexpectedly")

    def generate_batch(
        self,
        requests: Sequence[PromptRequest],
    ) -> tuple[GenerationResult, ...]:
        """Return responses in order using the persistent cache."""

        if not requests:
            return ()
        request_keys = [
            _build_cache_key(self._cache_namespace, request)
            for request in requests
        ]
        initially_cached: dict[str, str] = {}
        for cache_key in set(request_keys):
            row = self._connection.execute(
                "SELECT response_text FROM generation_cache "
                "WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                initially_cached[cache_key] = str(row[0])

        self._cache_hits += sum(
            cache_key in initially_cached for cache_key in request_keys
        )
        missing_by_key: dict[str, PromptRequest] = {}
        for cache_key, request in zip(request_keys, requests, strict=True):
            if cache_key not in initially_cached:
                missing_by_key.setdefault(cache_key, request)

        cached = dict(initially_cached)
        missing_items = list(missing_by_key.items())
        for start in range(0, len(missing_items), self._batch_limit):
            batch = missing_items[start : start + self._batch_limit]
            responses = self._generate_missing_batch(
                [request for _, request in batch]
            )
            with self._connection:
                self._connection.executemany(
                    "INSERT OR REPLACE INTO generation_cache "
                    "(cache_key, response_text) VALUES (?, ?)",
                    [
                        (cache_key, response)
                        for (cache_key, _), response in zip(
                            batch, responses, strict=True
                        )
                    ],
                )
            for (cache_key, _), response in zip(batch, responses, strict=True):
                cached[cache_key] = response

        return tuple(
            GenerationResult(
                text=cached[cache_key],
                cache_hit=cache_key in initially_cached,
            )
            for cache_key in request_keys
        )

    def snapshot_stats(self) -> InferenceStats:
        """Return an immutable snapshot of current-process counters."""

        return InferenceStats(
            cache_hits=self._cache_hits,
            backend_calls=self._backend_calls,
            submitted_inputs=self._submitted_inputs,
            retry_attempts=self._retry_attempts,
        )


def _create_cache(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS generation_cache ("
        "cache_key TEXT PRIMARY KEY, "
        "response_text TEXT NOT NULL)"
    )
    connection.commit()


def _verify_ollama_model(client: object, config: OllamaConfig) -> str:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InferenceError(
            "the Ollama backend requires the optional 'ollama' dependencies"
        ) from exc

    base_url = config.base_url.rstrip("/")
    try:
        show_response = client.post(  # type: ignore[attr-defined]
            f"{base_url}/api/show",
            json={"model": config.model_tag},
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise InferenceError(f"failed to verify Ollama model: {exc}") from exc
    if show_response.status_code >= 400:
        raise InferenceError(
            f"Ollama model {config.model_tag!r} is unavailable: "
            f"status {show_response.status_code}"
        )
    digest = ""
    try:
        payload = show_response.json()
        raw_digest = payload.get("digest")
        if isinstance(raw_digest, str):
            digest = raw_digest
    except (ValueError, AttributeError):
        payload = {}

    if not digest:
        try:
            tags_response = client.get(  # type: ignore[attr-defined]
                f"{base_url}/api/tags"
            )
            tags_response.raise_for_status()
            for item in tags_response.json().get("models", []):
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name") or item.get("model")
                item_digest = item.get("digest")
                if name == config.model_tag and isinstance(item_digest, str):
                    digest = item_digest
                    break
        except (httpx.HTTPError, ValueError) as exc:
            raise InferenceError(
                f"failed to resolve Ollama model digest: {exc}"
            ) from exc
    if not digest:
        raise InferenceError(
            "could not resolve the digest for Ollama model "
            f"{config.model_tag!r}"
        )
    if config.model_digest not in ("", "auto", digest):
        raise InferenceError(
            "Ollama model digest mismatch: configured "
            f"{config.model_digest!r}, actual {digest!r}"
        )
    return digest


def _open_transformers(config: TransformersConfig) -> tuple[object, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise InferenceError(
            "Transformers inference requires the optional 'transformers' "
            "dependencies"
        ) from exc

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.dtype]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            torch_dtype=dtype,
        )
        if config.device != "auto":
            model.to(config.device)
        model.eval()
    except Exception as exc:
        raise InferenceError(
            f"failed to load Transformers model {config.model_id!r}: {exc}"
        ) from exc
    return model, tokenizer


@contextmanager
def open_engine(
    config: InferenceConfig,
    cache_path: Path,
) -> Iterator[InferenceEngine]:
    """Open a configured backend and its persistent response cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    runtime_versions = {
        "python": sys.version.split()[0],
        "numpy": _installed_version("numpy"),
    }
    settings: dict[str, str | int | float] = {
        "max_new_tokens": config.decode.max_new_tokens,
        "context_window": config.decode.context_window,
        "seed": config.decode.seed,
    }
    client: object | None = None

    try:
        _create_cache(connection)
        if isinstance(config, OllamaConfig):
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise InferenceError(
                    "the Ollama backend requires the optional 'ollama' "
                    "dependencies"
                ) from exc
            client = httpx.Client(timeout=config.timeout_seconds)
            digest = _verify_ollama_model(client, config)
            runtime_versions["httpx"] = _installed_version("httpx")
            settings.update(
                {
                    "base_url": config.base_url,
                    "temperature": 0,
                    "max_attempts": config.max_attempts,
                }
            )
            metadata_record = InferenceMetadata(
                backend="ollama",
                model_id=config.model_tag,
                model_version=digest,
                template_digest=_template_digest(),
                runtime_versions=runtime_versions,
                settings=settings,
            )

            def backend(
                requests: Sequence[PromptRequest],
            ) -> tuple[str, ...]:
                return _generate_ollama(requests, client, config)

            batch_limit = 1
            max_attempts = config.max_attempts
        else:
            model, tokenizer = _open_transformers(config)
            runtime_versions["torch"] = _installed_version("torch")
            runtime_versions["transformers"] = _installed_version(
                "transformers"
            )
            settings.update(
                {
                    "device": config.device,
                    "dtype": config.dtype,
                    "batch_size": config.batch_size,
                }
            )
            metadata_record = InferenceMetadata(
                backend="transformers",
                model_id=config.model_id,
                model_version=config.revision,
                template_digest=_template_digest(),
                runtime_versions=runtime_versions,
                settings=settings,
            )

            def backend(
                requests: Sequence[PromptRequest],
            ) -> tuple[str, ...]:
                return _generate_transformers(
                    requests, model, tokenizer, config
                )

            batch_limit = config.batch_size
            max_attempts = 1

        engine = InferenceEngine(
            metadata=metadata_record,
            connection=connection,
            batch_limit=batch_limit,
            max_attempts=max_attempts,
            cache_namespace=_cache_namespace(config),
            generate_uncached=backend,
        )
        yield engine
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                close()
        connection.close()
