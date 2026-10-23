# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The model side of the harness: which models are known, and how one call is made.

**Why a serve rather than loading the weights here.** Every pipeline in this project talks to a
vLLM serve over an OpenAI-compatible endpoint, and the serve outlives the eval job, so the same
frozen model can be scored on both halves of the benchmark, and re-scored later, without being
loaded again. It also means the exact thing being measured is named once, by
``--served-model-name``, and can be checked against the endpoint before any work.

**Two failure modes are guarded up front, because both are silent at run time.**

* A served name that does not match what the endpoint answers to returns HTTP 404 on every call.
  Caught per call, that reads as a model that emitted nothing -- a clean-looking file of blanks.
  :func:`evaluation.serve_preflight.assert_endpoint_live` posts one real call first.
* Over the serve's ``limit_mm_per_prompt``, vLLM drops images rather than refusing, so the policy
  is judged on a prefix of the screens it was meant to see.
  :func:`assert_frame_budget` asserts the data fits before the run.

**Frames are downscaled** to the catalogue entry's ``max_image_dim`` (768 pixels on the
longest side for the open-weight entries; full resolution overflows a 16k context on the
taller pages). Every frame is re-encoded, JPEG frames as JPEG at Pillow's default quality
(lossy even when the size is unchanged).
"""

from __future__ import annotations

import base64
import functools
import io
import mimetypes
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from offline_evaluation import http


@dataclass(frozen=True)
class ModelConfig:
    """One model the harness can score.

    ``served_name`` is a default only. The serve's real ``--served-model-name`` is a command-line
    argument and is probed before the run, because a registry constant that drifts from the serve
    is exactly the mismatch that returns 404 on every call.
    """

    key: str
    hf_path: str
    vision: bool
    served_name: str
    #: Longest side each screen is downscaled to before being sent. ``None``: sent at its
    #: original size (the frontier models' recorded setting).
    max_image_dim: int | None
    #: Room for the whole reply. A reasoning model writes its deliberation before the call, so
    #: a budget sized for the call alone truncates it mid-thought and the row is then recorded
    #: as unparseable -- a model that answered well reads as one that cannot follow the format.
    #: Size this per model; there is no safe default. ``None``: no ``max_tokens`` is sent, so
    #: the provider's own default applies (the recorded setting of four frontier models).
    max_tokens: int | None


#: Where the model catalogue lives. One file per model, so adding one is a configuration
#: change and the settings that produced a published number can be read without reading
#: Python.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"

#: The settings a model file must set, and the type each must hold.
_REQUIRED: Mapping[str, type] = {
    "key": str,
    "hf_path": str,
    "vision": bool,
    "max_image_dim": int,
    "max_tokens": int,
}
#: Prose explaining why the settings are what they are. Required, because a budget with no
#: stated reason can only be copied, not reviewed.
_OPTIONAL: Mapping[str, type] = {"description": str}
#: Settings that are meaningless at zero or below: no room for a reply, no pixels to send.
_POSITIVE = ("max_image_dim", "max_tokens")
#: Of those, the ones a file may set to ``null`` (YAML) on purpose: see :class:`ModelConfig`.
_NULLABLE = ("max_image_dim", "max_tokens")


def _parse(path: Path) -> object:
    """The YAML in ``path``, with every failure re-raised naming the file.

    PyYAML given a string has no stream name, so its own errors say ``<unicode string>``.
    This catalogue is edited by hand and the loader chose these paths itself, so an error
    that does not say which file to open sends the reader through all of them.
    """
    import yaml  # noqa: PLC0415 - only the catalogue needs a YAML parser

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name}: cannot be read ({type(exc).__name__})") from exc

    # A repeated setting is kept silently by any YAML parser, last one wins. Copying a
    # block and forgetting to delete the original is the mistake a hand-edited file
    # invites most. Every setting here is a top-level scalar, so the duplicates are
    # findable by reading the keys; nested content is indented and never matches.
    top_level = [
        line.split(":", 1)[0]
        for line in text.splitlines()
        if line[:1].isalpha() and ":" in line
    ]
    repeated = sorted({k for k in top_level if top_level.count(k) > 1})
    if repeated:
        raise ValueError(f"{path.name}: setting {', '.join(repeated)} is given twice")

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{path.name}: is not valid YAML ({type(exc).__name__})"
        ) from exc


def _read_model(path: Path) -> ModelConfig:
    """One model file as a :class:`ModelConfig`, or raise saying which file and why.

    The filename must be the key. For a format whose whole premise is that one file is one
    model, a listing that disagrees with the catalogue is the one thing a reader cannot see.
    """
    loaded = _parse(path)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{path.name}: expected a mapping of settings, got {type(loaded).__name__}"
        )
    known = {**_REQUIRED, **_OPTIONAL}
    unknown = sorted(set(loaded) - set(known))
    if unknown:
        # Ignoring it would leave the reader believing a setting is in force when a typo
        # means the default is.
        raise ValueError(f"{path.name}: unknown setting(s) {', '.join(unknown)}")
    for field, want in known.items():
        if field not in loaded:
            raise ValueError(f"{path.name}: missing required setting {field}")
        value = loaded[field]
        if value is None and field in _NULLABLE:
            continue
        # bool subclasses int, so an int field has to reject True explicitly.
        if not isinstance(value, want) or (want is int and isinstance(value, bool)):
            raise ValueError(
                f"{path.name}: {field} must be {want.__name__}, got {type(value).__name__}"
            )
    for field in _POSITIVE:
        if loaded[field] is not None and loaded[field] <= 0:
            raise ValueError(
                f"{path.name}: {field} must be positive, got {loaded[field]}"
            )
    if loaded["key"] != path.stem:
        raise ValueError(
            f"{path.name}: declares key {loaded['key']}, expected {path.stem}"
        )
    return ModelConfig(
        key=loaded["key"],
        hf_path=loaded["hf_path"],
        vision=loaded["vision"],
        # Not a setting: the serve's real name is a command-line argument, probed before
        # the run. Carried on the dataclass only so a caller can default to the key.
        served_name=loaded["key"],
        max_image_dim=loaded["max_image_dim"],
        max_tokens=loaded["max_tokens"],
    )


def load_models(directory: Path) -> Mapping[str, ModelConfig]:
    """Every model file in ``directory``, keyed by its ``key``, in a fixed order.

    An empty or absent directory raises rather than returning nothing: a silent empty
    catalogue turns every model name into an unknown-key error, which reads as a typo in
    the command rather than as a missing catalogue.

    A file that is not ``.yaml`` raises too. Skipping it would make a whole model vanish
    with no error, in a loader whose point is that an absent catalogue is never silent.
    """
    if not directory.is_dir():
        raise ValueError(f"no model configuration files under {directory}")
    # A model written as .yml would be skipped by the glob below and the model would simply
    # not exist, which is the silent failure this loader exists to avoid. Documentation
    # alongside the files is fine and is not a model.
    stray = sorted(p.name for p in directory.iterdir() if p.suffix in (".yml", ".YAML"))
    if stray:
        raise ValueError(
            f"{directory}: a model file must end in .yaml, found {', '.join(stray)}"
        )
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"no model configuration files under {directory}")
    # Two files cannot claim one key: the key must equal the filename, and a directory
    # cannot hold two files with one name.
    return {config.key: config for config in (_read_model(path) for path in paths)}


#: Every model this harness can score. ``vision`` says whether the weights take images:
#: ``False`` marks a text-only checkpoint, which the runner refuses to run with
#: ``--screens on``. Each file says why its settings are what they are.
#:
#: Read on first use rather than at import, so a malformed file fails the program that asked
#: for a model instead of every module that imports anything from here.
@lru_cache(maxsize=1)
def models() -> Mapping[str, ModelConfig]:
    """The catalogue, read once."""
    return load_models(CONFIG_DIR)


def get_model(key: str) -> ModelConfig:
    """The configuration named ``key``.

    A typo raises with the known keys listed, rather than falling back to a default: scoring the
    wrong model and reporting it under the requested name is worse than not running.
    """
    try:
        return models()[key]
    except KeyError:
        raise KeyError(
            f"unknown model {key!r}; known models are {sorted(models())}"
        ) from None


def frame_data_uri(path: str, max_dim: int | None) -> str | None:
    """One screen as a data URI, downscaled to ``max_dim``, or ``None`` if unreadable.

    The MIME type comes from the filename rather than being hardcoded: this pool holds both PNG
    and JPEG frames, and declaring the wrong one makes some serves reject the image.
    """
    # Imported here rather than at module scope, so scoring an already-finished run needs
    # neither Pillow nor the frames themselves.
    from PIL import Image  # noqa: PLC0415

    try:
        image: Image.Image = Image.open(path)
        image.load()
    except (OSError, ValueError):
        return None
    if max_dim is not None and max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    jpeg = (mimetypes.guess_type(path)[0] or "").endswith("jpeg")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG" if jpeg else "PNG")
    mime = "image/jpeg" if jpeg else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def build_content(
    prompt: str, frames: Sequence[str], config: ModelConfig
) -> tuple[list[dict[str, Any]], int]:
    """The chat message content, and how many screens could not be read.

    Screens come before the text so the prompt reads as a caption to what is shown. A text-only
    model is sent no screens at all rather than a description of them: a stand-in would be a
    different task from the one the vision model is given, and the two would not be comparable.
    """
    content: list[dict[str, Any]] = []
    unreadable = 0
    if config.vision:
        for frame in frames:
            uri = frame_data_uri(frame, config.max_image_dim)
            if uri is None:
                unreadable += 1
                continue
            content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": prompt})
    return content, unreadable


@dataclass(frozen=True)
class Answer:
    """One policy call: what it said, and why it said nothing if it did."""

    text: str
    n_frames_sent: int
    n_frames_unreadable: int
    error: str | None = None
    #: Why the serve stopped generating: ``"stop"`` for a finished reply, ``"length"``
    #: when the output budget ran out. ``None`` when the serve reported nothing.
    #: A reasoning model that fills the budget never reaches its tool call, and without
    #: this the result is indistinguishable from a model that declined to call one.
    finish_reason: str | None = None


class ServeClient:
    """One frozen model behind an OpenAI-compatible endpoint.

    Greedy and seeded, so a reported number is reproducible from the recorded command. The
    endpoint is any OpenAI-compatible ``/v1`` base URL: a local vLLM serve, or a hosted API
    when ``api_key`` is given. ``post`` is injectable so a caller can run without an HTTP
    library and without a serve; it is resolved at construction rather than captured as a
    default argument, so the module's default can be replaced in one place.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        served_name: str,
        config: ModelConfig,
        timeout: float = 300.0,
        seed: int = 0,
        post: Callable[[str, dict[str, Any], float], Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.served_name = served_name
        self.config = config
        self.timeout = timeout
        self.seed = seed
        # Resolved through the module, not bound at import, so a test can replace the
        # package's one poster in a single place. ``api_key`` is only bound onto the
        # default poster: with no key the poster is called exactly as before, and an
        # injected poster is never handed a key it did not ask for.
        self._post = post or http.default_post
        if api_key and post is None:
            self._post = functools.partial(self._post, api_key=api_key)

    def answer(self, prompt: str, frames: Sequence[str]) -> Answer:
        """One call. Never raises: a failure is recorded and scored as an emission of nothing.

        A run must not die on one bad row, and a row that failed must not vanish -- dropping it
        would shrink the denominator and flatter the policy. There is **no retry**: an HTTP
        error (429, 5xx), a timeout or an unreadable body becomes ``error`` on the step, the
        step is scored wrong, and the score stage counts it in ``calls.n_errored``. A retry
        would change behaviour against the recorded runs, which made none.
        """
        content, unreadable = build_content(prompt, frames, self.config)
        payload = {
            "model": self.served_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "seed": self.seed,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        try:
            body = self._post(
                f"{self.endpoint}/chat/completions", payload, self.timeout
            )
            choice = body["choices"][0]
            message = choice["message"]
            stopped = choice.get("finish_reason")
            # A reasoning model puts its answer in `reasoning_content` when it spent the whole
            # budget thinking; taking `content` alone records those rows as silent.
            text = message.get("content") or message.get("reasoning_content") or ""
        except (
            Exception
        ) as exc:  # noqa: BLE001 - every failure becomes a recorded blank
            return Answer(
                "", len(content) - 1, unreadable, f"{type(exc).__name__}: {exc}"[:200]
            )
        return Answer(
            str(text),
            len(content) - 1,
            unreadable,
            None,
            None if stopped is None else str(stopped),
        )


def assert_frame_budget(frame_counts: Sequence[int], *, serve_limit: int) -> None:
    """Raise unless every item fits the serve's ``limit_mm_per_prompt``.

    Over the limit a vLLM serve drops images rather than refusing, so the policy is judged
    on a prefix it was never shown and the run looks clean.
    """
    over = [n for n in frame_counts if n > serve_limit]
    if over:
        raise ValueError(
            f"serve accepts {serve_limit} images but {len(over)} of {len(frame_counts)} "
            f"items need up to {max(over)}. Re-serve with "
            f"--limit-mm-per-prompt image={max(over)}, or lower the per-step frame cap."
        )
