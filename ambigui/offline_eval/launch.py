# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the three offline stages from one YAML file.

The offline evaluation is three programs, run in order on one output directory:

1. ``run``   -- ``offline_evaluation.run.run_ask_protocol``: the policy acts or asks at
   every recorded step, and a simulated user answers its questions.
2. ``judge`` -- ``offline_evaluation.run.judge_asks``: a judge model says which open items
   each recorded question covered.
3. ``score`` -- ``offline_evaluation.run.score_ask_protocol``: the metrics, computed from
   the files alone, with no model and no network.

Each program takes its settings as command-line flags. This launcher only reads a *run-set*
file (``configs/offline/run_sets/*.yaml``), turns it into those flags, and calls the
program's own ``main``. It adds no behaviour: the flags it builds are printed, and running
the program by hand with the same flags gives the same result.

A *run set* fixes the simulated user and the judge for every policy in it. Results from two
run sets must never be mixed, because the simulated user writes the words the policy reads.

Values written ``${NAME}`` in the YAML come from the environment, and ``${NAME:-default}``
falls back to ``default``. Endpoint keys are read from the environment by the programs
themselves (``AMBIGUI_POLICY_API_KEY``, ``AMBIGUI_ANSWERER_API_KEY``,
``AMBIGUI_JUDGE_API_KEY``) and never pass through this file or the command line.

Each stage also writes or updates ``run_manifest.json`` in the output directory: the git
commit, the run-set file's hash, the model ids and revisions, package versions, the seed,
the host, and the exact flags each stage was called with, with each stage's status
(``complete`` or ``failed``). A new ``run`` first removes any manifest left by an earlier
run. When a stage fails, the manifest is still written, with that stage and the top-level
``status`` marked ``failed``. No program reads it back.

**Policy output budget.** An open-weight run-set file may hold ``policy_max_tokens``: the
output-token budget each recorded pass used, per model and screen condition, read from
that pass's ``run.json``. When the file lists the pass being run, the launcher passes it as
``--max-tokens``; otherwise the catalogue entry's budget applies. The frontier file has no
such table.

Usage::

    python -m ambigui.offline_eval.launch run   --config configs/offline/run_sets/open_weight_glm46v.yaml \\
        --model qwen3-vl-8b-instruct --screens on --endpoint http://localhost:8000/v1
    python -m ambigui.offline_eval.launch judge --config ... --model qwen3-vl-8b-instruct --screens on
    python -m ambigui.offline_eval.launch score --config ... --model qwen3-vl-8b-instruct --screens on
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: The audited evaluation code. Its packages import each other by top-level name
#: (``offline_evaluation``, ``evaluation``, ...), so this folder goes on ``sys.path``.
SRC = Path(__file__).resolve().parent / "src"
REPO = Path(__file__).resolve().parents[2]
MODELS_FILE = REPO / "configs" / "offline" / "models.yaml"
PACKAGES = ("requests", "PyYAML", "pillow", "openai")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def expand(value: Any) -> Any:
    """``value`` with every ``${NAME}`` / ``${NAME:-default}`` replaced from the environment.

    A variable that is unset and has no default stops the launcher, naming the variable,
    rather than passing an empty string on to a program that would then fail later.
    """
    if isinstance(value, str):

        def one(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            got = os.environ.get(name)
            if got:
                return got
            if default is not None:
                return default
            raise SystemExit(f"FATAL: environment variable {name} is not set")

        return _VAR.sub(one, value)
    if isinstance(value, Mapping):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"FATAL: {path} is not a mapping")
    return loaded


def role(config: Mapping[str, Any], name: str, model: str) -> dict[str, Any]:
    """The answerer or judge settings for ``model``: the run set's, or its override."""
    base = dict(config.get(name) or {})
    override = (base.pop("per_policy", None) or {}).get(model) or {}
    return {**base, **override}


def out_dir(config: Mapping[str, Any], model: str, screens: str) -> Path:
    """``<out_root>/<model>/screens-<on|off>``, the layout the published run sets use."""
    return Path(config["out_root"]) / model / f"screens-{screens}"


def policy_max_tokens(config: Mapping[str, Any], model: str, screens: str) -> int | None:
    """The recorded policy budget for this pass from ``policy_max_tokens``, or ``None``.

    ``policy_max_tokens`` maps a model key to ``{screens-on: N, screens-off: N}``. A pass
    the table does not list gets ``None``, so the catalogue entry's budget applies.
    """
    table = config.get("policy_max_tokens") or {}
    if not isinstance(table, Mapping):
        raise SystemExit("FATAL: policy_max_tokens must map model keys to budgets")
    entry = table.get(model) or {}
    if not isinstance(entry, Mapping):
        raise SystemExit(
            f"FATAL: policy_max_tokens.{model} must map screens-on/screens-off to budgets"
        )
    value = entry.get(f"screens-{screens}")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(
            f"FATAL: policy_max_tokens.{model}.screens-{screens} must be a positive "
            f"integer, got {value!r}"
        )
    return value


def run_argv(config: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    run = config.get("run") or {}
    answerer = role(config, "answerer", args.model)
    argv = [
        "--benchmark", str(config["benchmark"]),
        "--image-root", str(config["image_root"]),
        "--out-dir", str(out_dir(config, args.model, args.screens)),
        "--model", args.model,
        "--endpoint", args.endpoint,
        "--served-model", args.served_model or args.model,
        "--answerer-endpoint", str(answerer["endpoint"]),
        "--answerer-model", str(answerer["served_model"]),
        "--screens", args.screens,
        "--seed", str(config.get("seed", 0)),
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    budget = policy_max_tokens(config, args.model, args.screens)
    if budget is not None:
        argv += ["--max-tokens", str(budget)]
    # The image limit is only meaningful, and only required, when screens are attached.
    if args.screens == "on" and "serve_image_limit" in run:
        argv += ["--serve-image-limit", str(run["serve_image_limit"])]
    return argv


def judge_argv(config: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    judge = role(config, "judge", args.model)
    argv = [
        str(out_dir(config, args.model, args.screens)),
        "--judge-endpoint", str(judge["endpoint"]),
        "--judge-model", str(judge["served_model"]),
        "--seed", str(config.get("seed", 0)),
    ]
    argv += ["--judge-registry-key", str(judge["registry_key"])]
    return argv


def serve_entry(name: str) -> dict[str, Any]:
    models = load_yaml(MODELS_FILE)
    for section in ("policies", "answerers_and_judges"):
        entry = (models.get(section) or {}).get(name)
        if entry is not None:
            return dict(entry)
    raise SystemExit(f"FATAL: {name!r} is not in {MODELS_FILE}")


def _git_commit() -> str | None:
    """The checkout's commit, or ``$AMBIGUI_GIT_COMMIT`` where there is no ``.git`` (Docker)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("AMBIGUI_GIT_COMMIT") or None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes, or ``None`` when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def _versions() -> dict[str, str | None]:
    from importlib import metadata  # noqa: PLC0415

    found: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def _revisions(names: Sequence[str]) -> dict[str, Any]:
    if not MODELS_FILE.is_file():
        return {}
    out: dict[str, Any] = {}
    for name in names:
        try:
            entry = serve_entry(name)
        except SystemExit:
            entry = {}
        out[name] = {"hf_id": entry.get("hf_id"), "revision": entry.get("revision")}
    return out


def write_manifest(
    directory: Path,
    stage: str,
    argv: Sequence[str],
    config_path: Path,
    config: Mapping[str, Any],
    model: str,
    status: str = "complete",
    error: str = "",
) -> None:
    """Add this stage, with its ``status``, to ``run_manifest.json``. Read by nothing.

    The top-level ``status`` is the status of the stage written last, so a failed stage
    marks the manifest ``failed``.
    """
    path = directory / "run_manifest.json"
    manifest: dict[str, Any] = {}
    # A new run starts a new manifest; judge and score add to the run's.
    if stage != "run" and path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    answerer = role(config, "answerer", model)
    judge = role(config, "judge", model)
    manifest.update(
        {
            "status": status,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "run_set": config.get("run_set"),
            "config_file": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "seed": config.get("seed", 0),
            "models": {
                **manifest.get("models", {}),
                **_revisions(
                    [model]
                    + [str(r["served_model"]) for r in (answerer, judge) if r.get("served_model")]
                ),
            },
            "packages": _versions(),
            "python": platform.python_version(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
        }
    )
    stages = manifest.setdefault("stages", [])
    stages.append(
        {
            "stage": stage,
            "status": status,
            **({"error": error} if error else {}),
            "argv": list(argv),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("run", "judge", "score"):
        p = sub.add_parser(stage)
        p.add_argument("--config", type=Path, required=True, help="a run-set YAML file")
        p.add_argument("--model", required=True, help="a key from the model catalogue")
        p.add_argument("--screens", choices=("on", "off"), required=True)
        if stage == "run":
            p.add_argument(
                "--endpoint",
                default=os.environ.get("AMBIGUI_POLICY_BASE_URL", ""),
                help="the policy's OpenAI-compatible base URL "
                "(default: $AMBIGUI_POLICY_BASE_URL)",
            )
            p.add_argument(
                "--served-model",
                default="",
                help="the name the policy serve answers to (default: the model key)",
            )
            p.add_argument(
                "--limit",
                type=_non_negative,
                default=None,
                help="run only the first N steps (default: every step). Only those "
                "steps' frames are needed",
            )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Only the parts a stage uses are expanded, so scoring a finished run does not need
    # the benchmark or the endpoints set in the environment.
    raw = load_yaml(args.config)
    needed = {
        "run": (
            "run_set", "benchmark", "image_root", "out_root", "seed", "run", "answerer",
            "policy_max_tokens",
        ),
        "judge": ("run_set", "out_root", "seed", "judge"),
        "score": ("run_set", "out_root", "seed"),
    }[args.stage]
    config = {key: expand(raw[key]) for key in needed if key in raw}
    directory = out_dir(config, args.model, args.screens)
    _src_on_path()
    if args.stage == "run":
        if not args.endpoint:
            raise SystemExit("FATAL: pass --endpoint or set AMBIGUI_POLICY_BASE_URL")
        stage_argv = run_argv(config, args)
        from offline_evaluation.run import run_ask_protocol as program  # noqa: PLC0415
    elif args.stage == "judge":
        stage_argv = judge_argv(config, args)
        from offline_evaluation.run import judge_asks as program  # noqa: PLC0415
    else:
        stage_argv = [str(directory)]
        from offline_evaluation.run import score_ask_protocol as program  # noqa: PLC0415
    if args.stage == "run":
        # A manifest left by an earlier run describes that run, not this one.
        (directory / "run_manifest.json").unlink(missing_ok=True)
    print(f"[launch] {program.__name__} {shlex.join(stage_argv)}", flush=True)
    try:
        result = int(program.main(stage_argv) or 0)
    except BaseException as failure:
        ok = isinstance(failure, SystemExit) and failure.code in (0, None)
        write_manifest(
            directory, args.stage, stage_argv, args.config, config, args.model,
            status="complete" if ok else "failed",
            error="" if ok else f"{type(failure).__name__}: {str(failure)[:300]}",
        )
        raise
    write_manifest(
        directory, args.stage, stage_argv, args.config, config, args.model,
        status="complete" if result == 0 else "failed",
        error="" if result == 0 else f"exit {result}",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
