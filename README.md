# AmbiGUI

AmbiGUI is a benchmark and code base for GUI agents. A GUI agent is a model that uses a website
or an app for a person: it reads the screen and the request, then clicks, types or scrolls.

Some requests leave out a detail that changes the result. For example, "book a table" does not
say for how many people. We call such a request **ambiguous**. A good agent then asks the person
a short **clarifying question** before it acts. When nothing important is missing (a **clear**
request), a good agent acts without asking. AmbiGUI measures whether an agent makes this choice
well.

## The parts

| part | what it does | documentation |
|---|---|---|
| Offline evaluation | Shows the agent recorded tasks step by step and checks, at each step, whether it acts correctly or asks a useful question. No live browser is needed. | [docs/offline.md](docs/offline.md) |
| Online evaluation | Runs the agent on live websites in a real browser, with a simulated user who answers its questions, and scores task success and asking quality. | [docs/online.md](docs/online.md) |
| In-context learning (ICL) baseline (ablation) | An ablation, not a main experiment. Runs a model whose weights are never changed on the online benchmark tasks twice: once with no examples (zero-shot) and once with four fixed worked examples placed in its prompt (ICL-4), with everything else matched, and compares how often it succeeds and asks. Includes the code that picks the four examples from training rollouts. ICL results are known to be hard to reproduce; a re-run is a new measurement. | [docs/icl.md](docs/icl.md) |
| Benchmark | The benchmark tasks as pointers into the source datasets plus our own annotations. You rebuild the full tasks from your own copy of each source dataset. | [benchmark/README.md](benchmark/README.md) |

No trained model weights, adapters or checkpoints are released with this repository. It holds
code, configurations and documentation only.

## Install

Python 3.12 is required. Each part has its own extra, because the parts pin different versions
of some shared packages. Install one extra per environment.

```bash
git clone https://github.com/facebookresearch/AmbiGUI.git ambigui && cd ambigui
python3.12 -m venv .venv && . .venv/bin/activate

pip install -e ".[offline]"        # offline evaluation
pip install -e ".[online]"         # online evaluation
pip install -e ".[online-serve]"   # vLLM, to serve open-weights models (often best in its own environment)
pip install -e ".[icl]"            # in-context learning baseline
pip install -e ".[icl-serve]"      # vLLM, to serve the ICL policy (often best in its own environment)
pip install -r requirements/benchmark.lock   # pyarrow, to rebuild the benchmark files (benchmark/build.py)
```

The online evaluation and the ICL baseline also need the browser:
`PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.playwright.dev python -m playwright install --with-deps chromium`
(the download address built into Playwright 1.44 no longer serves files; see
[docs/online.md](docs/online.md)).

For the exact package versions used to check this release, install from the lock file first and
then the package without dependencies, for example:

```bash
pip install -r requirements/offline.lock && pip install --no-deps -e .
```

The lock files are `requirements/offline.lock`, `requirements/online.lock`,
`requirements/icl.lock` and `requirements/benchmark.lock`. [DEPENDENCIES.md](DEPENDENCIES.md) lists every pinned package of
each extra. Each part also has a container recipe: `docker/offline.Dockerfile`,
`docker/online.Dockerfile` and `docker/icl.Dockerfile`. Build them from the repository root,
for example `docker build -f docker/offline.Dockerfile -t ambigui-offline .`. The page of each
part gives the full setup.

## License

AmbiGUI is licensed under the Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0). The full text is in [LICENSE](LICENSE).

Third-party code included in this repository, and the packages it depends on, keep their own
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Third-party content

Third party content pulled from other locations are subject to its own licenses and you may have other legal obligations or restrictions that govern your use of that content.

The source datasets of the benchmark (Mind2Web, META-GUI and OSWorld)
are not included in this repository. Each is governed by its own terms. You must get each
dataset from its own provider and follow its license and terms of use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

The citation will be added when the paper is published.

```bibtex
@misc{ambigui,
  title  = {AmbiGUI},
  author = {TBD},
  year   = {TBD},
  note   = {Citation placeholder}
}
```
