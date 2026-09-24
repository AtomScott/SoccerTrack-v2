# SoccerTrack v2

A full-pitch, multi-view soccer dataset for **Game State Reconstruction (GSR)**, **Ball Action Spotting (BAS)**, and **Multi-Object Tracking (MOT)** — plus the toolkit used to build it.

- **Paper (arXiv preprint)**: <https://arxiv.org/abs/2508.01802>
- **Dataset landing page**: <https://atomscott.github.io/SoccerTrack-v2/>
- **Hugging Face dataset**: <https://huggingface.co/datasets/atomscott/soccertrack-v2>
- **SoccerTrack Challenge 2025 (MOT)**: <https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025>

10 full-length panoramic 4K matches, with per-frame GSR annotations (2D pitch coordinates, jersey-based player IDs, roles, teams) and BAS labels for 12 action classes (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal).

## Repository layout

```
.
├── docs/          # public landing page (GitHub Pages) + developer docs + leaderboards
├── src/           # core Python modules (calibration, tracking, data_utils, evaluation, ...)
├── scripts/       # shell pipeline + one-off Python scripts
├── shell_scripts/ # additional shell helpers
├── baselines/     # minimal starter kits for GSR / BAS / MOT
├── configs/       # configuration files
├── notebooks/     # Jupyter notebooks (quickstart + analyses)
├── diagrams/      # architecture / pipeline diagrams
├── pyproject.toml # dependencies (managed by uv)
├── uv.lock        # pinned lockfile
├── Makefile       # serve docs, format
├── AGENTS.md      # working rules (workflow, conventions, commands)
├── CONTRIBUTING.md# how external contributors land a change
├── LICENSE        # MIT (code)
├── LICENSE-DATA   # CC BY 4.0 (dataset)
└── TODO.md        # → docs/TODO.md
```

The **journal paper** (Pattern Analysis and Applications) lives in a separate private repo (`AtomScott/soccertrack-v2-paper`) until acceptance; arXiv preprint above is the public version.

## Quick start

```bash
git clone https://github.com/AtomScott/SoccerTrack-v2.git
cd SoccerTrack-v2

uv sync                                 # creates .venv and installs deps from uv.lock
source .venv/bin/activate
```

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

## Common commands

```bash
make help            # list all targets
make serve-docs      # serve docs/ at http://localhost:8000
make format          # ruff autofix on src/
```

## Ground-truth pipeline

The toolkit produces per-match ground truth from raw BePro panoramic recordings. Full details in [`docs/ground_truth_creation.md`](docs/ground_truth_creation.md). One-shot:

```bash
./scripts/create_ground_truth.sh 117093          # single match
./scripts/create_ground_truth.sh 117093 118575   # multiple matches
```

Stage-by-stage scripts live under `scripts/` (`trim_video_into_halves.sh`, `convert_raw_to_pitch_plane.sh`, `calibrate_camera.sh`, `convert_pitch_plane_to_image_plane.sh`, `generate_detections.sh`, `convert_coordinates_to_bboxes.sh`, `plot_coordinates_on_video.sh`).

## Documentation

| Topic | Doc |
|---|---|
| Quickstart notebook | [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) |
| GSR annotation format | [`docs/format-gsr.md`](docs/format-gsr.md) |
| BAS annotation format | [`docs/format-bas.md`](docs/format-bas.md) |
| Baseline starter kits | [`baselines/README.md`](baselines/README.md) |
| Leaderboards | [`docs/leaderboard.html`](docs/leaderboard.html) |
| Project setup | [`docs/setup.md`](docs/setup.md) |
| CLI reference | [`docs/cli.md`](docs/cli.md) |
| Configuration | [`docs/configuration.md`](docs/configuration.md) |
| Data processing | [`docs/data_processing.md`](docs/data_processing.md) |
| Ground-truth creation | [`docs/ground_truth_creation.md`](docs/ground_truth_creation.md) |
| Camera calibration | [`docs/calibration.md`](docs/calibration.md) |
| Visualization | [`docs/visualization.md`](docs/visualization.md) |
| Landing page (HTML) | [`docs/index.html`](docs/index.html) / [`docs/index-ja.html`](docs/index-ja.html) |

## Tasks

- **GSR** — see [`docs/task-gsr.html`](docs/task-gsr.html). Reconstruct 2D pitch state (positions + IDs + roles) from panoramic video. Inspired by [SoccerNet GSR](https://www.soccer-net.org/tasks/game-state-reconstruction).
- **BAS** — see [`docs/task-bas.html`](docs/task-bas.html). Detect and classify 12 ball-action classes. Inspired by [SoccerNet BAS](https://www.soccer-net.org/tasks/ball-action-spotting).
- **MOT** — see [`docs/task-mot.html`](docs/task-mot.html). Persistent player tracking; basis of the [SoccerTrack Challenge 2025](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025).

## Open work

Live checklist: [`docs/TODO.md`](docs/TODO.md) (dataset release, format docs, landing page polish, baselines, CI).

## Contributing

PRs welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and skim [`AGENTS.md`](AGENTS.md) — together they cover the git workflow, issue triage, and code conventions.

## Citation

If you use SoccerTrack v2 in your research, please cite:

```bibtex
@article{scott2025soccertrackv2,
  title   = {{SoccerTrack v2}: A Full-Pitch Multi-View Soccer Dataset for Game State Reconstruction},
  author  = {Scott, Atom and Uchida, Ikuma and Kuroda, Kento and Kim, Yufi and Fujii, Keisuke},
  journal = {arXiv preprint arXiv:2508.01802},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01802}
}
```

The journal version (Pattern Analysis and Applications) is in preparation; bibtex will be updated on acceptance.

## License

- **Code** (this repo's source code, landing page, scripts, loaders, baselines): [MIT](LICENSE).
- **Dataset** (videos + GSR / BAS / MOT annotations, distributed via Hugging Face and Google Drive): [CC BY 4.0](LICENSE-DATA).

Both permit commercial use. Please cite the paper if you use the dataset in research.

## Contact

Atom Scott — <atom.james.scott@gmail.com> — Nagoya University / [Playbox](https://playbox.co.jp/).

## Match labels

The paper labels the ten matches M1 to M10 in ascending order of the provider identifier used by the release files. The per-match counts below were moved here from the paper (frames and coordinates are the annotated video frames of each match and the player-frame records over them; boxes are the human-annotated boxes of the curated four-minute tracking clip of each match).

| Label | Provider id | Frames | Duration (min) | Coordinates | Events | Boxes |
|---|---|---|---|---|---|---|
| M1 | 117092 | 134,951 | 90 | 2,967,629 | 2,086 | 132,000 |
| M2 | 117093 | 138,125 | 92 | 3,038,640 | 2,252 | 131,472 |
| M3 | 118575 | 145,777 | 97 | 3,198,454 | 2,117 | 132,000 |
| M4 | 118576 | 146,152 | 97 | 3,212,350 | 2,131 | 131,450 |
| M5 | 118577 | 145,427 | 97 | 3,193,894 | 1,946 | 132,000 |
| M6 | 118578 | 141,851 | 95 | 3,120,655 | 2,106 | 132,000 |
| M7 | 128057 | 138,025 | 92 | 3,036,550 | 1,932 | 131,472 |
| M8 | 128058 | 137,276 | 92 | 3,020,403 | 2,144 | 131,582 |
| M9 | 132831 | 136,151 | 91 | 2,996,079 | 2,440 | 131,472 |
| M10 | 132877 | 137,039 | 91 | 3,015,913 | 2,278 | 131,076 |
| Total | | 1,400,774 | 934 | 30,800,567 | 21,432 | 1,316,524 |

The ball track is clamped to the pitch rectangle in M9 and M10. The released match-level test split is M7 and M9 (the SoccerTrack Challenge 2025 pair).
