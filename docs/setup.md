# Project Setup

This guide walks through the setup process for SoccerTrack v2.

## Prerequisites

- Python 3.12 or higher
- [`uv`](https://docs.astral.sh/uv/) (used to manage the virtualenv and lockfile)
- Git
- FFmpeg (for video processing)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AtomScott/SoccerTrack-v2.git
   cd SoccerTrack-v2
   ```

2. **Create the environment and install dependencies**

   The project is uv-managed. `uv sync` creates a `.venv/` and installs the
   pinned dependencies from `pyproject.toml` + `uv.lock`.

   ```bash
   uv sync
   source .venv/bin/activate  # macOS / Linux
   # .venv\Scripts\activate   # Windows
   ```

   Alternatively, run commands without activating: `uv run python -m src.main ...`.

## Download the dataset

Use the helper script to fetch from Hugging Face (`atomscott/soccertrack-v2`):

```bash
./scripts/download.sh --dest ./data                                    # full dataset
./scripts/download.sh --dest ./data --match 117092 --match 117093      # subset
```

`--revision` pins a dataset tag / commit; see `scripts/download.sh --help` for
full usage. The Hugging Face CLI (installed with `huggingface_hub`) is
required.

## Verify installation

Run a quick load against the freshly downloaded data:

```python
from src.data_utils.soccertrack_v2 import load_match
m = load_match("./data", match_id="117093")
frames = m.gsr_frames(half=1)
print(len(frames), "GSR frames in 1st half")
print(len(m.bas_events()), "BAS events")
```

See [`../notebooks/quickstart.ipynb`](../notebooks/quickstart.ipynb) for a
fuller runnable example.

## Development setup

Dev dependencies (lint, test, notebooks) are declared in the `dev` optional
group:

```bash
uv sync --group dev
```

Format and lint with ruff (wrapper in the Makefile):

```bash
make format   # ruff autofix on src/
```

## Common issues

### FFmpeg installation

- **Ubuntu / Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: download from <https://ffmpeg.org/download.html>

### Accessing the dataset from code

If you prefer to let the loader fetch from Hugging Face on demand rather than
running `download.sh` first:

```python
from src.data_utils.load_from_hf import load_match_from_hf
m = load_match_from_hf("117093")
```

`snapshot_download` caches under `HF_HOME` so repeat calls are free.

## Next steps

1. Read the [Configuration Guide](configuration.md).
2. Browse [`../notebooks/quickstart.ipynb`](../notebooks/quickstart.ipynb).
3. Explore the [CLI reference](cli.md).
