# Project Setup

This guide walks through the setup process for SoccerTrack-V2.

## Prerequisites

- Python 3.12 or higher
- Git
- FFmpeg (for video processing)
- OpenCV dependencies

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/SoccerTrack-V2.git
   cd SoccerTrack-V2
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure Setup

1. **Create Required Directories**
   ```bash
   mkdir -p data/{raw,interim,processed}
   mkdir -p data/interim/{events,tracking,calibrated_video}
   mkdir -p models
   ```

2. **Configure Environment**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your settings
   nano .env
   ```

## Data Setup

1. **Download Sample Data**
   ```bash
   # Download sample match data
   python scripts/download_sample_data.py
   ```

2. **Verify Installation**
   ```bash
   # Run test command
   python -m src.main command=process-raw-data match_id=117093
   ```

## Development Setup

1. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Configure Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Setup Ruff**
   ```bash
   # Install ruff
   pip install ruff
   
   # Run formatter
   ruff format .
   ```

## Common Issues

### FFmpeg Installation

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
Download from [FFmpeg website](https://ffmpeg.org/download.html)

### OpenCV Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3-opencv
```

#### macOS
```bash
brew install opencv
```

## Verification

Run the verification script to check your setup:
```bash
python scripts/verify_setup.py
```

This will check:
- Python version
- Required dependencies
- Directory structure
- FFmpeg installation
- OpenCV installation
- Environment configuration

## Next Steps

1. Read the [Configuration Guide](configuration.md)
2. Try the [Quick Start](../README.md#quick-start) examples
3. Explore the [Command Line Interface](cli.md) 