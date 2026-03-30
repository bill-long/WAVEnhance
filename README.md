# WAVEnhance

AI-powered audio super-resolution for WAV files using [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference). Restores lost high-frequency content in music that was transcoded from lossy sources (MP3, AAC, etc.).

## What it does

Many WAV files are actually transcodes from lossy formats — they have the container and bit depth of lossless audio, but the high-frequency content above ~15 kHz was permanently removed during compression. This makes them sound flat and lifeless on headphones, even though they *look* like high-quality files.

WAVEnhance uses a diffusion-distilled AI model (FlashSR) to reconstruct the missing high-frequency content in a single inference step, restoring brilliance and air to the audio. It can also optionally upsample to 96 kHz.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [Python 3.11+](https://www.python.org/downloads/)
- An NVIDIA GPU with CUDA or AMD GPU with ROCm is recommended; CPU works but is much slower

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd WAVEnhance

# Create a Python virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch (CUDA — use the appropriate command for your setup)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# For ROCm (AMD GPUs on Linux):
# Install python-pytorch-opt-rocm from your distro, then create venv with:
# python -m venv --system-site-packages .venv
# pip install torchaudio torchvision --no-deps --index-url https://download.pytorch.org/whl/rocm7.0

# For CPU-only:
# pip install torch torchaudio

# Clone FlashSR and install
git clone --depth 1 https://github.com/jakeoneijk/FlashSR_Inference.git .flashsr
pip install -e .flashsr

# Install audio dependencies
pip install scipy soundfile soxr

# Download FlashSR model weights (~3.1 GB)
python -c "
from huggingface_hub import hf_hub_download
for f in ['student_ldm.pth', 'sr_vocoder.pth', 'vae.pth']:
    hf_hub_download(repo_id='jakeoneijk/FlashSR_weights', filename=f, repo_type='dataset', local_dir='ModelWeights')
"

# Restore .NET packages
dotnet restore
```

## Usage

```bash
# Enhance an entire directory of WAV files
dotnet run -- -i "/path/to/album" -o "/path/to/output"

# Enhance a single file
dotnet run -- -i "/path/to/song.wav" -o "/path/to/output"

# Enhance and upsample to 96 kHz
dotnet run -- -i "/path/to/album" -o "/path/to/output" -s 96000
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | *(required)* | Input WAV file or directory |
| `--output` | `-o` | *(required)* | Output directory |
| `--sample-rate` | `-s` | `48000` | Target sample rate: `48000` or `96000` |
| `--seed` | | `42` | Random seed for reproducibility |
| `--device` | `-d` | `auto` | Compute device: `auto`, `cpu`, `cuda` |

## How it works

1. **C# CLI** (`Program.cs`) discovers and validates input WAV files, then orchestrates processing
2. **Python bridge** (`enhance.py`) loads the FlashSR model and runs inference:
   - Audio is resampled to 48 kHz and split into 5.12-second chunks (FlashSR's fixed window)
   - Stereo files are split into separate channels and processed independently
   - Adjacent chunks are merged with a linear crossfade to prevent seams
   - FlashSR uses a single diffusion step (distilled from AudioSR's 50-step process)
3. If `--sample-rate 96000` is specified, the enhanced 48 kHz output is upsampled using high-quality sinc interpolation (NAudio MediaFoundation resampler at quality level 60)

## Performance

FlashSR is dramatically faster than AudioSR thanks to single-step diffusion distillation:

| Hardware | Approx. speed |
|----------|---------------|
| NVIDIA GPU (CUDA) | ~50–200× realtime |
| AMD GPU (ROCm) | ~30–100× realtime |
| CPU | ~1–5× realtime |

## Architecture

```
WAVEnhance/
├── Program.cs          # C# CLI — arg parsing, batch processing, resampling
├── enhance.py          # Python bridge — FlashSR model loading and inference
├── WAVEnhance.csproj   # .NET project file (depends on NAudio)
├── ModelWeights/       # FlashSR model weights (auto-downloaded, not committed)
├── .flashsr/           # FlashSR source (cloned from GitHub, not committed)
├── .venv/              # Python virtual environment (not committed)
└── .gitignore
```

## Acknowledgments

- [FlashSR](https://github.com/jakeoneijk/FlashSR_Inference) by Jaekwon Im & Juhan Nam — one-step diffusion-distilled audio super-resolution
- [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) by Haohe Liu et al. — the original diffusion-based audio super-resolution model (FlashSR's teacher)
- [NAudio](https://github.com/naudio/NAudio) — .NET audio library for WAV I/O and resampling
