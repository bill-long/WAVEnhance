# WAVEnhance

AI-powered audio super-resolution for WAV files using [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution). Restores lost high-frequency content in music that was transcoded from lossy sources (MP3, AAC, etc.).

## What it does

Many WAV files are actually transcodes from lossy formats — they have the container and bit depth of lossless audio, but the high-frequency content above ~15 kHz was permanently removed during compression. This makes them sound flat and lifeless on headphones, even though they *look* like high-quality files.

WAVEnhance uses a diffusion-based AI model (AudioSR) to reconstruct the missing high-frequency content, restoring brilliance and air to the audio. It can also optionally upsample to 96 kHz.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [Python 3.11](https://www.python.org/downloads/) (3.12+ may have compatibility issues with AudioSR dependencies)
- An NVIDIA GPU with CUDA is recommended for reasonable speed; CPU works but is much slower

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

# For CPU-only:
# pip install torch torchaudio

# Install AudioSR and audio dependencies
pip install versatile-audio-upscaler scipy soundfile pydub soxr

# Restore .NET packages
dotnet restore
```

The AudioSR model (~5.9 GB) is downloaded automatically from HuggingFace on the first run.

## Usage

```bash
# Enhance an entire directory of WAV files
dotnet run -- -i "C:\path\to\album" -o "C:\path\to\output"

# Enhance a single file
dotnet run -- -i "C:\path\to\song.wav" -o "C:\path\to\output"

# Enhance and upsample to 96 kHz
dotnet run -- -i "C:\path\to\album" -o "C:\path\to\output" -s 96000

# Quick preview with fewer steps (faster, slightly lower quality)
dotnet run -- -i "C:\path\to\album" -o "C:\path\to\output" --steps 10
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | *(required)* | Input WAV file or directory |
| `--output` | `-o` | *(required)* | Output directory |
| `--sample-rate` | `-s` | `48000` | Target sample rate: `48000` or `96000` |
| `--steps` | | `50` | DDIM inference steps — higher = better quality, slower |
| `--guidance` | | `3.5` | AudioSR guidance scale |
| `--seed` | | `42` | Random seed for reproducibility |
| `--device` | `-d` | `auto` | Compute device: `auto`, `cpu`, `cuda` |
| `--chunk-seconds` | | `10.24` | Audio chunk length in seconds |
| `--overlap` | | `1.0` | Crossfade overlap between chunks in seconds |

## How it works

1. **C# CLI** (`Program.cs`) discovers and validates input WAV files, then orchestrates processing
2. **Python bridge** (`enhance.py`) loads the AudioSR model and runs inference:
   - Stereo files are split into separate channels and processed independently
   - Long audio is split into overlapping chunks (~10s each) to fit in GPU memory
   - Adjacent chunks are merged with a linear crossfade to prevent seams
3. If `--sample-rate 96000` is specified, the enhanced 48 kHz output is upsampled using high-quality sinc interpolation (NAudio MediaFoundation resampler at quality level 60)

## Performance

Processing time depends heavily on hardware and `--steps`:

| Hardware | Steps | Approx. speed |
|----------|-------|---------------|
| NVIDIA GPU (CUDA) | 50 | ~2–5× realtime |
| CPU | 50 | ~0.02× realtime (~50 min per minute of audio) |
| CPU | 10 | ~0.1× realtime (~10 min per minute of audio) |

**Tip:** Start with `--steps 10` to preview results quickly. The quality difference between 10 and 50 steps is often subtle.

## Architecture

```
WAVEnhance/
├── Program.cs          # C# CLI — arg parsing, batch processing, resampling
├── enhance.py          # Python bridge — AudioSR model loading and inference
├── WAVEnhance.csproj   # .NET project file (depends on NAudio)
├── .venv/              # Python virtual environment (not committed)
├── pytorch_model.bin   # AudioSR model weights (auto-downloaded, not committed)
└── .gitignore
```

## Acknowledgments

- [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) by Haohe Liu et al. — the diffusion-based audio super-resolution model
- [NAudio](https://github.com/naudio/NAudio) — .NET audio library for WAV I/O and resampling
