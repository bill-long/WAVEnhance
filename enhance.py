"""
AudioSR Bridge Script — called by WAVEnhance.exe to process WAV files.

Usage:
    python enhance.py --input <path> --output <path> [--guidance 3.5] [--steps 50] [--seed 42]

Outputs progress lines to stdout in the format:
    PROGRESS:<percent>:<message>
    RESULT:SUCCESS:<output_path>
    RESULT:ERROR:<error_message>
"""

import argparse
import sys
import os
import tempfile


def log_progress(percent: int, message: str):
    print(f"PROGRESS:{percent}:{message}", flush=True)


def log_result(success: bool, message: str):
    tag = "SUCCESS" if success else "ERROR"
    print(f"RESULT:{tag}:{message}", flush=True)


def _detect_device(preference: str) -> str:
    """Detect the best available compute device."""
    import torch

    if preference == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if preference == "directml":
        try:
            import torch_directml
            return str(torch_directml.device())
        except ImportError:
            print("WARNING: torch-directml not installed. Falling back to CPU.", flush=True)

    return "cpu"


def _enhance_mono(model, mono_path: str, args) -> "np.ndarray":
    """Run AudioSR on a mono WAV file and return the enhanced waveform as numpy array."""
    import torch
    import numpy as np
    from audio_upscaler.pipeline import super_resolution

    # Monkey-patch to fix off-by-one bug in DDIM timestep generation:
    # range(0, 1000, 333) produces [0,333,666,999], and +1 gives 1000 which is OOB.
    import audio_upscaler.latent_diffusion.models.ddim as ddim_mod
    _orig_make_ddim_timesteps = ddim_mod.make_ddim_timesteps

    def _patched_make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
        steps = _orig_make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose)
        return np.clip(steps, 0, num_ddpm_timesteps - 1)

    ddim_mod.make_ddim_timesteps = _patched_make_ddim_timesteps
    try:
        waveform = super_resolution(
            model,
            mono_path,
            seed=args.seed,
            guidance_scale=args.guidance,
            ddim_steps=args.steps,
        )
    finally:
        ddim_mod.make_ddim_timesteps = _orig_make_ddim_timesteps

    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()

    waveform = waveform.squeeze()

    # Replace any NaN/Inf with 0 (can occur with very few DDIM steps)
    if not np.all(np.isfinite(waveform)):
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

    return waveform


def main():
    parser = argparse.ArgumentParser(description="AudioSR audio enhancement bridge")
    parser.add_argument("--input", "-i", required=True, help="Input WAV file path")
    parser.add_argument("--output", "-o", required=True, help="Output WAV file path")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, or directml (default: auto)")
    args = parser.parse_args()

    if args.steps < 5:
        print(f"WARNING: --steps {args.steps} is very low. Minimum recommended is 10, default is 50.", flush=True)

    if not os.path.isfile(args.input):
        log_result(False, f"Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    try:
        log_progress(5, "Loading AudioSR model...")
        import torch
        import numpy as np
        import soundfile as sf
        from audio_upscaler.pipeline import build_model

        device = _detect_device(args.device)
        log_progress(10, f"Using device: {device}")

        audiosr_model = build_model(model_name="basic", device=device)
        log_progress(25, "Model loaded")

        # Read input to determine channel count
        data, sr = sf.read(args.input, dtype="float32")
        is_stereo = data.ndim == 2 and data.shape[1] == 2

        if is_stereo:
            log_progress(30, "Stereo input — processing each channel separately")
            with tempfile.TemporaryDirectory() as tmpdir:
                left_path = os.path.join(tmpdir, "left.wav")
                right_path = os.path.join(tmpdir, "right.wav")

                sf.write(left_path, data[:, 0], sr, subtype="PCM_16")
                sf.write(right_path, data[:, 1], sr, subtype="PCM_16")

                log_progress(35, "Processing left channel...")
                left_enhanced = _enhance_mono(audiosr_model, left_path, args)

                log_progress(60, "Processing right channel...")
                args.seed += 1  # Slightly different seed to avoid identical output
                right_enhanced = _enhance_mono(audiosr_model, right_path, args)

            # Match lengths
            min_len = min(len(left_enhanced), len(right_enhanced))
            stereo_out = np.column_stack([
                left_enhanced[:min_len],
                right_enhanced[:min_len]
            ])
            stereo_out = np.clip(stereo_out, -1.0, 1.0)
            sf.write(args.output, stereo_out, 48000, subtype="PCM_16")
        else:
            log_progress(35, f"Processing: {os.path.basename(args.input)}")
            enhanced = _enhance_mono(audiosr_model, args.input, args)
            enhanced = np.clip(enhanced, -1.0, 1.0)
            sf.write(args.output, enhanced, 48000, subtype="PCM_16")

        log_progress(90, "Output saved")
        log_result(True, args.output)

    except Exception as e:
        log_result(False, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
