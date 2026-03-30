"""
FlashSR Bridge Script — called by WAVEnhance.exe to process WAV files.

Usage:
    python enhance.py --input <path> --output <path> [--seed 42]

Outputs progress lines to stdout in the format:
    PROGRESS:<percent>:<message>
    RESULT:SUCCESS:<output_path>
    RESULT:ERROR:<error_message>
"""

import argparse
import sys
import os

# FlashSR processes fixed 5.12-second chunks (245760 samples at 48 kHz)
FLASHSR_CHUNK_SAMPLES = 245760
FLASHSR_SAMPLE_RATE = 48000


def log_progress(percent: int, message: str):
    print(f"PROGRESS:{percent}:{message}", flush=True)


def log_result(success: bool, message: str):
    tag = "SUCCESS" if success else "ERROR"
    print(f"RESULT:{tag}:{message}", flush=True)


def _detect_device(preference: str) -> str:
    """Detect the best available compute device.

    torch.cuda.is_available() returns True for both NVIDIA CUDA and AMD ROCm,
    so a single 'cuda' check covers both GPU backends.
    """
    import torch

    if preference == "cpu":
        return "cpu"

    # torch.cuda covers both NVIDIA CUDA and AMD ROCm (HIP)
    if torch.cuda.is_available():
        return "cuda"

    if preference == "directml":
        try:
            import torch_directml
            return str(torch_directml.device())
        except ImportError:
            print("WARNING: torch-directml not installed. Falling back to CPU.", flush=True)

    return "cpu"


def _device_label(device: str) -> str:
    """Return a human-friendly label for the active device."""
    import torch

    if device != "cuda":
        return device

    name = torch.cuda.get_device_name(0)
    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"ROCm/HIP {hip} — {name}"
    return f"CUDA {torch.version.cuda} — {name}"


def _enhance_mono(model, data: "np.ndarray", sr: int, device: str, seed: int) -> "np.ndarray":
    """Run FlashSR on a mono waveform and return the enhanced waveform as numpy array."""
    import torch
    import numpy as np
    import soxr

    torch.manual_seed(seed)

    # Resample to 48 kHz if needed
    if sr != FLASHSR_SAMPLE_RATE:
        data = soxr.resample(data, sr, FLASHSR_SAMPLE_RATE)

    total_samples = len(data)
    chunk_samples = FLASHSR_CHUNK_SAMPLES
    overlap_samples = int(0.5 * FLASHSR_SAMPLE_RATE)  # 0.5s overlap
    stride_samples = chunk_samples - overlap_samples

    num_chunks = max(1, 1 + (total_samples - chunk_samples + stride_samples - 1) // stride_samples)
    if total_samples <= chunk_samples:
        num_chunks = 1

    enhanced_chunks = []

    for ci in range(num_chunks):
        start = ci * stride_samples
        end = min(start + chunk_samples, total_samples)
        chunk_data = data[start:end]

        # Pad to exact chunk size if needed (FlashSR requires fixed length)
        original_len = len(chunk_data)
        if original_len < chunk_samples:
            chunk_data = np.pad(chunk_data, (0, chunk_samples - original_len))

        # FlashSR expects [batch, time] tensor at 48 kHz
        audio_tensor = torch.from_numpy(chunk_data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced = model(audio_tensor, lowpass_input=False)

        waveform = enhanced.squeeze(0).cpu().numpy()

        # Trim padding back off
        waveform = waveform[:original_len]

        # Normalize output RMS to match input RMS (prevents per-chunk gain drift)
        input_rms = np.sqrt(np.mean(chunk_data[:original_len] ** 2)) + 1e-10
        output_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-10
        waveform = waveform * (input_rms / output_rms)

        enhanced_chunks.append(waveform)

        if num_chunks > 1:
            log_progress(0, f"  Chunk {ci + 1}/{num_chunks} done")

    # Merge chunks with linear crossfade in overlap regions
    if len(enhanced_chunks) == 1:
        result = enhanced_chunks[0]
    else:
        overlap_out = overlap_samples
        stride_out = stride_samples
        total_out = stride_out * (len(enhanced_chunks) - 1) + len(enhanced_chunks[-1])
        result = np.zeros(total_out, dtype=np.float32)
        weights = np.zeros(total_out, dtype=np.float32)

        for ci, chunk in enumerate(enhanced_chunks):
            offset = ci * stride_out
            chunk_len = len(chunk)
            fade = np.ones(chunk_len, dtype=np.float32)

            if ci > 0 and overlap_out > 0:
                ramp_len = min(overlap_out, chunk_len)
                fade[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

            if ci < len(enhanced_chunks) - 1 and overlap_out > 0:
                ramp_len = min(overlap_out, chunk_len)
                fade[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)

            end_idx = offset + chunk_len
            if end_idx > len(result):
                chunk_len = len(result) - offset
                chunk = chunk[:chunk_len]
                fade = fade[:chunk_len]

            result[offset:offset + chunk_len] += chunk * fade
            weights[offset:offset + chunk_len] += fade

        mask = weights > 0
        result[mask] /= weights[mask]

    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

    return result


def main():
    parser = argparse.ArgumentParser(description="FlashSR audio enhancement bridge")
    parser.add_argument("--input", "-i", required=True, help="Input WAV file path")
    parser.add_argument("--output", "-o", required=True, help="Output WAV file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, or directml (default: auto)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to FlashSR model weights directory")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        log_result(False, f"Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    try:
        log_progress(5, "Loading FlashSR model...")
        import torch
        import numpy as np
        import soundfile as sf

        device = _detect_device(args.device)
        log_progress(10, f"Using device: {_device_label(device)}")

        # Locate model weights
        model_dir = args.model_dir
        if model_dir is None:
            for candidate in [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelWeights"),
                os.path.join(os.getcwd(), "ModelWeights"),
            ]:
                if os.path.isdir(candidate):
                    model_dir = candidate
                    break

        if model_dir is None or not os.path.isdir(model_dir):
            log_result(False, "ModelWeights directory not found. Download FlashSR weights first.")
            sys.exit(1)

        from FlashSR.FlashSR import FlashSR

        flashsr_model = FlashSR(
            student_ldm_ckpt_path=os.path.join(model_dir, "student_ldm.pth"),
            sr_vocoder_ckpt_path=os.path.join(model_dir, "sr_vocoder.pth"),
            autoencoder_ckpt_path=os.path.join(model_dir, "vae.pth"),
        )
        flashsr_model = flashsr_model.to(device)
        flashsr_model.eval()
        log_progress(25, "Model loaded")

        # Read input to determine channel count
        data, sr = sf.read(args.input, dtype="float32")
        is_stereo = data.ndim == 2 and data.shape[1] == 2

        if is_stereo:
            log_progress(30, "Stereo input — processing each channel separately")

            log_progress(35, "Processing left channel...")
            left_enhanced = _enhance_mono(flashsr_model, data[:, 0], sr, device, args.seed)

            log_progress(60, "Processing right channel...")
            right_enhanced = _enhance_mono(flashsr_model, data[:, 1], sr, device, args.seed + 1)

            min_len = min(len(left_enhanced), len(right_enhanced))
            stereo_out = np.column_stack([
                left_enhanced[:min_len],
                right_enhanced[:min_len]
            ])
            stereo_out = np.clip(stereo_out, -1.0, 1.0)
            sf.write(args.output, stereo_out, FLASHSR_SAMPLE_RATE, subtype="PCM_16")
        else:
            log_progress(35, f"Processing: {os.path.basename(args.input)}")
            enhanced = _enhance_mono(flashsr_model, data, sr, device, args.seed)
            enhanced = np.clip(enhanced, -1.0, 1.0)
            sf.write(args.output, enhanced, FLASHSR_SAMPLE_RATE, subtype="PCM_16")

        log_progress(90, "Output saved")
        log_result(True, args.output)

    except Exception as e:
        log_result(False, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
