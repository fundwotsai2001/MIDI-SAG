"""
Trim leading silence from a vocal audio file using Silero VAD.
Finds the start of the first speech segment and saves the trimmed audio.

Usage:
    python vad_trim.py <input_audio> <output_audio>
"""
import sys
import numpy as np
import soundfile as sf
import librosa
import torch


def vad_trim_leading_silence(audio_path: str, output_path: str,
                              vad_threshold: float = 0.5,
                              min_speech_ms: int = 250,
                              min_silence_ms: int = 100):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio_mono = audio.mean(axis=1).astype(np.float32)
    else:
        audio_mono = audio.astype(np.float32)

    # Resample to 16k for Silero VAD
    audio_16k = (librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
                 .astype(np.float32)) if sr != 16000 else audio_mono

    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    model.eval()
    get_speech_timestamps = utils[0]

    ts = get_speech_timestamps(
        torch.from_numpy(audio_16k).float(), model,
        sampling_rate=16000,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )

    if not ts:
        print("[VAD] No speech detected, copying original audio.")
        sf.write(output_path, audio_mono, sr)
        return

    speech_start_sec = ts[0]['start'] / 16000
    start_sample = int(speech_start_sec * sr)
    trimmed = audio_mono[start_sample:]
    sf.write(output_path, trimmed, sr)
    print(f"[VAD] Leading silence trimmed: {speech_start_sec:.3f}s removed.")
    print(f"[VAD] Saved to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_audio> <output_audio>")
        sys.exit(1)
    vad_trim_leading_silence(sys.argv[1], sys.argv[2])
