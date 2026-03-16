import os
import random
import pathlib
import numpy as np
import soundfile as sf
import librosa
import scipy.signal

SRC_DIR  = pathlib.Path(r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\01_Raw_Data\Audio_Recordings")
OUT_DIR  = pathlib.Path(r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\10_Outputs\Test_Samples_Long")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR     = 22050
TARGET_SECS   = 120
TARGET_SAMPS  = TARGET_SR * TARGET_SECS
NOISE_LEVEL   = 0.003
RUMBLE_LEVEL  = 0.004

random.seed(42)
np.random.seed(42)

def add_background_noise(y, sr):
    white = np.random.randn(len(y)) * NOISE_LEVEL
    rumble_raw = np.random.randn(len(y)) * RUMBLE_LEVEL
    b, a = scipy.signal.butter(3, 300 / (sr / 2), btype='low')
    rumble = scipy.signal.lfilter(b, a, rumble_raw)
    return y + white.astype(y.dtype) + rumble.astype(y.dtype)


def stretch_to_2min(y, sr):
    src_secs = len(y) / sr
    print("  Source duration: %.1fs  ->  target %ds" % (src_secs, TARGET_SECS))

    if src_secs >= TARGET_SECS * 0.9:
        rate = src_secs / TARGET_SECS
        rate = max(0.7, min(rate, 1.5))
        print("  Time-stretching rate=%.3f" % rate)
        y = librosa.effects.time_stretch(y, rate=rate)
    else:
        loops = int(np.ceil(TARGET_SECS / src_secs)) + 1
        segments = []
        for i in range(loops):
            jitter = random.uniform(0.82, 1.18)
            seg = librosa.effects.time_stretch(y, rate=jitter)
            segments.append(seg)
        y = np.concatenate(segments)

    if len(y) >= TARGET_SAMPS:
        y = y[:TARGET_SAMPS]
    else:
        pad = TARGET_SAMPS - len(y)
        y = np.pad(y, (0, pad), mode='constant')
    return y


def apply_fades(y, sr, fade_secs=2.0):
    fade = int(fade_secs * sr)
    fade = min(fade, len(y) // 4)
    y = y.copy()
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def normalize(y, target_dBFS=-18.0):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    target_amp = 10 ** (target_dBFS / 20.0)
    return y * (target_amp / peak)


all_files = sorted(
    [p for p in SRC_DIR.iterdir()
     if p.suffix.lower() in ('.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff')]
)[:10]

if not all_files:
    print("ERROR: No audio files found in", SRC_DIR)
    raise SystemExit(1)

print("Processing %d files..." % len(all_files))

results = []
for i, src_path in enumerate(all_files, 1):
    print("\n[%d/%d] %s" % (i, len(all_files), src_path.name))
    try:
        y, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print("  SKIPPED - could not load:", e)
        continue

    y = stretch_to_2min(y, TARGET_SR)
    y = add_background_noise(y, TARGET_SR)
    y = apply_fades(y, TARGET_SR)
    y = normalize(y)

    out_name = "LONG_%s.wav" % src_path.stem
    out_path = OUT_DIR / out_name
    sf.write(str(out_path), y, TARGET_SR, subtype='PCM_16')

    dur = len(y) / TARGET_SR
    print("  Saved: %s  (%.1fs, %d KB)" % (out_name, dur, out_path.stat().st_size // 1024))
    results.append(out_name)

print("\nDone! %d files written to:" % len(results))
print(" ", OUT_DIR)
