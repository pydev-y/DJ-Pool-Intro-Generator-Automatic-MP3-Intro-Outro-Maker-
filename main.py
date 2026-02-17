"""
DJ Pool Style Intro Maker (UPDATED)

Goal:
✅ Take ONE BAR from the song's own groove (not synthetic)
✅ Loop it 8 times as: 1-2-3-4 (x8)
✅ Make it sound like DJ pool intros:
   - Bars 1–4: DRUMS-ONLY (clean count-in)
   - Bars 5–8: DRUMS + LIGHT INSTRUMENTAL (still no vocals)
   - Tiny crossfade BETWEEN every bar repeat so the loop seam is invisible
   - Smooth crossfade INTO the real song

Stems:
- Uses Demucs 4-stems: drums, bass, other, vocals (cached for speed)

Install:
  pip install demucs librosa soundfile audioread numpy==1.26.4
Requires:
  ffmpeg in PATH

Folders:
  mp3s/        input mp3
  output/      edited mp3
  cache_stems/ cached stems
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# =========================
# SETTINGS
# =========================
INPUT_DIR = Path("mp3s")
OUTPUT_DIR = Path("output6")
CACHE_DIR = Path("cache_stems")

OUTPUT_PREFIX = "edited_"
BITRATE = "320k"

SR = 44100
BEATS_PER_BAR = 4
INTRO_BARS = 8

# search where to grab the groove bar from
INTRO_SEARCH_START = 8.0
INTRO_SEARCH_END = 80.0

# small edge fades for slices
EDGE_FADE_MS = 35

# crossfade between repeated bars (THIS removes obvious looping)
BAR_REPEAT_XFADE_SEC = 0.02  # 20ms (try 0.03 if seams still obvious)

# smooth transition into the real song
INTRO_TO_SONG_XFADE_SEC = 0.12  # 0.08 tighter, 0.15 smoother

# Demucs CPU (transformer-safe segment)
DEMUCS_JOBS = 4
DEMUCS_SEGMENT = 7
DEMUCS_OVERLAP = 0.25
DEMUCS_SHIFTS = 0

# MIX: DJ pool style (keep instrumental subtle to avoid vocal bleed)
# Bars 1–4 will be drums-only regardless.
W_BASS = 0.80
W_OTHER = 0.45   # IMPORTANT: lower = cleaner "pool" intro

# Kick scoring band (strong "1")
KICK_LOW = 40.0
KICK_HIGH = 170.0
KICK_WINDOW_SEC = 0.16

# Vocal safety for choosing the bar
VOCAL_MAX_RMS = 0.012

# Penalize bars that look like fills/rolls/breaks
ONSETS_HOP = 256
ONSET_PENALTY = 0.10  # increase to avoid fills more aggressively

# Optional last-bar "lift" (DJ-ish)
LIFT_ENABLED = True
LIFT_STRENGTH = 0.55
LIFT_HIGHPASS_START_HZ = 120.0
LIFT_HIGHPASS_END_HZ = 950.0


# =========================
# HELPERS
# =========================
def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())


def safe_name(s):
    return s.replace(":", "_").replace("\\", "_").replace("/", "_")


def edge_fade(stereo: np.ndarray, ms: int) -> np.ndarray:
    n = int(SR * ms / 1000)
    if n <= 0 or stereo.shape[0] < n * 2:
        return stereo
    ramp_in = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    ramp_out = np.linspace(1, 0, n, dtype=np.float32)[:, None]
    stereo[:n] *= ramp_in
    stereo[-n:] *= ramp_out
    return stereo


def crossfade(a: np.ndarray, b: np.ndarray, xfade_sec: float) -> np.ndarray:
    """Crossfade end of a with start of b."""
    if xfade_sec <= 0:
        return np.vstack([a, b])
    n = int(SR * xfade_sec)
    if n <= 0 or a.shape[0] < n or b.shape[0] < n:
        return np.vstack([a, b])

    fade_out = np.linspace(1, 0, n, dtype=np.float32)[:, None]
    fade_in = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.vstack([a[:-n], mid, b[n:]])


def rms_mono(stereo: np.ndarray) -> float:
    m = np.mean(stereo, axis=1)
    if m.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(m * m)))


def stft_band_energy_mono(y: np.ndarray, f_lo: float, f_hi: float) -> float:
    if y.size < 2048:
        return 0.0
    S = librosa.stft(y, n_fft=2048, hop_length=512, window="hann")
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.0
    return float(np.mean(mag[band, :]))


def count_onsets(y: np.ndarray) -> int:
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=ONSETS_HOP)
    peaks = librosa.util.peak_pick(onset_env, pre_max=2, post_max=2, pre_avg=4, post_avg=4, delta=0.20, wait=2)
    return int(len(peaks))


def highpass_filter_rise(stereo: np.ndarray, start_hz: float, end_hz: float, strength: float) -> np.ndarray:
    """
    DJ-style 'lift': gradually high-pass across the bar (STFT mask).
    """
    n = stereo.shape[0]
    if n < 4096 or strength <= 0:
        return stereo

    out = stereo.copy()
    for ch in (0, 1):
        y = out[:, ch].astype(np.float32)
        S = librosa.stft(y, n_fft=2048, hop_length=512, window="hann")
        mag, phase = np.abs(S), np.angle(S)
        freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)

        frames = mag.shape[1]
        mask = np.ones_like(mag, dtype=np.float32)
        for t in range(frames):
            frac = t / max(1, frames - 1)
            cutoff = start_hz + (end_hz - start_hz) * frac
            mask[freqs < cutoff, t] = 0.0

        filtered = librosa.istft((mag * mask) * np.exp(1j * phase), hop_length=512, length=n).astype(np.float32)
        out[:, ch] = (1.0 - strength) * out[:, ch] + strength * filtered

    return out


# =========================
# DEMUCS (CACHED)
# =========================
def demucs_split(song: Path) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    folder = CACHE_DIR / safe_name(song.stem)
    stems = folder / "htdemucs" / safe_name(song.stem)

    if (stems / "drums.wav").exists() and (stems / "bass.wav").exists() and (stems / "other.wav").exists() and (stems / "vocals.wav").exists():
        return stems

    run([
        "demucs",
        "--device", "cpu",
        "--segment", str(DEMUCS_SEGMENT),
        "--overlap", str(DEMUCS_OVERLAP),
        "--shifts", str(DEMUCS_SHIFTS),
        "-j", str(DEMUCS_JOBS),
        "-o", str(folder),
        str(song)
    ])
    return stems


# =========================
# PICK BEST 1 BAR
# =========================
def pick_best_bar_index(beat_times: np.ndarray, drums_st: np.ndarray, vocals_st: np.ndarray) -> int | None:
    """
    Choose bar start i (downbeat aligned) for a 1-bar loop.
    Prefer:
      - strong kick on beat 1
      - steady bar (few onsets/fills)
      - low vocal energy
    """
    if len(beat_times) < (BEATS_PER_BAR + 2):
        return None

    drums_m = np.mean(drums_st, axis=1)
    best_i = None
    best_score = -1e9

    start_i = int(np.searchsorted(beat_times, INTRO_SEARCH_START))
    for i in range(start_i, len(beat_times) - (BEATS_PER_BAR + 1)):
        if beat_times[i] > INTRO_SEARCH_END:
            break
        if i % BEATS_PER_BAR != 0:
            continue

        s = int(beat_times[i] * SR)
        e = int(beat_times[i + BEATS_PER_BAR] * SR)
        if e <= s + 2048 or e > len(drums_m):
            continue

        # reject if vocals too present in this bar (helps DJ-pool cleanliness)
        if rms_mono(vocals_st[s:e]) > VOCAL_MAX_RMS:
            continue

        # kick energy around downbeat
        k_end = int(min(len(drums_m), (beat_times[i] + KICK_WINDOW_SEC) * SR))
        kick = stft_band_energy_mono(drums_m[s:k_end], KICK_LOW, KICK_HIGH)

        # bar energy
        bar = drums_m[s:e]
        rms = float(np.sqrt(np.mean(bar * bar))) if bar.size else 0.0

        # steadiness penalty (avoid fills)
        onset_n = count_onsets(bar)
        penalty = onset_n * ONSET_PENALTY

        score = (kick * 3.2) + (rms * 1.1) - penalty
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


# =========================
# BUILD INTRO
# =========================
def build_intro_pool_style(one_bar_drums: np.ndarray, one_bar_inst: np.ndarray) -> np.ndarray:
    """
    Bars 1-4: drums-only
    Bars 5-8: drums + light instrumental
    Each bar boundary is crossfaded slightly to hide seams.
    Optional lift applied to the last bar.
    """
    bars = []
    for idx in range(INTRO_BARS):
        if idx < 4:
            bars.append(one_bar_drums.copy())
        else:
            bars.append(one_bar_inst.copy())

    # lift on last bar (optional)
    if LIFT_ENABLED and bars:
        bars[-1] = highpass_filter_rise(
            bars[-1],
            start_hz=LIFT_HIGHPASS_START_HZ,
            end_hz=LIFT_HIGHPASS_END_HZ,
            strength=LIFT_STRENGTH,
        )
        bars[-1] = edge_fade(bars[-1], EDGE_FADE_MS)

    # glue bars with tiny crossfade
    intro = bars[0]
    for b in bars[1:]:
        intro = crossfade(intro, b, BAR_REPEAT_XFADE_SEC)

    intro = edge_fade(intro, EDGE_FADE_MS)
    return intro


# =========================
# PROCESS
# =========================
def process_song(song: Path):
    out_mp3 = OUTPUT_DIR / f"{OUTPUT_PREFIX}{song.name}"
    if out_mp3.exists():
        print("⏭️ Skip:", song.name)
        return

    print("\n🎧 Processing:", song.name)

    stems = demucs_split(song)

    # Load stems (librosa returns (2,n) for mono=False)
    drums, _ = librosa.load(stems / "drums.wav", sr=SR, mono=False)
    bass, _ = librosa.load(stems / "bass.wav", sr=SR, mono=False)
    other, _ = librosa.load(stems / "other.wav", sr=SR, mono=False)
    vocals, _ = librosa.load(stems / "vocals.wav", sr=SR, mono=False)

    # Convert to (n,2)
    drums = np.vstack(drums).T.astype(np.float32)
    bass = np.vstack(bass).T.astype(np.float32)
    other = np.vstack(other).T.astype(np.float32)
    vocals = np.vstack(vocals).T.astype(np.float32)

    n = min(len(drums), len(bass), len(other), len(vocals))
    drums = drums[:n]
    bass = bass[:n]
    other = other[:n]
    vocals = vocals[:n]

    # Beat detection on drums
    drums_m = np.mean(drums, axis=1)
    _, beat_frames = librosa.beat.beat_track(y=drums_m, sr=SR)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR)

    if len(beat_times) < 16:
        print("❌ Beat detection failed:", song.name)
        return

    best_i = pick_best_bar_index(beat_times, drums, vocals)
    if best_i is None:
        best_i = 0

    # Cut exactly 1 bar for DRUMS
    s = int(beat_times[best_i] * SR)
    e = int(beat_times[best_i + BEATS_PER_BAR] * SR)
    one_bar_drums = drums[s:e].copy()
    one_bar_drums = edge_fade(one_bar_drums, EDGE_FADE_MS)

    # Build light instrumental for bars 5–8 (still no vocals)
    one_bar_inst = (drums[s:e] * 1.0) + (bass[s:e] * W_BASS) + (other[s:e] * W_OTHER)
    peak = float(np.max(np.abs(one_bar_inst))) if one_bar_inst.size else 0.0
    if peak > 0.98:
        one_bar_inst = one_bar_inst / peak * 0.98
    one_bar_inst = edge_fade(one_bar_inst.astype(np.float32), EDGE_FADE_MS)

    intro = build_intro_pool_style(one_bar_drums, one_bar_inst)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"tmp_{safe_name(song.stem)}_") as td:
        td = Path(td)
        intro_wav = td / "intro.wav"
        sf.write(intro_wav, intro, SR)

        # Smooth intro -> song
        run([
            "ffmpeg", "-y",
            "-i", str(intro_wav),
            "-i", str(song),
            "-filter_complex",
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
            f"[a0][a1]acrossfade=d={INTRO_TO_SONG_XFADE_SEC}:c1=tri:c2=tri[a]",
            "-map", "[a]",
            "-c:a", "libmp3lame",
            "-b:a", BITRATE,
            str(out_mp3)
        ])

    print("✅ Saved:", out_mp3.name)
    print(f"   Picked bar @ {beat_times[best_i]:.2f}s | bar-xfade={BAR_REPEAT_XFADE_SEC:.2f}s | drop-xfade={INTRO_TO_SONG_XFADE_SEC:.2f}s")


def main():
    songs = sorted(INPUT_DIR.glob("*.mp3"))
    if not songs:
        print("No mp3 files found in mp3s/")
        return

    for song in songs:
        process_song(song)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
