import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .core import FRAME_PERIOD, SAMPLE_RATE, PathLike, PitchResult, _check_number


@dataclass
class NoteSegment:
    """Represents a musical note segment with timing and pitch information.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        pitch_median: Median pitch frequency in Hz
        pitch_midi: MIDI note number (0-127): the note nearest pitch_median,
            or with lam=None the most frequent semitone of the segment
    """

    start: float
    end: float
    pitch_median: float
    pitch_midi: int


def _frame_period(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return FRAME_PERIOD
    diffs = np.diff(timestamps)
    fp = float(np.median(diffs))
    # Absolute and relative tolerance together accept float rounding at any frame rate.
    if fp <= 0 or not np.all(np.abs(diffs - fp) <= 1e-9 + 1e-7 * fp):
        raise ValueError("timestamps must be strictly increasing and uniformly spaced")
    return fp


def _median_runs(midi: np.ndarray, width: int) -> np.ndarray:
    # Each voiced run is filtered on its own so the filter never smears pitch across an
    # unvoiced gap. The window is edge-padded and odd, at most the run length rounded up to odd.
    out = midi.copy()
    voiced = np.flatnonzero(np.isfinite(midi))
    if not len(voiced):
        return out
    for run in np.split(voiced, np.flatnonzero(np.diff(voiced) > 1) + 1):
        if len(run) < 3:
            continue
        k = min(width, len(run) | 1)
        if k <= 1:
            continue
        half = k // 2
        padded = np.pad(midi[run], half, mode="edge")
        out[run] = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    return out


def _rise_gates(audio: np.ndarray, timestamps: np.ndarray, fp: float) -> np.ndarray:
    # RMS over a 64 ms window centered on each frame. A gate fires on the first frame whose RMS
    # is at least 1/0.6 (4.4 dB) above the RMS 32 ms earlier: a re-articulated note. The 1e-12
    # floor keeps digital silence from gating on rounding noise.
    n = len(timestamps)
    w = max(2, round(0.064 * SAMPLE_RATE))
    if w % 2:
        w += 1
    half = w // 2
    power = np.zeros(len(audio) + w)
    np.square(audio, out=power[half : half + len(audio)])
    sums = np.empty(len(power) + 1)
    sums[0] = 0.0
    np.cumsum(power, out=sums[1:])
    starts = np.clip(np.rint(timestamps * SAMPLE_RATE).astype(int), 0, len(power) - w)
    rms = np.sqrt(np.maximum(0.0, (sums[starts + w] - sums[starts]) / w))
    lag = max(1, round(0.032 / fp))
    rising = np.zeros(n, dtype=bool)
    rising[lag:] = (rms[:-lag] <= 0.6 * rms[lag:]) & (rms[lag:] > 1e-12)
    gates = rising.copy()
    gates[1:] &= ~rising[:-1]
    return gates


def _changepoints(x: np.ndarray, penalty: Optional[float]) -> List[Tuple[int, int]]:
    # Optimal partitioning by dynamic programming: the cost of a segmentation is the sum of
    # squared error of each piece around its own mean plus `penalty` per piece. PELT pruning
    # drops a start once its cost up to `end` exceeds the best cost at `end`, which is exact for
    # this cost because splitting a piece never raises its error, and keeps the loop near linear
    # on real contours. Subtracting x[0] keeps the prefix sums small.
    m = len(x)
    if m == 0:
        return []
    if penalty is None:
        return [(0, m)]
    x = x - x[0]
    s1 = np.empty(m + 1)
    s1[0] = 0.0
    np.cumsum(x, out=s1[1:])
    s2 = np.empty(m + 1)
    s2[0] = 0.0
    np.cumsum(x * x, out=s2[1:])
    cost = np.full(m + 1, np.inf)
    cost[0] = 0.0
    back = np.zeros(m + 1, dtype=np.intp)
    active = np.zeros(m + 1, dtype=np.intp)
    n_active = 1
    for end in range(1, m + 1):
        starts = active[:n_active]
        sse = np.maximum(0.0, s2[end] - s2[starts] - (s1[end] - s1[starts]) ** 2 / (end - starts))
        candidates = cost[starts] + sse + penalty
        k = int(np.argmin(candidates))
        cost[end], back[end] = candidates[k], starts[k]
        keep = cost[starts] + sse <= cost[end]
        n_keep = int(np.count_nonzero(keep))
        active[:n_keep] = starts[keep]
        active[n_keep] = end
        n_active = n_keep + 1
    out, end = [], m
    while end:
        start = int(back[end])
        out.append((start, end))
        end = start
    return out[::-1]


def segment_notes(
    result: PitchResult,
    *,
    lam: Optional[float] = 250.0,
    min_note_duration: float = 0.05,
    detect_repeated_notes: bool = True,
) -> List[NoteSegment]:
    """
    Segments a pitch contour into discrete musical notes.

    Voicing is decided from the confidence with hysteresis (a note starts at
    0.5 and continues while confidence stays at or above 0.3). The voiced
    pitch is median-filtered over 64 ms, cut where the audio shows a sudden
    rise in loudness (a re-articulated note), and each remaining stretch is
    split into constant-pitch segments by an exact changepoint fit whose
    penalty is `lam`. Fragments on the same pitch separated by gaps of at
    most 80 ms are merged, and notes with less than `min_note_duration` of
    voiced evidence are dropped.

    Args:
        result: PitchResult with timestamps, pitch_hz, confidence and the
            audio the contour was computed from; the audio starts at the
            first frame, as produced by detect and by the stream
        lam: Penalty for a pitch change. Lower values split on smaller or
            shorter pitch changes, higher values keep longer notes: 100 for
            heavily ornamented material, 150 to 375 otherwise. None (or
            infinity) turns pitch splitting off, so each voiced stretch
            between loudness rises becomes one note at its most frequent
            semitone.
        min_note_duration: Minimum voiced duration in seconds of a note
        detect_repeated_notes: Cut notes at sudden loudness rises so that a
            repeated note on the same pitch is reported twice. Set to False
            for long held tones, where a note that gets louder would be cut in two.

    Returns:
        List of NoteSegment objects ordered in time. `pitch_median` is the
        median of the note's smoothed pitch in Hz; `pitch_midi` is the nearest
        MIDI note (for lam=None the most frequent semitone).

    Raises:
        TypeError: For a non-boolean detect_repeated_notes or a non-numeric lam or min_note_duration
        ValueError: For non-uniform timestamps or invalid settings

    Example:
        >>> result = swiftf0.detect_file("audio.wav")
        >>> notes = segment_notes(result)
        >>> for note in notes[:3]:
        ...     print(f"Note: {note.pitch_midi} ({note.pitch_median:.1f} Hz) "
        ...           f"from {note.start:.2f}s to {note.end:.2f}s")
    """
    if not isinstance(detect_repeated_notes, (bool, np.bool_)):
        raise TypeError("detect_repeated_notes must be a boolean")

    timestamps, pitch, confidence = result.timestamps, result.pitch_hz, result.confidence
    if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
        raise ValueError("timestamps must be finite and nonnegative")
    finite_conf = confidence[np.isfinite(confidence)]
    if np.any(finite_conf < 0) or np.any(finite_conf > 1):
        raise ValueError("confidence must lie in [0, 1]")

    if lam is not None:
        lam = _check_number("lam", lam, allow_inf=True, nonnegative=True)
        if np.isinf(lam):
            lam = None
    min_duration = _check_number("min_note_duration", min_note_duration, nonnegative=True)
    n = len(timestamps)
    if n == 0:
        return []

    # All windows are given in seconds and converted to frames: a 64 ms median window, no cut
    # within 32 ms of a run start or 16 ms of its end, gaps of up to 80 ms bridged. The penalty
    # is `lam * FRAME_PERIOD` at the model's frame period and grows with the frames per second
    # like the fit error does, so `lam` means the same for any frame period.
    fp = _frame_period(timestamps)
    median_width = 2 * round(0.032 / fp) + 1
    start_guard = max(1, round(0.032 / fp))
    end_guard = max(1, round(0.016 / fp))
    max_gap = int(0.080 / fp + 1e-9)
    min_frames = max(1, math.ceil(min_duration / fp - 1e-9))
    penalty = None if lam is None else lam * FRAME_PERIOD**2 / fp

    # Hysteresis keeps a note from flickering off when the confidence dips briefly below 0.5.
    valid = np.isfinite(pitch) & (pitch > 0) & np.isfinite(confidence)
    voiced = np.zeros(n, dtype=bool)
    on = False
    for i in range(n):
        on = bool(valid[i] and confidence[i] >= (0.3 if on else 0.5))
        voiced[i] = on
    midi = np.full(n, np.nan)
    midi[voiced] = 69.0 + 12.0 * np.log2(pitch[voiced] / 440.0)
    midi = _median_runs(midi, median_width)

    if detect_repeated_notes:
        audio = np.asarray(result.audio, dtype=np.float64)
        if audio.size < round((timestamps[-1] - timestamps[0]) * SAMPLE_RATE) + 1:
            raise ValueError("the result's audio does not cover its frames; set detect_repeated_notes=False")
        gates = _rise_gates(audio, timestamps - timestamps[0], fp)
    else:
        gates = np.zeros(n, dtype=bool)

    # Cut each voiced run at its loudness-rise gates, then split every piece at pitch changes.
    intervals: List[Tuple[int, int]] = []
    voiced_idx = np.flatnonzero(voiced)
    for run in np.split(voiced_idx, np.flatnonzero(np.diff(voiced_idx) > 1) + 1):
        if not len(run):
            continue
        start, end = int(run[0]), int(run[-1]) + 1
        cuts = start + start_guard + np.flatnonzero(gates[start + start_guard : end - end_guard])
        bounds = [start, *cuts.tolist(), end]
        for left, right in zip(bounds[:-1], bounds[1:]):
            for a, b in _changepoints(midi[left:right], penalty):
                intervals.append((left + a, left + b))

    # Join fragments on the same semitone across short gaps unless a gate marks a new attack there.
    # Each entry carries the median of its piece so unmerged pieces are measured once.
    merged: List[Tuple[int, int, float]] = []
    for a, b in intervals:
        p2 = float(np.nanmedian(midi[a:b]))
        if merged:
            left, end, p1 = merged[-1]
            protected = bool(np.any(gates[max(left + 1, end - 1) : min(n, a + 2)]))
            if a - end <= max_gap and abs(p1 - p2) < 0.5 and not protected:
                merged[-1] = (left, b, float(np.nanmedian(midi[left:b])))
                continue
        merged.append((a, b, p2))

    notes: List[NoteSegment] = []
    for a, b, _ in merged:
        x = midi[a:b]
        x = x[np.isfinite(x)]
        if len(x) < min_frames:
            continue
        # With the changepoint fit each piece sits on one pitch, so its median names the note.
        # Without it (lam=None) a piece may span several pitches, so the most frequent one is used.
        midi_median = float(np.median(x))
        if penalty is None:
            values, counts = np.unique(np.round(x), return_counts=True)
            selected = float(values[np.argmax(counts)])
        else:
            selected = midi_median
        notes.append(
            NoteSegment(
                start=float(timestamps[a]),
                end=float(timestamps[b - 1] + fp),
                pitch_median=440.0 * 2.0 ** ((midi_median - 69.0) / 12.0),
                pitch_midi=round(selected),
            )
        )
    return notes


def export_to_midi(
    notes: List[NoteSegment],
    output_path: PathLike,
    *,
    tempo: int = 120,
    velocity: int = 80,
    track_name: str = "SwiftF0 Notes",
) -> None:
    """
    Export note segments to MIDI file.

    Args:
        notes: List of NoteSegment objects containing note information
        output_path: Path to save the MIDI file
        tempo: MIDI tempo in BPM, 4 to 300 (default 120)
        velocity: MIDI note velocity 0-127 (default 80)
        track_name: Name for the MIDI track (default "SwiftF0 Notes")

    Raises:
        ImportError: If mido is not installed
        ValueError: For empty notes list or invalid parameters
    """
    try:
        import mido
    except ImportError:
        raise ImportError('mido is required for MIDI export. Install with: pip install "swift-f0[midi]"') from None

    if not notes:
        raise ValueError("Cannot export empty notes list")
    for name, value in (("tempo", tempo), ("velocity", velocity)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    # The set_tempo meta message holds at most 0xFFFFFF microseconds per beat, about 3.58 BPM.
    if not 4 <= tempo <= 300:
        raise ValueError("Tempo must be between 4 and 300 BPM")
    if not 0 <= velocity <= 127:
        raise ValueError("Velocity must be between 0 and 127")

    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))

    def seconds_to_ticks(seconds: float) -> int:
        return round(seconds * (ticks_per_beat * tempo / 60))

    # Absolute events; a note_off sorts before a note_on at the same tick so repeated and
    # overlapping notes keep their lengths, and every note lasts at least one tick.
    events = []
    for note in notes:
        midi_note = max(0, min(127, note.pitch_midi))
        start_tick = seconds_to_ticks(note.start)
        end_tick = max(start_tick + 1, seconds_to_ticks(note.end))
        events.append((start_tick, 1, "note_on", midi_note, velocity))
        events.append((end_tick, 0, "note_off", midi_note, 0))
    events.sort()

    previous_tick = 0
    for tick, _, kind, midi_note, event_velocity in events:
        track.append(mido.Message(kind, channel=0, note=midi_note, velocity=event_velocity, time=tick - previous_tick))
        previous_tick = tick

    mid.save(os.fspath(output_path))
