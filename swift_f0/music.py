import math
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from .core import FRAME_PERIOD, PathLike, PitchResult, _check_number


@dataclass
class Note:
    """A note found by `segment_notes`.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        pitch_hz: The note's fitted pitch in Hz: the observed frame pitch, rounded to a
            cent, that minimizes the note's confidence-weighted pitch error, capped at
            2 semitones per frame
    """

    start: float
    end: float
    pitch_hz: float


def _nearest_midi(pitch_hz: float) -> int:
    return math.floor(69 + 12 * math.log2(pitch_hz / 440) + 0.5)


def segment_notes(result: PitchResult, *, pitch_hold_ms: float = 80.0) -> List[Note]:
    """
    Segments a pitch contour into notes.

    Splits the frames into notes and gaps with the smallest total cost

        J = sum over notes of [beta + sum over its frames of (offpitch + unvoiced + quiet)]

    found exactly by dynamic programming. Each note has one pitch mu. Gaps cost nothing.

    - offpitch = c * min(|m - mu|, 2), where m is the frame's pitch in semitones and c
      its confidence, 0 for a frame without a pitch
    - unvoiced = -ln(c / (1 - c)), with c clipped to [0.01, 0.99]
    - quiet = the dB by which the frame lies more than 3 dB below the loudest frames
      within 64 ms on both sides, otherwise 0

    beta = pitch_hold_ms / 16 ms is the cost of each note. mu is one of the measured
    pitches, rounded to a cent. At the default of 80 ms, a new pitch one semitone away
    becomes a note of its own once it lasts longer than 80 ms. A jump of two semitones
    or more needs half that time. A pitch that returns to the old note needs twice as
    long. The article at https://swift-f0.github.io/how/#how-notes explains the method.

    Five minutes of audio take about 0.2 s, one hour about 3.5 s.

    Args:
        result: PitchResult of one take, timestamps spaced FRAME_PERIOD apart
        pitch_hold_ms: Penalty per note. Higher gives fewer, longer notes. At 0 a
            note starts at every pitch change, and frames at or below confidence 0.5
            are dropped.

    Returns:
        List of Note objects ordered in time.

    Raises:
        TypeError: For a non-numeric pitch_hold_ms
        ValueError: For a negative or infinite pitch_hold_ms, timestamps not spaced
            FRAME_PERIOD apart, confidence outside [0, 1] or non-finite loudness

    Example:
        >>> notes = segment_notes(detector.detect_file("audio.wav"))
        >>> print(notes[0].start, notes[0].end, notes[0].pitch_hz)
    """
    hold_ms = _check_number("pitch_hold_ms", pitch_hold_ms, nonnegative=True)
    timestamps, pitch, confidence, level = result.timestamps, result.pitch_hz, result.confidence, result.loudness_db
    if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
        raise ValueError("timestamps must be finite and nonnegative")
    if len(timestamps) > 1 and not np.all(np.abs(np.diff(timestamps) - FRAME_PERIOD) <= 1e-9):
        raise ValueError("timestamps must be spaced FRAME_PERIOD apart")
    if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("confidence must lie in [0, 1]")
    if not np.isfinite(level).all():
        raise ValueError("loudness_db must be finite")

    n = len(timestamps)
    valid = np.isfinite(pitch) & (pitch > 0)
    if not valid.any():
        return []
    m = np.zeros(n)
    m[valid] = 69.0 + 12.0 * np.log2(pitch[valid] / 440.0)
    w = np.where(valid, confidence, 0.0)
    cc = np.clip(w, 0.01, 0.99)
    padded = np.pad(level, 4, mode="symmetric")
    left = np.maximum.reduce([padded[j : j + n] for j in range(5)])
    right = np.maximum.reduce([padded[4 + j : 4 + j + n] for j in range(5)])
    q = -np.log(cc / (1.0 - cc)) + np.maximum(0.0, np.minimum(left, right) - level - 10 * np.log10(2.0))
    mu = np.unique(np.floor(m[valid] * 100 + 0.5) / 100)
    beta = hold_ms / 1000 / FRAME_PERIOD

    vn = np.full(len(mu), np.inf)
    note_start = np.zeros(len(mu), dtype=np.intp)
    back = np.zeros(n + 1, dtype=np.intp)
    kind = np.full(n + 1, -1, dtype=np.intp)
    work = np.empty(len(mu))
    best = 0.0
    for t in range(n):
        start = best + beta
        new = start < vn
        note_start[new] = t
        np.minimum(vn, start, out=vn)
        np.subtract(mu, m[t], out=work)
        np.abs(work, out=work)
        np.minimum(work, 2.0, out=work)
        work *= w[t]
        work += q[t]
        vn += work
        j = int(np.argmin(vn))
        if vn[j] < best:
            best = float(vn[j])
            back[t + 1] = note_start[j]
            kind[t + 1] = j
        else:
            back[t + 1] = t

    notes: List[Note] = []
    b = n
    while b > 0:
        a = int(back[b])
        if kind[b] >= 0:
            pitch_hz = float(440.0 * 2.0 ** ((mu[kind[b]] - 69.0) / 12.0))
            notes.append(Note(start=float(timestamps[a]), end=float(timestamps[b - 1] + FRAME_PERIOD), pitch_hz=pitch_hz))
        b = a
    notes.reverse()
    return notes


def export_to_midi(
    notes: List[Note],
    output_path: PathLike,
    *,
    tempo: int = 120,
    velocity: int = 80,
    track_name: str = "SwiftF0 Notes",
) -> None:
    """
    Export notes to a MIDI file. Each note is written as the MIDI number nearest
    its pitch, halves rounded up.

    Args:
        notes: List of Note objects
        output_path: Path to save the MIDI file
        tempo: MIDI tempo in BPM, 4 to 300 (default 120)
        velocity: MIDI note velocity 0-127 (default 80)
        track_name: Name for the MIDI track (default "SwiftF0 Notes")

    Raises:
        ImportError: If mido is not installed
        ValueError: For an empty notes list, a note without a finite pitch_hz above 0
            or invalid parameters
    """
    try:
        import mido
    except ImportError:
        raise ImportError('mido is required for MIDI export. Install with: pip install "swift-f0[midi]"') from None

    notes = list(notes)
    if not notes:
        raise ValueError("Cannot export empty notes list")
    if not all(math.isfinite(note.pitch_hz) and note.pitch_hz > 0 for note in notes):
        raise ValueError("Every note needs a finite pitch_hz above 0")
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
        midi_note = max(0, min(127, _nearest_midi(note.pitch_hz)))
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
