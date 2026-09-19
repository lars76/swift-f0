# SwiftF0

[![PyPI version](https://img.shields.io/pypi/v/swift-f0.svg)](https://pypi.org/project/swift-f0/)
[![Python versions](https://img.shields.io/pypi/pyversions/swift-f0.svg)](https://pypi.org/project/swift-f0/)
[![License](https://img.shields.io/github/license/lars76/swift-f0.svg)](https://github.com/lars76/swift-f0/blob/main/LICENSE)
[![Demo](https://img.shields.io/badge/demo-online-blue.svg)](https://swift-f0.github.io/)
[![Pitch Benchmark](https://img.shields.io/badge/benchmark-pitch--benchmark-green.svg)](https://github.com/lars76/pitch-benchmark/)

**SwiftF0** is a fast and accurate pitch detector for monophonic audio (one voice or instrument, not chords). It turns the audio into a spectrogram and a small neural network (14 386 parameters, a 135 KB file) reads the pitch from it, together with a confidence that the pitch is right.

In the [Pitch Detection Benchmark](https://github.com/lars76/pitch-benchmark/), SwiftF0 has the highest pitch F1 of the 19 trackers tested. It runs at 450 times real time on a laptop CPU. Streaming needs 176 ms of lookahead. It supports frequencies between **46.875 Hz and 2093.75 Hz** (G1 to C7).

## Live Demo

Try SwiftF0 in your browser at [swift-f0.github.io](https://swift-f0.github.io/). The demo runs entirely client-side with ONNX Runtime Web, so your audio stays private.

## Installation

```bash
pip install swift-f0
```

Requires Python 3.8 or newer. The only hard dependencies are numpy and onnxruntime.

Optional extras:

```bash
pip install "swift-f0[audio]"   # soundfile and soxr: file loading, and resampling of arrays and streams not at 16 kHz
pip install "swift-f0[viz]"     # matplotlib: plotting
pip install "swift-f0[midi]"    # mido: MIDI export
pip install "swift-f0[full]"    # everything above
```

## Quick Start

```python
from swift_f0 import SwiftF0, segment_notes, plot_pitch, export_to_csv

f0 = SwiftF0()                               # this example needs pip install "swift-f0[audio,viz]"

# From a file (soundfile + soxr) ...
result = f0.detect_file("audio.wav")
# ... or from an array
# result = f0.detect(audio, sample_rate)
# Restrict the pitch range when you know it, e.g. speech:
# result = f0.detect_file("speech.wav", fmin=65, fmax=400)

result.timestamps     # seconds, one per 16 ms frame
result.pitch_hz       # a pitch for every frame
result.confidence     # voicing score, voiced when >= 0.5
result.audio          # the 16 kHz mono signal the contour came from

voiced = result.confidence >= 0.5            # your own mask. The plot and export functions apply the threshold themselves
mean_f0 = result.pitch_hz[voiced].mean()

plot_pitch(result, show=False, output_path="pitch.jpg")
export_to_csv(result, "pitch_data.csv")

# Turn the contour into notes. lam is the pitch-change penalty, None turns pitch splitting off
notes = segment_notes(result, lam=250)
```

Live audio: push chunks as they arrive, flush at the end.

```python
from swift_f0 import SwiftF0, concat, segment_notes

f0 = SwiftF0(spin=False)                     # no busy-waiting between chunks
stream = f0.stream()
results = []

def on_audio(chunk, sample_rate):            # microphone callback
    results.append(stream.push(chunk, sample_rate))
    recent = concat(results[-40:])           # the last 40 chunks (about 10 s with 250 ms chunks)
    show(segment_notes(recent))              # your display, provisional notes

def on_stop():
    results.append(stream.flush())
    final = segment_notes(concat(results))   # the same as the batch result
```

## API Reference

### Pitch detection

#### `SwiftF0`

```python
SwiftF0(threads: Optional[int] = None, spin: bool = True)
```

Loads the bundled model. `threads` sets the size of the ONNX Runtime thread pool, which defaults to the number of physical cores. More than about six threads do not help this model. `spin=False` lets the pool sleep between calls instead of busy-waiting, which keeps the CPU idle between streaming chunks at a cost of about 1 ms per call. Build one detector and reuse it. `detect` may be called from several threads at once, a stream must be driven from one thread.

The package exports the model constants `SAMPLE_RATE` (16000), `FRAME_PERIOD` (0.016 s), `FMIN` (46.875 Hz) and `FMAX` (2093.75 Hz).

#### `SwiftF0.detect`

```python
SwiftF0.detect(audio, sample_rate, fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchResult
```

Detects the pitch of an array at any integer sample rate. Multichannel input (channels last) is averaged to mono, and rates other than 16 kHz are resampled with soxr. Float arrays are expected in -1 to 1. Integer arrays are scaled to -1 to 1 by their bit depth, int16 by 32768.

`fmin` and `fmax` restrict the pitch search to that band inside the model. The default is the model's full range, to which wider requests are clipped. The confidence is not affected by the band. `fmin` must be below `fmax`, and the band must hold at least one model bin (`fmax >= 1.04125 * fmin`, about 4 %).

The confidence is a calibrated score of the frame being voiced with the pitch within 50 cents. Its threshold of 0.5 is tuned on the training data for the best F1, so the score is not a plain probability.

The model applies no level normalization and was trained on audio peaking between -35 and -5 dBFS. Scale very quiet recordings (peak below about -35 dBFS) before calling `detect`, for example `audio = audio / np.abs(audio).max() * 0.5`.

SwiftF0 reports any pitched sound. Background music under speech is detected with high confidence as soon as the voice pauses, so a speech application needs a level gate: for example, discard frames whose RMS over the 16 ms hop is more than 20 dB below the median RMS of the voiced frames.

#### `SwiftF0.detect_file`

```python
SwiftF0.detect_file(path, fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchResult
```

Reads the file with soundfile (WAV, FLAC, OGG, MP3, AIFF and others) and calls `detect`.

#### `SwiftF0.stream`

```python
SwiftF0.stream(fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchStream
PitchStream.push(audio, sample_rate) -> PitchResult
PitchStream.flush() -> PitchResult
```

Creates a stream for chunked input. Each `push` returns the frames that became final with that chunk, possibly none: a frame is final once 176 ms of audio after its center have arrived. `flush` returns the remaining frames and closes the stream. Both methods raise on a closed stream. The sample rate must not change within a stream. Timestamps continue across pushes. Each push runs one inference with about 1 ms of fixed cost, so the chunk size sets the speed: 20 ms chunks run at about 16 times real time, 100 ms chunks at about 60, and 1 s chunks approach the batch speed.

#### `PitchResult`

```python
@dataclass
class PitchResult:
    timestamps: np.ndarray    # frame times in seconds
    pitch_hz: np.ndarray      # F0 estimate in Hz for each frame
    confidence: np.ndarray    # voicing score for each frame, voiced when >= 0.5
    audio: np.ndarray         # mono signal at SAMPLE_RATE the frames were computed from
```

#### `concat`

```python
concat(results: Iterable[PitchResult]) -> PitchResult
```

Joins consecutive results of one stream into one result. Results of separate `detect` calls each start at time zero and cannot be joined. An empty sequence raises.

#### `export_to_csv`

```python
export_to_csv(result: PitchResult, output_path, *, threshold: float = 0.5) -> None
```

Writes a CSV with the columns timestamp, pitch_hz, confidence and voiced. A frame is voiced when its confidence is at least `threshold`.

### Notes

#### `segment_notes`

```python
segment_notes(
    result: PitchResult,
    *,
    lam: Optional[float] = 250.0,
    min_note_duration: float = 0.05,
    detect_repeated_notes: bool = True,
) -> List[NoteSegment]
```

Segments a pitch contour into notes with an exact changepoint fit. A note starts at confidence 0.5 and continues while the confidence stays at or above 0.3. `lam` is the pitch-change penalty and decides where a note is split on pitch: lower values split on smaller or shorter pitch changes, higher values keep longer notes (100 for heavily ornamented material, 150 to 375 otherwise). `lam=None` turns pitch splitting off. Each voiced stretch between two loudness rises then becomes a single note.

`detect_repeated_notes` (the default) uses the loudness in `result.audio` to separate repeated notes on the same pitch. Set it to `False` for long held tones, where a note that gets louder would be cut in two. Fragments on the same semitone separated by at most 80 ms are merged. Notes with less than `min_note_duration` seconds of voiced sound are dropped.

Returns the notes ordered in time. `pitch_median` is the median of the note's smoothed pitch in Hz. `pitch_midi` is the nearest MIDI note, or with `lam=None` the most frequent semitone, so the two can disagree with `lam=None`.

#### `NoteSegment`

```python
@dataclass
class NoteSegment:
    start: float         # start time in seconds
    end: float           # end time in seconds
    pitch_median: float  # median pitch in Hz
    pitch_midi: int      # MIDI note number (0-127)
```

#### `export_to_midi`

```python
export_to_midi(
    notes: List[NoteSegment],
    output_path,
    *,
    tempo: int = 120,
    velocity: int = 80,
    track_name: str = "SwiftF0 Notes",
) -> None
```

Writes the notes to a MIDI file. `tempo` sets the MIDI tempo in beats per minute, 4 to 300. Note times in seconds are preserved whatever tempo is chosen. `velocity` sets how loud each note sounds (0 to 127). Requires mido.

### Plots

All plot functions require matplotlib. `output_path` saves the figure at `dpi`, `show` displays it. `figsize` is passed to matplotlib. `style` is applied when matplotlib has it, otherwise the default style is used.

#### `plot_pitch`

```python
plot_pitch(
    result: PitchResult,
    *,
    threshold: float = 0.5,
    output_path=None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None
```

Plots the pitch contour, drawing frames with confidence at or above `threshold` as voiced.

#### `plot_notes`

```python
plot_notes(
    notes: List[NoteSegment],
    *,
    output_path=None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 6),
    style: str = "seaborn-v0_8",
) -> None
```

Plots the notes as a piano roll, each note a rectangle colored by pitch. Notes wider than 2 % of the plot are labeled with their MIDI number.

#### `plot_pitch_and_notes`

```python
plot_pitch_and_notes(
    result: PitchResult,
    segments: List[NoteSegment],
    *,
    threshold: float = 0.5,
    output_path=None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None
```

Plots the pitch contour (voiced at or above `threshold`) with the notes overlaid. Notes wider than 1 % of the plot are labeled with their MIDI number.

## Changelog

See [CHANGELOG.md](https://github.com/lars76/swift-f0/blob/main/CHANGELOG.md). Bugs and feature requests go to the [issue tracker](https://github.com/lars76/swift-f0/issues).

## Citation

The paper describes SwiftF0 0.1.x: a single STFT followed by a 2D convolutional network with 95 721 parameters. Version 0.2.0 replaces it with a learned harmonic comb: three STFTs are pooled onto a logarithmic frequency grid, where the harmonics of every candidate pitch sit at fixed offsets, and a 14 386-parameter network scores each candidate from the energy at those offsets, as subharmonic summation does with fixed weights. The pitch range and the frame rate are unchanged. If you use SwiftF0 in your research, please cite:

```bibtex
@misc{nieradzik2025swiftf0,
      title={SwiftF0: Fast and Accurate Monophonic Pitch Detection},
      author={Lars Nieradzik},
      year={2025},
      eprint={2508.18440},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.18440},
}
```

## License

MIT, see [LICENSE](https://github.com/lars76/swift-f0/blob/main/LICENSE).
