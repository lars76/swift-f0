# SwiftF0

[![PyPI version](https://img.shields.io/pypi/v/swift-f0.svg)](https://pypi.org/project/swift-f0/)
[![Python versions](https://img.shields.io/pypi/pyversions/swift-f0.svg)](https://pypi.org/project/swift-f0/)
[![License](https://img.shields.io/github/license/lars76/swift-f0.svg)](https://github.com/lars76/swift-f0/blob/main/LICENSE)
[![Demo](https://img.shields.io/badge/demo-online-blue.svg)](https://swift-f0.github.io/)
[![Pitch Benchmark](https://img.shields.io/badge/benchmark-pitch--benchmark-green.svg)](https://github.com/lars76/pitch-benchmark/)

**SwiftF0** is a fast and accurate pitch detector for monophonic audio (one voice or instrument, not chords). A small neural network with 14 386 parameters reads the pitch from a spectrogram of the audio. For every frame it also gives a confidence that the pitch is right.

In the [Pitch Detection Benchmark](https://github.com/lars76/pitch-benchmark/), SwiftF0 has the highest pitch F1 of the 19 trackers tested, statistically tied with RMVPE. It runs at 180 times real time on a single laptop CPU core. Streaming needs 176 ms of lookahead. It supports frequencies between **46.875 Hz and 2093.75 Hz** (roughly F♯1 to C7). The [article](https://swift-f0.github.io/how/) explains how it works and the [training repository](https://github.com/lars76/swift-f0-training) how it was trained.

## Live Demo

Try SwiftF0 in your browser at [swift-f0.github.io](https://swift-f0.github.io/). The demo runs entirely client-side with ONNX Runtime Web, so your audio stays private.

## Installation

```bash
pip install swift-f0
```

Requires Python 3.8 or newer. The only hard dependencies are numpy and onnxruntime.

Optional extras:

```bash
pip install "swift-f0[audio]"   # soundfile and soxr: file loading and resampling of arrays and streams not at 16 kHz
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
result.loudness_db    # level in dB of the 32 ms around each frame, used by segment_notes

voiced = result.confidence >= 0.5            # your own mask. The plot and export functions apply the threshold themselves
mean_f0 = result.pitch_hz[voiced].mean()

plot_pitch(result, show=False, output_path="pitch.jpg")
export_to_csv(result, "pitch_data.csv")

# Turn the contour into notes. pitch_hold_ms (default 80) is the penalty per note: raise it for fewer, longer notes
notes = segment_notes(result)
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
    show(segment_notes(recent))              # your display: notes near either end of the window can still change

def on_stop():
    results.append(stream.flush())
    final = segment_notes(concat(results))   # the batch result, up to float rounding in the pitch
```

## API Reference

### Pitch detection

#### `SwiftF0`

```python
SwiftF0(threads: Optional[int] = None, spin: bool = True)
```

Loads the bundled model. `threads` sets the number of CPU threads, by default one per physical core. More than about six do not make this model faster. `spin=False` lets idle threads sleep between calls instead of busy-waiting, which saves CPU when streaming. Build one detector and reuse it. `detect` can be called from several threads at once. A stream must be used from one thread.

The package exports the model constants `SAMPLE_RATE` (16000 Hz), `FRAME_PERIOD` (0.016 s, the 16 ms frame), `FMIN` (46.875 Hz) and `FMAX` (2093.75 Hz).

#### `SwiftF0.detect`

```python
SwiftF0.detect(audio, sample_rate, fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchResult
```

Detects the pitch of an array. Multichannel audio (channels last) is mixed to mono and resampled to 16 kHz if needed. Float audio should lie in -1 to 1, while integer audio is scaled by its bit depth. Long audio is processed in 30 s windows, so the memory use stays constant.

`fmin` and `fmax` limit the pitch search to a band. By default the model's full range is used. A frame whose best pitch lies outside the band gets confidence 0, so it counts as unvoiced. The band must be at least about 4 % wide (`fmax >= 1.04125 * fmin`).

The confidence says how sure the model is that the frame is voiced and its pitch is right. A frame counts as voiced at 0.5 or above.

The model does not normalize the level. Scale very quiet recordings (peak below about -35 dBFS) first, for example `audio = audio / np.abs(audio).max() * 0.5`.

SwiftF0 reports any pitched sound, not only voices. Background music under speech is detected wherever the voice pauses. A level gate on `loudness_db` can remove such frames.

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

Creates a stream for live audio. Each `push` returns the frames that are final, possibly none. A frame is final once 176 ms of audio after it have arrived. `flush` returns the rest and closes the stream. Keep the sample rate the same within a stream. Larger chunks run faster: 20 ms chunks at about 16 times real time, 100 ms chunks at about 60.

#### `PitchResult`

```python
@dataclass
class PitchResult:
    timestamps: np.ndarray    # frame times in seconds
    pitch_hz: np.ndarray      # F0 estimate in Hz for each frame
    confidence: np.ndarray    # voicing score for each frame, voiced when >= 0.5
    loudness_db: np.ndarray   # RMS level in dB relative to full scale of the 32 ms of audio centered on each frame
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

Writes a CSV with the columns timestamp (seconds), pitch_hz, confidence and voiced. A frame is voiced when its confidence is at least `threshold`.

### Notes

#### `segment_notes`

```python
segment_notes(result: PitchResult, *, pitch_hold_ms: float = 80.0) -> List[Note]
```

Groups the pitch contour into notes, each with one pitch. A new pitch one semitone away becomes a note of its own once it lasts longer than `pitch_hold_ms`. Higher values give fewer, longer notes. A repeated note on the same pitch splits at a dip in loudness. The [article](https://swift-f0.github.io/how/#how-notes) explains the method.

`result` must come from one `detect` call or one stream. One hour of audio takes about 3.5 s.

Returns the notes ordered in time.

#### `Note`

```python
@dataclass
class Note:
    start: float     # start time in seconds
    end: float       # end time in seconds
    pitch_hz: float  # the note's fitted pitch in Hz
```

`pitch_hz` is one of the measured pitches of the note, rounded to a cent. The MIDI export and the plots use the nearest MIDI number.

#### `export_to_midi`

```python
export_to_midi(
    notes: List[Note],
    output_path,
    *,
    tempo: int = 120,
    velocity: int = 80,
    track_name: str = "SwiftF0 Notes",
) -> None
```

Writes the notes to a MIDI file. `tempo` sets the beats per minute (4 to 300) and `velocity` the loudness of each note (0 to 127).

### Plots

`output_path` saves the figure and `show` displays it. The pitch axis is logarithmic and fits the voiced frames. One wrong frame far from the voice can stretch it. Setting `fmin` and `fmax` when detecting avoids that.

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
    notes: List[Note],
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
    notes: List[Note],
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

The paper describes the first version of SwiftF0 (0.1.x). Version 0.2.0 replaced its network with a smaller one, which the [article](https://swift-f0.github.io/how/) describes. If you use SwiftF0 in your research, please cite:

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
