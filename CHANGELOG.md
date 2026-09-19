# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-19

This release changes the API. The Changed and Removed sections list every difference.

### Added
- A new model with 14 386 parameters in a 135 KB file. It scores higher on the [pitch benchmark](https://github.com/lars76/pitch-benchmark) than the 0.1.2 model and runs about twice as fast. The package exports its constants `SAMPLE_RATE`, `FRAME_PERIOD`, `FMIN` and `FMAX`.
- Streaming. `SwiftF0.stream()` returns a `PitchStream`. Push audio chunks into it as they arrive. Each push returns the frames that have become final, which trails the input by 176 ms. Call `flush()` at the end to get the last frames, and `concat()` to join all results into one.
- A new note segmentation. `segment_notes` finds the notes with an exact changepoint fit. The pitch-change penalty `lam` (default 250) decides where a note is split on pitch. Lower values split on finer pitch changes. `lam=None` turns pitch splitting off, the right setting for speech. Repeated notes on the same pitch are separated by the loudness in `result.audio`. Compared with the 0.1.2 algorithm, the new one scores higher on sung melodies and level on instrument stems, with one setting for both.
- A pitch range per call. `detect`, `detect_file` and `stream` accept `fmin` and `fmax`. The model restricts its search to that band without affecting the confidence.
- Thread pool control. `SwiftF0(threads=...)` sets the size of the ONNX Runtime thread pool. `SwiftF0(spin=False)` stops the pool from busy-waiting between calls, which matters for streaming.
- A rewritten [web demo](https://swift-f0.github.io/). It runs the same model with ONNX Runtime Web, shows the pitch as you record, and plays the detected notes on a piano or the original recording.

### Changed
- A frame is voiced when `result.confidence >= 0.5`. The threshold is no longer a detector setting. `plot_pitch`, `plot_pitch_and_notes` and `export_to_csv` take a `threshold` argument instead.
- `PitchResult` is constructed as `PitchResult(timestamps, pitch_hz, confidence, audio)`. The new `audio` field holds the 16 kHz mono signal the contour was computed from. Only `segment_notes` reads it, so a result built by hand may carry an empty array when `segment_notes` is called with `detect_repeated_notes=False`.
- The ONNX Runtime thread pool defaults to the number of physical cores. 0.1.2 was limited to one thread. Pass `threads=1` to get the old behavior.
- `plot_pitch`, `plot_notes`, `plot_pitch_and_notes`, `export_to_csv` and `export_to_midi` take all options as keyword arguments.
- `export_to_midi` writes note-off events with velocity 0.
- A missing optional package raises an `ImportError` that names the extra to install.
- Frames are centered on samples `k * 256`. Timestamps are no longer offset by 8 ms.
- Frames whose audio peaks below -60 dBFS get confidence 0.
- Files are read with `soundfile` and resampled with `soxr` instead of `librosa`. Install both with `pip install "swift-f0[audio]"`.
- Integer arrays are scaled to -1 to 1 by their bit depth, int16 by 32768, where 0.1.2 passed them through unscaled. Unsigned types have their midpoint removed. Non-integral sample rates are rejected.
- The package requires `onnxruntime>=1.18` and `numpy>=1.21.6`. Dependencies are declared in `pyproject.toml` only, so `requirements.txt` and `MANIFEST.in` are gone. The ONNX model takes the inputs `audio`, `fmin` and `fmax` and returns `pitch` (float64 Hz) and `confidence`. A `py.typed` marker lets type checkers use the package's annotations. `swift_f0.__version__` is read from the installed package metadata.

### Removed
- The `confidence_threshold`, `fmin` and `fmax` arguments of `SwiftF0()`.
- `PitchResult.voicing`.
- `detect_from_array` and `detect_from_file`. They are renamed to `detect` and `detect_file`.
- The `split_semitone_threshold` and `unvoiced_grace_period` arguments of `segment_notes`.
- The `music` extra. Install `swift-f0[viz,midi]` instead.
- The module attributes `__author__` and `__description__`.
- The 0.1.2 class constants of `SwiftF0`, such as `TARGET_SAMPLE_RATE`, `HOP_LENGTH`, `MODEL_FMIN` and `MODEL_FMAX`, and the instance attributes `confidence_threshold`, `fmin`, `fmax` and `pitch_session`.

### Fixed
- `export_to_midi` started a note that overlapped the previous one only when the previous one ended, which delayed every later note by the length of the overlap.
- `export_to_midi` truncated note times to whole ticks, and a note shorter than one tick was written with its end before its start. Times are now rounded and every note lasts at least one tick.
- `export_to_midi` accepted tempos below 4 BPM that a MIDI file cannot store.
- The plot functions changed the global matplotlib style for the rest of the process.

## [0.1.2] - 2025-07-25

### Added
- **Note Segmentation**: New `segment_notes()` function to convert pitch contours into discrete musical notes
- **MIDI Export**: `export_to_midi()` function to save note segments as standard MIDI files  
- **Note Visualization**: `plot_notes()` for piano roll visualization of segmented notes
- **Combined Analysis**: `plot_pitch_and_notes()` for unified pitch contour + note segment visualization
- **Advanced Note Parameters**: Configurable segmentation thresholds and duration constraints
- **NoteSegment Dataclass**: Structured representation of musical notes with timing and pitch information
- **Full Documentation**: Complete API reference for all new musical analysis features
- **Web Demo:** Interactive, browser-based demo using WebAssembly and ONNX.js.

### Changed
- Updated README with complete workflow examples including musical note analysis

## [0.1.1] - 2025-07-08

### Changed
- Renamed package from `swift_f0` to `swift-f0` for consistency with PyPI naming conventions

## [0.1.0] - 2025-07-08

### Added
- Initial release of SwiftF0 pitch detection library
- Core pitch detection functionality via ONNX model
- `SwiftF0` class with audio file and array processing capabilities
- `PitchResult` dataclass for structured pitch detection results
- Basic visualization with `plot_pitch()` function
- CSV export functionality with `export_to_csv()`
- PyPI packaging and distribution setup
- Comprehensive documentation and installation instructions
- Support for frequencies between 46.875 Hz and 2093.75 Hz (G1 to C7)
- Real-time analysis optimization (132 ms for 5 seconds of audio on CPU)

[Unreleased]: https://github.com/lars76/swift-f0/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lars76/swift-f0/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/lars76/swift-f0/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/lars76/swift-f0/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/lars76/swift-f0/releases/tag/v0.1.0
