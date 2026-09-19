import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import onnxruntime

SAMPLE_RATE = 16000
HOP = 256
FRAME_PERIOD = HOP / SAMPLE_RATE
FMIN = 46.875
FMAX = 2093.75
# The model scores 95 log-spaced pitch bins; a band narrower than one bin holds no candidate.
BIN_RATIO = (FMAX / FMIN) ** (1 / 94)
# The model's receptive field reaches 2815 samples each way: half of the 2048 window plus seven
# frames. A frame's own hop covers 256 of those samples on the future side, so a streamed frame
# is final once 10 later frames exist, and 11 earlier frames reproduce the batch result.
LOOKAHEAD_FRAMES = 10
LEFT_FRAMES = 11

# Digital silence sits on the spectrogram's log floor in every bin, a texture the model never saw,
# and it answers with random voiced frames. Frames whose audio peaks below this get confidence 0.
SILENCE_PEAK = 1e-3

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(eq=False)
class PitchResult:
    """Container for pitch detection results containing:
    - timestamps: Time positions (seconds) for each frame
    - pitch_hz: Estimated fundamental frequency in Hz for each frame
    - confidence: Voicing score (0-1) for each frame, calibrated so that a
      frame is voiced when its confidence is at least 0.5
    - audio: The mono 16 kHz signal the frames were computed from
    """

    timestamps: np.ndarray
    pitch_hz: np.ndarray
    confidence: np.ndarray
    audio: np.ndarray

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.pitch_hz = np.asarray(self.pitch_hz, dtype=np.float64)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if not (self.timestamps.ndim == self.pitch_hz.ndim == self.confidence.ndim == 1):
            raise ValueError("timestamps, pitch_hz and confidence must be 1-D arrays")
        if not (len(self.timestamps) == len(self.pitch_hz) == len(self.confidence)):
            raise ValueError("timestamps, pitch_hz and confidence must have the same length")
        if self.audio.ndim != 1:
            raise ValueError("audio must be a 1-D array")


def concat(results: Iterable[PitchResult]) -> PitchResult:
    results = list(results)
    if not results:
        raise ValueError("Cannot concatenate an empty sequence of results")
    if not all(isinstance(r, PitchResult) for r in results):
        raise TypeError("concat takes PitchResult objects")
    previous = None
    for r in results:
        if previous is not None and len(r.timestamps) and abs(r.timestamps[0] - previous.timestamps[-1] - FRAME_PERIOD) > 1e-9:
            raise ValueError("concat joins consecutive results of one stream; these timestamps do not continue")
        if len(r.timestamps):
            previous = r
    return PitchResult(
        timestamps=np.concatenate([r.timestamps for r in results]),
        pitch_hz=np.concatenate([r.pitch_hz for r in results]),
        confidence=np.concatenate([r.confidence for r in results]),
        audio=np.concatenate([r.audio for r in results]),
    )


def _check_number(name: str, value: object, *, allow_inf: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and not allow_inf) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be a {'' if allow_inf else 'finite '}number{' >= 0' if nonnegative else ''}")
    return number


def _mono(audio: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(audio)
    dtype = array.dtype
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError("audio has no channels")
        if array.shape[1] > max(array.shape[0], 32):
            raise ValueError(f"audio must be channels last, got shape {array.shape}")
    elif array.ndim != 1:
        raise ValueError("audio must be a 1-D (mono) or 2-D (channels last) array")
    if not np.isfinite(array).all():
        raise ValueError("audio contains non-finite values")
    integer = np.issubdtype(dtype, np.integer)
    if not integer and np.abs(array).max(initial=0) > np.finfo(np.float32).max:
        raise ValueError("audio values exceed the float32 range")
    array = array.astype(np.float64 if integer else np.float32, copy=False)
    if array.ndim == 2:
        # Adding the channel columns is far faster than a numpy reduction along a length-2 axis.
        channels = array.shape[1]
        array = array[:, 0] if channels == 1 else sum(array[:, c] for c in range(channels)) / channels
    if integer:
        # Integer audio is mapped to -1..1 as soundfile does: signed types are divided by
        # 2**(bits-1), unsigned types first have their midpoint removed.
        full_scale = 2.0 ** (np.iinfo(dtype).bits - 1)
        offset = 0.0 if np.issubdtype(dtype, np.signedinteger) else full_scale
        array = (array - offset) / full_scale
    # A copy, so the result never aliases the caller's array.
    return np.array(array, dtype=np.float32)


def _check_rate(sample_rate: float) -> int:
    rate = _check_number("sample_rate", sample_rate)
    if not rate.is_integer() or rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return int(rate)


def _soxr() -> Any:
    try:
        import soxr
    except ImportError:
        raise ImportError('soxr is required to resample audio. Install with: pip install "swift-f0[audio]"') from None
    return soxr


def _range(fmin: Optional[float], fmax: Optional[float]) -> Tuple[float, float]:
    fmin = FMIN if fmin is None else max(FMIN, _check_number("fmin", fmin, allow_inf=True))
    fmax = FMAX if fmax is None else min(FMAX, _check_number("fmax", fmax, allow_inf=True))
    if not fmin < fmax:
        raise ValueError(f"require fmin < fmax within the model range {FMIN} to {FMAX} Hz, got fmin={fmin}, fmax={fmax}")
    if fmax < fmin * BIN_RATIO:
        raise ValueError(f"fmax must be at least {BIN_RATIO:.5f} times fmin so the band holds a pitch candidate")
    return fmin, fmax


def _timestamps(first_frame: int, n_frames: int) -> np.ndarray:
    return (np.arange(n_frames) + first_frame) * FRAME_PERIOD


class SwiftF0:
    """SwiftF0 - A fast and accurate fundamental frequency (F0) detector using an ONNX model.

    The model takes mono audio at 16 kHz and returns a pitch and a confidence
    for every 256-sample frame (16 ms). A frame is voiced when its confidence
    is at least 0.5. Pitches lie between 46.875 Hz and 2093.75 Hz.

    Construction takes about 20 ms and allocates the thread pool, so build one
    detector and reuse it. `detect` may be called from several threads at
    once; a `PitchStream` must be driven from one thread.

    `threads` sets the size of the ONNX Runtime thread pool; the default is
    the number of physical cores, and more than about six threads do not help
    this model. `spin=False` lets the pool sleep between calls instead of
    busy-waiting, which costs about a millisecond per call and frees the CPU
    between calls: the right setting for streaming and for shared servers.
    """

    def __init__(self, threads: Optional[int] = None, spin: bool = True) -> None:
        if threads is not None and (isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1):
            raise ValueError("threads must be a positive integer")
        options = onnxruntime.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = threads
        if not spin:
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    def _run(self, audio: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
        pitch, confidence = self.session.run(
            ["pitch", "confidence"],
            {"audio": audio[None, :], "fmin": np.asarray(fmin, dtype=np.float32), "fmax": np.asarray(fmax, dtype=np.float32)},
        )
        pitch, confidence = np.asarray(pitch[0], dtype=np.float64), np.asarray(confidence[0], dtype=np.float64)
        n = len(confidence)
        hops = audio[: n * HOP].reshape(n, HOP) if len(audio) >= HOP else audio[None, :]
        confidence[np.abs(hops).max(axis=1) < SILENCE_PEAK] = 0.0
        return pitch, confidence

    def detect(self, audio: npt.ArrayLike, sample_rate: float, fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchResult:
        sample_rate = _check_rate(sample_rate)
        fmin, fmax = _range(fmin, fmax)
        signal = _mono(audio)
        if signal.size == 0:
            raise ValueError("audio must not be empty")
        if sample_rate != SAMPLE_RATE:
            signal = _soxr().resample(signal, sample_rate, SAMPLE_RATE)
            if signal.size == 0:
                raise ValueError("audio is too short to resample to 16 kHz")
        pitch, confidence = self._run(signal, fmin, fmax)
        return PitchResult(_timestamps(0, len(pitch)), pitch, confidence, signal)

    def detect_file(self, path: PathLike, fmin: Optional[float] = None, fmax: Optional[float] = None) -> PitchResult:
        try:
            import soundfile
        except ImportError:
            raise ImportError('soundfile is required to read audio files. Install with: pip install "swift-f0[audio]"') from None
        audio, sample_rate = soundfile.read(os.fspath(path), dtype="float32", always_2d=True)
        return self.detect(audio, sample_rate, fmin, fmax)

    def stream(self, fmin: Optional[float] = None, fmax: Optional[float] = None) -> "PitchStream":
        return PitchStream(self, fmin, fmax)


class PitchStream:
    def __init__(self, detector: SwiftF0, fmin: Optional[float] = None, fmax: Optional[float] = None) -> None:
        self._detector = detector
        self._fmin, self._fmax = _range(fmin, fmax)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._base = 0
        self._emitted = 0
        self._resampler: Optional[Any] = None
        self._sample_rate: Optional[int] = None
        self._closed = False

    def push(self, audio: npt.ArrayLike, sample_rate: float) -> PitchResult:
        if self._closed:
            raise RuntimeError("the stream has been flushed")
        sample_rate = _check_rate(sample_rate)
        if self._sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError("the sample rate must not change within a stream")
        signal = _mono(audio)
        if self._sample_rate is None:
            if sample_rate != SAMPLE_RATE:
                self._resampler = _soxr().ResampleStream(sample_rate, SAMPLE_RATE, 1, dtype="float32")
            self._sample_rate = sample_rate
        if self._resampler is not None:
            signal = self._resampler.resample_chunk(signal)
        self._buffer = np.concatenate([self._buffer, signal])
        return self._emit(final=False)

    def flush(self) -> PitchResult:
        if self._closed:
            raise RuntimeError("the stream has been flushed")
        if self._resampler is not None:
            tail = self._resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
            self._buffer = np.concatenate([self._buffer, tail])
        result = self._emit(final=True)
        self._closed = True
        return result

    def _emit(self, final: bool) -> PitchResult:
        # `base` is the frame index of buffer[0] in the whole stream and `emitted` the number
        # of frames returned so far; `first` and `last` index frames relative to the buffer.
        # A stream shorter than one hop still yields one frame at flush, as `detect` does.
        first = self._emitted - self._base
        available = len(self._buffer) // HOP
        if final and self._base == 0 and available == 0 and len(self._buffer):
            available = 1
        last = available if final else available - LOOKAHEAD_FRAMES
        if last > first:
            pitch, confidence = self._detector._run(self._buffer, self._fmin, self._fmax)
            pitch, confidence = pitch[first:last].copy(), confidence[first:last].copy()
        else:
            last = first
            pitch, confidence = np.zeros(0), np.zeros(0)
        audio = (self._buffer[first * HOP :] if final else self._buffer[first * HOP : last * HOP]).copy()
        result = PitchResult(_timestamps(self._emitted, len(pitch)), pitch, confidence, audio)
        self._emitted += len(pitch)
        if final:
            self._buffer = np.zeros(0, dtype=np.float32)
        else:
            keep = max(0, self._emitted - LEFT_FRAMES) - self._base
            self._buffer = self._buffer[keep * HOP :]
            self._base += keep
        return result


def export_to_csv(result: PitchResult, output_path: PathLike, *, threshold: float = 0.5) -> None:
    """
    Export pitch detection results to CSV file.

    Args:
        result: PitchResult object containing detection results
        output_path: Path to save the CSV file
        threshold: Confidence at or above which a frame is written as voiced (default 0.5)

    Raises:
        ValueError: For empty results
    """
    threshold = _check_number("threshold", threshold)
    if len(result.timestamps) == 0:
        raise ValueError("Cannot export empty results")

    with open(os.fspath(output_path), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "pitch_hz", "confidence", "voiced"])
        for t, p, c in zip(result.timestamps, result.pitch_hz, result.confidence):
            writer.writerow([f"{t:.4f}", f"{p:.2f}", f"{c:.4f}", "true" if c >= threshold else "false"])
