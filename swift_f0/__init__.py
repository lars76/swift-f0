"""
SwiftF0 - A fast and accurate fundamental frequency (F0) detector

SwiftF0 is a Python library for pitch detection using ONNX.
It provides a simple API for detecting pitch from audio files, numpy arrays or a live stream.
"""

from importlib.metadata import version as _version

from .core import FMAX, FMIN, FRAME_PERIOD, SAMPLE_RATE, PitchResult, PitchStream, SwiftF0, concat, export_to_csv
from .music import NoteSegment, export_to_midi, segment_notes
from .plot import plot_notes, plot_pitch, plot_pitch_and_notes

__all__ = [
    # Model constants
    "SAMPLE_RATE",
    "FRAME_PERIOD",
    "FMIN",
    "FMAX",
    # Core pitch detection
    "SwiftF0",
    "PitchStream",
    "PitchResult",
    "concat",
    "export_to_csv",
    # Musical analysis
    "NoteSegment",
    "segment_notes",
    "export_to_midi",
    # Plots
    "plot_pitch",
    "plot_notes",
    "plot_pitch_and_notes",
]
__version__ = _version("swift-f0")
