import os
from typing import Any, List, Optional, Tuple

import numpy as np

from .core import FMAX, FRAME_PERIOD, PathLike, PitchResult, _check_number
from .music import NoteSegment


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plotting. Install with: pip install "swift-f0[viz]"') from None
    return plt


def _pitch_axes(plt: Any, result: PitchResult, threshold: float, figsize: Tuple[float, float]) -> Tuple[Any, Any, float, float]:
    # Draws the contour, gray for every frame with the voiced frames in blue on top, and
    # returns the figure, the axes and the y-limits, which the callers extend.
    if len(result.timestamps) == 0:
        raise ValueError("Cannot plot empty results")
    voiced = result.confidence >= _check_number("threshold", threshold)
    voiced_frequencies = result.pitch_hz[voiced]
    if len(voiced_frequencies) > 0:
        ylow, yhigh = max(1.0, voiced_frequencies.min() * 0.9), min(FMAX, voiced_frequencies.max() * 1.1)
    else:
        ylow, yhigh = 50.0, 500.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.timestamps, result.pitch_hz, color="lightgray", alpha=0.7, linewidth=1.0, label="Unvoiced", zorder=1)
    ax.plot(result.timestamps, np.where(voiced, result.pitch_hz, np.nan), color="blue", linewidth=1.8, label="Voiced", zorder=2)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlim(result.timestamps[0], max(result.timestamps[-1], result.timestamps[0] + FRAME_PERIOD))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.grid(True, alpha=0.3)
    return fig, ax, ylow, yhigh


def _finish(plt: Any, fig: Any, output_path: Optional[PathLike], show: bool, dpi: int) -> None:
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(os.fspath(output_path), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_pitch(
    result: PitchResult,
    *,
    threshold: float = 0.5,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None:
    """
    Plot pitch with voicing information, optionally saving and/or showing.

    Args:
        result: PitchResult object containing detection results
        threshold: Confidence at or above which a frame is drawn as voiced (default 0.5)
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively (default True)
        dpi: Image resolution for saving (default 300)
        figsize: Figure size in inches (width, height) (default (12, 4))
        style: Matplotlib style to use (default "seaborn-v0_8")

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: For empty results
    """
    plt = _pyplot()
    with plt.style.context(style if style in plt.style.available else "default"):
        fig, ax, _, _ = _pitch_axes(plt, result, threshold, figsize)
        ax.set_title("SwiftF0 Pitch Detection")
        ax.legend(loc="upper right")
        _finish(plt, fig, output_path, show, dpi)


def plot_notes(
    notes: List[NoteSegment],
    *,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 6),
    style: str = "seaborn-v0_8",
) -> None:
    """
    Plot note segments as a piano roll visualization, optionally saving and/or showing.

    Args:
        notes: List of NoteSegment objects containing note information
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively (default True)
        dpi: Image resolution for saving (default 300)
        figsize: Figure size in inches (width, height) (default (12, 6))
        style: Matplotlib style to use (default "seaborn-v0_8")

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: For empty notes list
    """
    plt = _pyplot()
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    if not notes:
        raise ValueError("Cannot plot empty notes list")

    time_min = min(note.start for note in notes)
    time_max = max(note.end for note in notes)
    # Two semitones of room above and below keep the outer notes clear of the axes.
    midi_min = min(note.pitch_midi for note in notes) - 2
    midi_max = max(note.pitch_midi for note in notes) + 2

    with plt.style.context(style if style in plt.style.available else "default"):
        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=midi_min, vmax=midi_max)
        colormap = plt.get_cmap("viridis")

        for note in notes:
            duration = note.end - note.start
            ax.add_patch(
                Rectangle(
                    (note.start, note.pitch_midi - 0.4),
                    duration,
                    0.8,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=colormap(norm(note.pitch_midi)),
                    alpha=0.8,
                )
            )
            # Label only notes wider than 2 % of the plotted time, so labels do not overlap.
            if duration > (time_max - time_min) * 0.02:
                ax.text(
                    note.start + duration / 2,
                    note.pitch_midi,
                    str(note.pitch_midi),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if note.pitch_midi < (midi_min + midi_max) / 2 else "black",
                )

        ax.set_xlim(time_min - 0.1, time_max + 0.1)
        ax.set_ylim(midi_min, midi_max)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Note Number")
        ax.set_title("Note Segments (Piano Roll View)")
        ax.grid(True, alpha=0.3)

        # Note names fit on the axis up to two octaves; beyond that the numeric ticks stay.
        if midi_max - midi_min <= 24:
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            y_ticks = list(range(midi_min, midi_max + 1))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{note_names[midi % 12]}{midi // 12 - 1}" for midi in y_ticks])

        _finish(plt, fig, output_path, show, dpi)


def plot_pitch_and_notes(
    result: PitchResult,
    segments: List[NoteSegment],
    *,
    threshold: float = 0.5,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None:
    """
    Plot pitch contour with overlaid note segments, optionally saving and/or showing.

    Displays the continuous pitch contour from PitchResult with shaded regions
    showing the segmented notes. Each segment is labeled with its MIDI note number.
    Unvoiced regions appear as gaps in the pitch line.

    Args:
        result: PitchResult object containing pitch detection results
        segments: List of NoteSegment objects from segment_notes()
        threshold: Confidence at or above which a frame is drawn as voiced (default 0.5)
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively (default True)
        dpi: Image resolution for saving (default 300)
        figsize: Figure size in inches (width, height) (default (12, 4))
        style: Matplotlib style to use (default "seaborn-v0_8")

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: For empty results

    Example:
        >>> result = detector.detect_file("audio.wav")
        >>> segments = segment_notes(result)
        >>> plot_pitch_and_notes(result, segments, output_path="analysis.png")
    """
    plt = _pyplot()
    with plt.style.context(style if style in plt.style.available else "default"):
        fig, ax, ylow, yhigh = _pitch_axes(plt, result, threshold, figsize)
        for i, segment in enumerate(segments):
            ax.axvspan(segment.start, segment.end, color="orange", alpha=0.3, zorder=0, label="Note Segments" if i == 0 else "")

        y_offset = (yhigh - ylow) * 0.05
        total_duration = result.timestamps[-1] - result.timestamps[0]
        for segment in segments:
            segment_duration = segment.end - segment.start
            # Label only segments wider than 1 % of the plotted time, so labels do not overlap.
            if segment_duration > total_duration * 0.01:
                ax.text(
                    (segment.start + segment.end) / 2,
                    segment.pitch_median + y_offset,
                    f"MIDI {segment.pitch_midi}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    zorder=3,
                )

        ax.set_title("SwiftF0 Pitch Detection with Note Segments")
        ax.legend(loc="upper right")
        _finish(plt, fig, output_path, show, dpi)
