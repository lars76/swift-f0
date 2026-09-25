import math
import os
from typing import Any, List, Optional, Tuple

import numpy as np

from .core import FMAX, FMIN, FRAME_PERIOD, PathLike, PitchResult, _check_number
from .music import Note, _nearest_midi

PAD_SEMITONES = 2.0
MIN_SPAN_SEMITONES = 8.0
FALLBACK_OCTAVES = 1.0


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plotting. Install with: pip install "swift-f0[viz]"') from None
    return plt


def _axis_range(voiced_frequencies: np.ndarray) -> Tuple[float, float]:
    centre = math.sqrt(FMIN * FMAX)
    lo, hi = math.log2(centre) - FALLBACK_OCTAVES, math.log2(centre) + FALLBACK_OCTAVES
    usable = np.asarray(voiced_frequencies, dtype=np.float64)
    usable = usable[np.isfinite(usable) & (usable > 0)]
    if len(usable):
        lo, hi = math.log2(float(usable.min())) - PAD_SEMITONES / 12, math.log2(float(usable.max())) + PAD_SEMITONES / 12
    if hi - lo < MIN_SPAN_SEMITONES / 12:
        centred = (lo + hi) / 2
        lo, hi = centred - MIN_SPAN_SEMITONES / 24, centred + MIN_SPAN_SEMITONES / 24
    return 2.0**lo, 2.0**hi


def _pitch_axes(plt: Any, result: PitchResult, threshold: float, figsize: Tuple[float, float]) -> Tuple[Any, Any, float, float]:
    # Draws the contour, gray for every frame inside the axis with every voiced frame in blue on top, and returns
    # the figure, the axes and the y-limits, which the callers use to place their own labels.
    if len(result.timestamps) == 0:
        raise ValueError("Cannot plot empty results")
    voiced = result.confidence >= _check_number("threshold", threshold)
    ylow, yhigh = _axis_range(result.pitch_hz[voiced])
    inside = (result.pitch_hz >= ylow) & (result.pitch_hz <= yhigh)
    shown = voiced & np.isfinite(result.pitch_hz) & (result.pitch_hz > 0)
    alone = shown & ~np.concatenate(([False], shown[:-1])) & ~np.concatenate((shown[1:], [False]))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        result.timestamps,
        np.where(inside, result.pitch_hz, np.nan),
        color="lightgray",
        alpha=0.7,
        linewidth=1.0,
        label="Unvoiced",
        zorder=1,
    )
    ax.plot(
        result.timestamps,
        np.where(shown, result.pitch_hz, np.nan),
        color="blue",
        linewidth=1.8,
        label="Voiced",
        zorder=2,
    )
    ax.plot(result.timestamps[alone], result.pitch_hz[alone], linestyle="none", marker="o", markersize=2.5, color="blue", zorder=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(ylow, yhigh)
    # The axis is log spaced, so an even step crowds its rungs into the top once the span grows.
    # The decade ladder is tried first and kept when it has enough rungs to read.
    decades = [10.0**e for e in range(math.floor(math.log10(ylow)), math.floor(math.log10(yhigh)) + 1)]
    ticks = [m * e for e in decades for m in (1, 2, 5) if ylow <= m * e <= yhigh]
    if len(ticks) < 4:
        unit = 10.0 ** math.floor(math.log10((yhigh - ylow) / 8))
        step = next(m * unit for m in (1, 2, 2.5, 5, 10) if (yhigh - ylow) / (m * unit) <= 8)
        ticks = [k * step for k in range(math.ceil(ylow / step), math.floor(yhigh / step) + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:.0f}" for tick in ticks])
    ax.minorticks_off()
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
        ax.legend(loc="best")
        _finish(plt, fig, output_path, show, dpi)


def plot_notes(
    notes: List[Note],
    *,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 6),
    style: str = "seaborn-v0_8",
) -> None:
    """
    Plot notes as a piano roll, each at the MIDI number nearest its pitch (halves
    rounded up), optionally saving and/or showing.

    Args:
        notes: List of Note objects
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively (default True)
        dpi: Image resolution for saving (default 300)
        figsize: Figure size in inches (width, height) (default (12, 6))
        style: Matplotlib style to use (default "seaborn-v0_8")

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: For an empty notes list or a note without a finite pitch_hz above 0
    """
    plt = _pyplot()
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    notes = list(notes)
    if not notes:
        raise ValueError("Cannot plot empty notes list")
    if not all(math.isfinite(note.pitch_hz) and note.pitch_hz > 0 for note in notes):
        raise ValueError("Every note needs a finite pitch_hz above 0")

    time_min = min(note.start for note in notes)
    time_max = max(note.end for note in notes)
    midis = [_nearest_midi(note.pitch_hz) for note in notes]
    # Two semitones of room above and below keep the outer notes clear of the axes.
    midi_min = min(midis) - 2
    midi_max = max(midis) + 2

    with plt.style.context(style if style in plt.style.available else "default"):
        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=midi_min, vmax=midi_max)
        colormap = plt.get_cmap("viridis")

        for note, midi in zip(notes, midis):
            duration = note.end - note.start
            ax.add_patch(
                Rectangle(
                    (note.start, midi - 0.4),
                    duration,
                    0.8,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=colormap(norm(midi)),
                    alpha=0.8,
                )
            )
            # Label only notes wider than 2 % of the plotted time, so labels do not overlap.
            if duration > (time_max - time_min) * 0.02:
                ax.text(
                    note.start + duration / 2,
                    midi,
                    str(midi),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if midi < (midi_min + midi_max) / 2 else "black",
                )

        ax.set_xlim(time_min - 0.1, time_max + 0.1)
        ax.set_ylim(midi_min, midi_max)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Note Number")
        ax.set_title("Notes (Piano Roll View)")
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
    notes: List[Note],
    *,
    threshold: float = 0.5,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 4),
    style: str = "seaborn-v0_8",
) -> None:
    """
    Plot pitch contour with overlaid notes, optionally saving and/or showing.

    Displays the continuous pitch contour from PitchResult with shaded regions
    showing the notes. Each note is labeled with the MIDI number nearest its pitch.
    Unvoiced frames appear as gaps in the blue voiced line.

    Args:
        result: PitchResult object containing pitch detection results
        notes: List of Note objects from segment_notes()
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
        >>> notes = segment_notes(result)
        >>> plot_pitch_and_notes(result, notes, output_path="analysis.png")
    """
    plt = _pyplot()
    notes = list(notes)
    with plt.style.context(style if style in plt.style.available else "default"):
        fig, ax, ylow, yhigh = _pitch_axes(plt, result, threshold, figsize)
        for i, note in enumerate(notes):
            ax.axvspan(note.start, note.end, color="orange", alpha=0.3, zorder=0, label="Notes" if i == 0 else "")

        y_offset = (yhigh / ylow) ** 0.05
        total_duration = result.timestamps[-1] - result.timestamps[0]
        for note in notes:
            note_duration = note.end - note.start
            # Label only notes wider than 1 % of the plotted time, so labels do not overlap, and
            # only where the axis reaches: a label outside it would grow the saved figure to fit.
            if note_duration > total_duration * 0.01 and ylow <= note.pitch_hz * y_offset <= yhigh:
                ax.text(
                    (note.start + note.end) / 2,
                    note.pitch_hz * y_offset,
                    f"MIDI {_nearest_midi(note.pitch_hz)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                    zorder=3,
                )

        ax.set_title("SwiftF0 Pitch Detection with Notes")
        ax.legend(loc="best")
        _finish(plt, fig, output_path, show, dpi)
