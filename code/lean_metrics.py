"""Low-memory correlograms and quality metrics on a lazy zarr SortingAnalyzer
(SpikeInterface 0.105), for sortings of billions of spikes.

1. Spike vector. A lazy analyzer's sorting holds a ZarrSpikeVector: every
   to_spike_vector() user (the per-unit spike train cache, the per-unit index
   cache, correlograms) decompresses the whole vector into a new array, 180 GB at
   7.5 billion spikes, on top of what it builds. And synchrony's numba kernel
   cannot read a ZarrSpikeVector, so it fails and its columns are NaN.
   memmap_sorting_spikes() points the analyzer's sorting at the sorting's
   spikes.npy, opened as a read-only memmap, after checking it holds the same
   spikes (same_spikes). The same values, read from the page cache instead of
   copied.

2. Amplitudes. Amplitude metrics read spike_amplitudes per unit, which on a lazy
   analyzer is zarr orthogonal indexing: a unit's spikes span the whole recording,
   so each unit decompresses nearly all of the array (30 GB at 7.5 billion
   spikes), once per unit and per metric. amplitudes_in_memory() loads the array
   once for the quality metrics and puts the zarr array back afterwards.

The SpikeInterface functions these rely on are fingerprinted; check_sources()
raises if they change. Checked against SpikeInterface 0.105.0 and 0.105.2, and
0.105.5 with PR #4948 (alejoe91/spikeinterface@chronic-qc, unchanged sources).
With PR #4948, quality metrics read amplitudes in spike batches themselves, so
the capsule skips amplitudes_in_memory() there.
(spike_amplitudes needs nothing here: in 0.105.2 SpikeInterface's SpikeRetriever
reads the lazy spike vector per chunk itself.)
"""

import hashlib
import inspect
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from spikeinterface.core import analyzer_extension_core, basesorting, core_tools

# sha256 of inspect.getsource(): the accepted versions of each function. Where
# 0.105.2 differs from 0.105.0, the difference was read and does not change what
# this module relies on.
EXPECTED_SOURCES = {
    "BaseSorting.to_spike_vector": (basesorting.BaseSorting.to_spike_vector, "f60f71f55f95b9cb9d1835029ddc91b04a90166157b46ec92dabcf37c5f2e5e7"),
    "BaseSorting._get_spike_vector_segment_slices": (
        basesorting.BaseSorting._get_spike_vector_segment_slices,
        (
            "f82c0ac53678d9c8390a856d5c6a02b659759bd279a2e725da17a94fd76ca95b",  # 0.105.0
            # 0.105.2: one segment -> [[0, n]] without a search; the same slices
            "7b08798954eb007ac35d9f1a133249b493967a585fb8d688821105b62ff1bee1",
        ),
    ),
    "BaseSorting.to_reordered_spike_vector": (
        basesorting.BaseSorting.to_reordered_spike_vector,
        "e9231e064ebc48adb7790e24f528873f329597fc36d3fe9e16d7dde111ca5ca7",
    ),
    "BaseSorting.get_spike_vector_to_indices": (
        basesorting.BaseSorting.get_spike_vector_to_indices,
        "057c36b5e689534b7ad62fb3294363d6e5a86bb41ac75f910b64d758c25ff021",
    ),
    "SpikeVectorSortingSegment.get_unit_spike_train": (
        basesorting.SpikeVectorSortingSegment.get_unit_spike_train,
        "1ca2fb64b748fa0b9d995591a04b70711de712957227f3703285d9b5688ed6ab",
    ),
    "BaseSpikeVectorExtension._get_data": (
        analyzer_extension_core.BaseSpikeVectorExtension._get_data,
        "241b353ee2faf9bc167e5c97ed68967cdd5c68e778e0a81d131d184e13984bb2",
    ),
    "core_tools.slice_rows": (core_tools.slice_rows, "bf7d800a94225ab6c6b67c533bcda72c0cf516d898f928fa9d7f8adc932781f5"),
}

COARSE_STEP = 65_536

# Rows compared between spikes.npy and the analyzer's sorting, per block
CHECK_BLOCK_ROWS = 1_000_000
CHECK_BLOCKS = 16


def check_sources() -> None:
    def accepted(expected):
        return {expected} if isinstance(expected, str) else set(expected)

    changed = [
        name
        for name, (func, expected) in EXPECTED_SOURCES.items()
        if hashlib.sha256(inspect.getsource(func).encode()).hexdigest() not in accepted(expected)
    ]
    if changed:
        raise RuntimeError(
            f"SpikeInterface changed {changed}, which lean_metrics.py relies on. "
            "Check it against the new code and update EXPECTED_SOURCES."
        )


def same_spikes(sorting_analyzer, spikes_npy: Path) -> bool:
    """Whether spikes.npy holds the analyzer's spikes, in the same order.

    Compares the number of spikes, the unit IDs and CHECK_BLOCKS blocks of rows
    (first, last and evenly spaced) without loading either spike vector.
    """
    spikes_npy = Path(spikes_npy)
    spikes = np.load(spikes_npy, mmap_mode="r")
    if spikes.dtype.names is None or not {"sample_index", "unit_index", "segment_index"} <= set(spikes.dtype.names):
        logging.warning(f"\t{spikes_npy} is not a spike vector")
        return False
    info_file = spikes_npy.parent / "numpysorting_info.json"
    if info_file.is_file():
        unit_ids = np.array(json.loads(info_file.read_text())["unit_ids"])
        if not np.array_equal(unit_ids, np.asarray(sorting_analyzer.unit_ids)):
            logging.warning("\tspikes.npy's unit IDs differ from the analyzer's")
            return False
    ref = sorting_analyzer.sorting.to_spike_vector()  # lazy (zarr) for a lazy analyzer
    n = spikes.shape[0]
    if ref.size != n:
        logging.warning(f"\tspikes.npy has {n:,} spikes, the analyzer {ref.size:,}")
        return False
    length = min(CHECK_BLOCK_ROWS, n)
    starts = np.unique(np.linspace(0, n - length, CHECK_BLOCKS).astype(np.int64))
    for s in starts:
        a = np.array(spikes[s : s + length])
        b = ref[s : s + length]
        for field in ("sample_index", "unit_index", "segment_index"):
            if not np.array_equal(a[field], np.asarray(b[field])):
                logging.warning(f"\tspikes.npy differs from the analyzer's spikes ({field}, rows from {s:,})")
                return False
    return True


def _segment_slices(spikes: np.ndarray, num_segments: int) -> np.ndarray:
    """BaseSorting._get_spike_vector_segment_slices for a memmapped spike vector:
    np.searchsorted(spikes["segment_index"], arange(num_segments + 1)), without
    reading the whole field (searchsorted copies a strided field into memory)."""
    n = spikes.shape[0]
    field = spikes["segment_index"]
    coarse = np.array(field[::COARSE_STEP])
    bounds = []
    for value in range(num_segments + 1):
        j = int(np.searchsorted(coarse, value, side="left"))
        if j == 0:
            bounds.append(0)
            continue
        a = (j - 1) * COARSE_STEP
        b = min(j * COARSE_STEP, n)
        bounds.append(a + int(np.searchsorted(np.array(field[a:b]), value, side="left")))
    slices = np.zeros((num_segments, 2), dtype="int64")
    for s in range(num_segments):
        slices[s] = bounds[s], bounds[s + 1]
    return slices


def memmap_sorting_spikes(sorting_analyzer, spikes_npy: Path | None) -> bool:
    """Point sorting_analyzer.sorting's spike vector at spikes_npy (read-only
    memmap), if it holds the analyzer's spikes. Returns whether it did.

    The sorting's _kwargs keep the zarr spike vector, so pickling the sorting stays
    small. The spike vector caches are reset, so they are built from the memmap.
    """
    if spikes_npy is None or not Path(spikes_npy).is_file():
        logging.warning("\tNo spikes.npy: the analyzer's spike vector stays lazy (zarr)")
        return False
    if not same_spikes(sorting_analyzer, spikes_npy):
        logging.warning("\tspikes.npy differs: the analyzer's spike vector stays lazy (zarr)")
        return False
    check_sources()
    sorting = sorting_analyzer.sorting
    spikes = np.load(spikes_npy, mmap_mode="r")
    sorting._cached_spike_vector = spikes
    for segment in sorting.segments:
        segment.spikes = spikes
        segment.spikes_in_seg = None
    sorting._cached_spike_vector_segment_slices = _segment_slices(spikes, sorting.get_num_segments())
    sorting._cached_spike_vector_to_indices = None
    sorting._cached_lexsorted_spike_vector = {}
    logging.info(f"\tAnalyzer spike vector: {Path(spikes_npy).name} (memmap)")
    return True


@contextmanager
def amplitudes_in_memory(sorting_analyzer):
    """spike_amplitudes' amplitudes as an in-memory array while active (if the
    extension exists and holds a zarr array); the zarr array is put back after."""
    extension = sorting_analyzer.get_extension("spike_amplitudes")
    if extension is None or isinstance(extension.data.get("amplitudes"), np.ndarray):
        yield
        return
    check_sources()
    original = extension.data["amplitudes"]
    extension.data["amplitudes"] = core_tools.materialize_array(original)
    logging.info(
        f"\tspike_amplitudes loaded for the quality metrics "
        f"({extension.data['amplitudes'].nbytes / 1e9:.1f} GB)"
    )
    try:
        yield
    finally:
        extension.data["amplitudes"] = original
