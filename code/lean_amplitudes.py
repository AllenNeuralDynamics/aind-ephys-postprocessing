"""Low-memory spike_amplitudes for SpikeInterface 0.105.

ComputeSpikeAmplitudes feeds the spikes to its node pipeline through a
SpikeRetriever, whose __init__ (sorting_to_peaks) builds an in-memory "peaks"
array of every spike, 40 bytes per spike, from the full spike vector (24 bytes per
spike, loaded from the analyzer's zarr). With mp_context="spawn", that array is
also pickled to every worker. At 7.5 billion spikes that is 301 GB of peaks plus
180 GB of spike vector in the main process, before any worker starts.

MemmapSpikeRetriever is a SpikeRetriever that reads the spikes from the sorting's
spikes.npy as a memmap, only those of the chunk being computed, and builds the same
peak rows for them as sorting_to_peaks does. Each worker opens the memmap itself
(nothing but the file path is pickled). Everything else is SpikeInterface's own
code: the chunks, margins, peak shifts, uV scaling, and the gathering of the
amplitudes into the analyzer's zarr. So the amplitudes are the same as
SpikeInterface's.

It is used only while spike_amplitudes is computed (memmap_spike_amplitudes), and
only if spikes.npy holds the same spikes as the analyzer's sorting. The functions it
reproduces or relies on are fingerprinted; check_sources() raises if they change.
"""

import hashlib
import inspect
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import spikeinterface.postprocessing.spike_amplitudes as spike_amplitudes_module
from spikeinterface.core import node_pipeline
from spikeinterface.core.base import spike_peak_dtype
from spikeinterface.core.node_pipeline import PipelineNode, SpikeRetriever

# sha256 of inspect.getsource() in spikeinterface 0.105.0
EXPECTED_SOURCES = {
    "SpikeRetriever.__init__": (SpikeRetriever.__init__, "289b2f8b643f35e9355fb1b1f863c3863451dd50ee70326f175b4621a064b561"),
    "SpikeRetriever.get_peak_slice": (SpikeRetriever.get_peak_slice, "20e62795a5631b23809fd46a082fea116f22b0960448945ac8b92f6445ae2584"),
    "SpikeRetriever.compute": (SpikeRetriever.compute, "3cfe82cfe2f7e7543547dc75bdef2c237e123cc37f1e7778105e341e2b6feafa"),
    "node_pipeline.sorting_to_peaks": (node_pipeline.sorting_to_peaks, "8c55adeb137242dc9d6274537bd1d156aded9f90134deeeccfc78705bcea6424"),
    "node_pipeline._compute_peak_pipeline_chunk": (
        node_pipeline._compute_peak_pipeline_chunk,
        "f5f1faf3040c1f96610e40695ecee82964092c04a670a223ddf3ad9831e42dc5",
    ),
    "ComputeSpikeAmplitudes._get_pipeline_nodes": (
        spike_amplitudes_module.ComputeSpikeAmplitudes._get_pipeline_nodes,
        "6f99ac75b708d99ecfeee8dd70236057b4d34dd987ae1eb33d14791ca1059c6c",
    ),
    "SpikeAmplitudeNode.compute": (
        spike_amplitudes_module.SpikeAmplitudeNode.compute,
        "88a340d7008b53cf53183854de23303f600166ae73bf18cbb2d9340529b1f26f",
    ),
}

# Every COARSE_STEP-th spike's value is kept in memory to locate a chunk's spikes;
# then at most COARSE_STEP rows (1.5 MB) are read from the memmap per search
COARSE_STEP = 65_536

# Rows compared between spikes.npy and the analyzer's sorting, per block
CHECK_BLOCK_ROWS = 1_000_000
CHECK_BLOCKS = 16

_SPIKES_NPY: str | None = None  # set by memmap_spike_amplitudes


def check_sources() -> None:
    changed = [
        name
        for name, (func, expected) in EXPECTED_SOURCES.items()
        if hashlib.sha256(inspect.getsource(func).encode()).hexdigest() != expected
    ]
    if changed:
        raise RuntimeError(
            f"SpikeInterface changed {changed}, which the low-memory spike_amplitudes "
            "relies on. Check lean_amplitudes.py against the new code and update "
            "EXPECTED_SOURCES."
        )


class MemmapSpikeRetriever(SpikeRetriever):
    """SpikeRetriever(sorting, recording, channel_from_template=True), with the
    spikes read from spikes.npy (memmap) per chunk instead of held in memory."""

    def __init__(
        self,
        sorting,
        recording,
        channel_from_template=True,
        extremum_channel_inds=None,
        radius_um=50,
        peak_sign="neg",
        include_spikes_in_margin=False,
    ):
        if not channel_from_template or include_spikes_in_margin or extremum_channel_inds is not None:
            raise NotImplementedError(
                "MemmapSpikeRetriever supports only channel_from_template=True, "
                "include_spikes_in_margin=False (as spike_amplitudes uses it)"
            )
        if _SPIKES_NPY is None:
            raise RuntimeError("MemmapSpikeRetriever is used through memmap_spike_amplitudes")
        PipelineNode.__init__(self, recording, return_output=False)
        self.channel_from_template = True
        self._dtype = spike_peak_dtype
        self.include_spikes_in_margin = False
        main_channel_ids = sorting.get_property("main_channel_id")
        assert main_channel_ids is not None, "SpikeRetriever needs the sorting to have `main_channel_id`s."
        self.main_channel_indices = recording.ids_to_indices(main_channel_ids)

        self.spikes_npy = _SPIKES_NPY
        self.coarse_step = COARSE_STEP
        self._open()
        n = self._spikes.shape[0]

        # as SpikeRetriever: searchsorted(segment_index, [s, s + 1]) per segment
        segment_coarse = self._coarse("segment_index", 0, n)
        self.segment_slices = []
        self._sample_coarse = []
        for segment_index in range(recording.get_num_segments()):
            i0 = self._bisect_left("segment_index", segment_coarse, 0, n, segment_index)
            i1 = self._bisect_left("segment_index", segment_coarse, 0, n, segment_index + 1)
            self.segment_slices.append(slice(i0, i1))
            self._sample_coarse.append(self._coarse("sample_index", i0, i1))

    # the memmap is reopened in each worker; only the path is pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_spikes"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def _open(self):
        self._spikes = np.load(self.spikes_npy, mmap_mode="r")

    def _coarse(self, field, lo, hi):
        """spikes[field][lo:hi:coarse_step], read into memory."""
        return np.array(self._spikes[field][lo:hi:self.coarse_step])

    def _bisect_left(self, field, coarse, lo, hi, value):
        """np.searchsorted(spikes[field][lo:hi], value, side="left") + lo, for a
        sorted field, reading at most coarse_step rows."""
        j = int(np.searchsorted(coarse, value, side="left"))
        if j == 0:
            return lo  # spikes[field][lo] = coarse[0] >= value
        # coarse[j - 1] < value <= coarse[j]: the answer is in (a, b]
        a = lo + (j - 1) * self.coarse_step
        b = min(lo + j * self.coarse_step, hi)
        window = np.array(self._spikes[field][a:b])
        return a + int(np.searchsorted(window, value, side="left"))

    def get_peak_slice(self, segment_index, start_frame, end_frame, max_margin):
        sl = self.segment_slices[segment_index]
        coarse = self._sample_coarse[segment_index]
        i0 = self._bisect_left("sample_index", coarse, sl.start, sl.stop, start_frame)
        i1 = self._bisect_left("sample_index", coarse, sl.start, sl.stop, end_frame)
        return i0 - sl.start, i1 - sl.start

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin, peak_slice):
        sl = self.segment_slices[segment_index]
        i0, i1 = peak_slice
        rows = np.array(self._spikes[sl.start + i0 : sl.start + i1])
        # the rows sorting_to_peaks makes for these spikes
        local_peaks = np.zeros(rows.size, dtype=self._dtype)
        local_peaks["sample_index"] = rows["sample_index"]
        local_peaks["channel_index"] = self.main_channel_indices[rows["unit_index"]]
        local_peaks["amplitude"] = 0.0
        local_peaks["segment_index"] = rows["segment_index"]
        local_peaks["unit_index"] = rows["unit_index"]
        # then as SpikeRetriever.compute: sample index local to traces, amplitude
        # at the main channel (SpikeRetriever assigns it spike by spike)
        local_peaks["sample_index"] -= start_frame - max_margin
        local_peaks["amplitude"] = traces[local_peaks["sample_index"], local_peaks["channel_index"]]
        return (local_peaks,)


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


@contextmanager
def memmap_spike_amplitudes(sorting_analyzer, spikes_npy: Path | None):
    """While active, sorting_analyzer.compute("spike_amplitudes") reads the spikes
    from spikes_npy (memmap), if it holds the analyzer's spikes; otherwise
    SpikeInterface's own SpikeRetriever is used."""
    global _SPIKES_NPY
    if spikes_npy is None or not Path(spikes_npy).is_file() or not same_spikes(sorting_analyzer, spikes_npy):
        logging.warning(
            "\tspike_amplitudes: SpikeInterface's SpikeRetriever (all spikes in memory)"
        )
        yield
        return
    check_sources()
    logging.info(f"\tspike_amplitudes: spikes read from {Path(spikes_npy).name} (memmap)")
    original = spike_amplitudes_module.SpikeRetriever
    spike_amplitudes_module.SpikeRetriever = MemmapSpikeRetriever
    _SPIKES_NPY = str(Path(spikes_npy).resolve())
    try:
        yield
    finally:
        spike_amplitudes_module.SpikeRetriever = original
        _SPIKES_NPY = None
