import warnings
warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess


URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-postprocessing"
VERSION = "0.1.0"


sparsity_params=dict(
    method="radius",
    radius_um=100
)

qm_params = {
    'presence_ratio': {'bin_duration_s': 60},
    'snr':  {
        'peak_sign': 'neg',
        'peak_mode': 'extremum',
        'random_chunk_kwargs_dict': None
    },
    'isi_violation': {
        'isi_threshold_ms': 1.5, 'min_isi_ms': 0
    },
    'rp_violation': {
        'refractory_period_ms': 1, 'censored_period_ms': 0.0
    },
    'sliding_rp_violation': {
        'bin_size_ms': 0.25,
        'window_size_s': 1,
        'exclude_ref_period_below_ms': 0.5,
        'max_ref_period_ms': 10,
        'contamination_values': None
    },
    'amplitude_cutoff': {
        'peak_sign': 'neg',
        'num_histogram_bins': 100,
        'histogram_smoothing_value': 3,
        'amplitudes_bins_min_ratio': 5
    },
    'amplitude_median': {
        'peak_sign': 'neg'
    },
    'nearest_neighbor': {
        'max_spikes': 10000, 'min_spikes': 10, 'n_neighbors': 4
    },
    'nn_isolation': {
        'max_spikes': 10000,
        'min_spikes': 10,
        'n_neighbors': 4,
        'n_components': 10,
        'radius_um': 100
    },
    'nn_noise_overlap': {
        'max_spikes': 10000,
        'min_spikes': 10,
        'n_neighbors': 4,
        'n_components': 10,
        'radius_um': 100
    }
}
qm_metric_names = ['num_spikes', 'firing_rate', 'presence_ratio', 'snr',
                   'isi_violation', 'rp_violation', 'sliding_rp_violation',
                   'amplitude_cutoff', 'drift', 'isolation_distance',
                   'l_ratio', 'd_prime']

postprocessing_params = dict(
    duplicate_threshold=0.9,
    sparsity=sparsity_params,
    waveforms_deduplicate=dict(ms_before=0.5,
                               ms_after=1.5,
                               max_spikes_per_unit=100,
                               return_scaled=False,
                               dtype=None,
                               precompute_template=('average', ),
                               use_relative_path=True,),
    waveforms=dict(ms_before=3.0,
                   ms_after=4.0,
                   max_spikes_per_unit=500,
                   return_scaled=True,
                   dtype=None,
                   precompute_template=('average', 'std'),
                   use_relative_path=True,),
    spike_amplitudes=dict(peak_sign='neg',
                          return_scaled=True,
                          outputs='concatenated',),
    similarity=dict(method="cosine_similarity"),
    correlograms=dict(window_ms=100.0,
                      bin_ms=2.0,),
    isis=dict(window_ms=100.0,
              bin_ms=5.0,),
    locations=dict(method="monopolar_triangulation"),
    template_metrics=dict(upsampling_factor=10, sparsity=None),
    principal_components=dict(n_components=5,
                              mode='by_channel_local',
                              whiten=True),
    quality_metrics=dict(qm_params=qm_params, metric_names=qm_metric_names, n_jobs=1),
)

job_kwargs = {
    'n_jobs': -1,
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
scratc_folder = Path("../scratch")
results_folder = Path("../results/")

tmp_folder = results_folder / "tmp"
tmp_folder.mkdir()


if __name__ == "__main__":
    data_process_prefix = "data_process_postprocessing"
    
    si.set_global_job_kwargs(**job_kwargs)

    ####### POSTPROCESSING ########
    print("\nPOSTPROCESSING")
    t_postprocessing_start_all = time.perf_counter()

    # check if test
    if (data_folder / "preprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
    else:
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder

    preprocessed_folders = [p for p in preprocessed_folder.iterdir() if p.is_dir() and "preprocessed_" in p.name]

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        recording_names = []
        for json_file in job_config_json_files:
            with open(json_file, "r") as f:
                config = json.load(f)
            recording_name = config["recording_name"]
            assert (preprocessed_folder / f"preprocessed_{recording_name}").is_dir(), f"Preprocessed folder for {recording_name} not found!"
            recording_names.append(recording_name)
    else:
        recording_names = [("_").join(p.name.split("_")[1:]) for p in preprocessed_folders]

    for recording_name in recording_names:
        datetime_start_postprocessing = datetime.now()
        t_postprocessing_start = time.perf_counter()
        postprocessing_notes = ""

        print(f"\tProcessing {recording_name}")
        postprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        postprocessing_output_folder = results_folder / f"postprocessed_{recording_name}"
        postprocessing_sorting_output_folder = results_folder / f"postprocessed-sorting_{recording_name}"

        recording = si.load_extractor(preprocessed_folder / f"preprocessed_{recording_name}")
        # make sure we have spikesorted output for the block-stream
        sorted_folder = spikesorted_folder / f"spikesorted_{recording_name}"
        if not sorted_folder.is_dir():
            raise FileNotFoundError(f"Spike sorted data for {recording_name} not found!")

        sorting = si.load_extractor(sorted_folder)

        # first extract some raw waveforms in memory to deduplicate based on peak alignment
        wf_dedup_folder = tmp_folder / "postprocessed" / recording_name
        we_raw = si.extract_waveforms(recording, sorting, folder=wf_dedup_folder,
                                      **postprocessing_params["waveforms_deduplicate"])
        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(we_raw, duplicate_threshold=postprocessing_params["duplicate_threshold"])
        print(f"\tNumber of original units: {len(we_raw.sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}")
        n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
        postprocessing_notes += f"\n- Removed {n_duplicated} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids
        # use existing deduplicated waveforms to compute sparsity
        sparsity_raw = si.compute_sparsity(we_raw, **sparsity_params)
        sparsity_mask = sparsity_raw.mask[sorting.ids_to_indices(deduplicated_unit_ids), :]
        sparsity = si.ChannelSparsity(mask=sparsity_mask, unit_ids=deduplicated_unit_ids, channel_ids=recording.channel_ids)
        shutil.rmtree(wf_dedup_folder)
        del we_raw

        # this is a trick to make the postprocessed folder "self-contained
        sorting_deduplicated = sorting_deduplicated.save(folder=postprocessing_sorting_output_folder)

        # now extract waveforms on de-duplicated units
        print(f"\tSaving sparse de-duplicated waveform extractor folder")
        we = si.extract_waveforms(recording, sorting_deduplicated, 
                                  folder=postprocessing_output_folder, sparsity=sparsity, sparse=True,
                                  overwrite=True, **postprocessing_params["waveforms"])
        print("\tComputing spike amplitides")
        amps = spost.compute_spike_amplitudes(we, **postprocessing_params["spike_amplitudes"])
        print("\tComputing unit locations")
        unit_locs = spost.compute_unit_locations(we, **postprocessing_params["locations"])
        print("\tComputing spike locations")
        spike_locs = spost.compute_spike_locations(we, **postprocessing_params["locations"])
        print("\tComputing correlograms")
        corr = spost.compute_correlograms(we, **postprocessing_params["correlograms"])
        print("\tComputing ISI histograms")
        tm = spost.compute_isi_histograms(we, **postprocessing_params["isis"])
        print("\tComputing template similarity")
        sim = spost.compute_template_similarity(we, **postprocessing_params["similarity"])
        print("\tComputing template metrics")
        tm = spost.compute_template_metrics(we, **postprocessing_params["template_metrics"])
        print("\tComputing PCA")
        pc = spost.compute_principal_components(we, **postprocessing_params["principal_components"])
        print("\tComputing quality metrics")
        qm = sqm.compute_quality_metrics(we, **postprocessing_params["quality_metrics"])

        t_postprocessing_end = time.perf_counter()
        elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

        # save params in output
        postprocessing_params["recording_name"] = recording_name
        postprocessing_outputs = dict(
            duplicated_units=n_duplicated
        )
        postprocessing_process = DataProcess(
                name="Ephys postprocessing",
                version=VERSION, # either release or git commit
                start_date_time=datetime_start_postprocessing,
                end_date_time=datetime_start_postprocessing + timedelta(seconds=np.floor(elapsed_time_postprocessing)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=postprocessing_params,
                outputs=postprocessing_outputs,
                notes=postprocessing_notes
            )
        with open(postprocessing_output_process_json, "w") as f:
            f.write(postprocessing_process.json(indent=3))

    t_postprocessing_end_all = time.perf_counter()
    elapsed_time_postprocessing_all = np.round(t_postprocessing_end_all - t_postprocessing_start_all, 2)
    print(f"POSTPROCESSING time: {elapsed_time_postprocessing_all}s")

