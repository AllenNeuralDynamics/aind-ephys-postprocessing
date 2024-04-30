import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from pathlib import Path
import shutil
import json
import argparse
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing"
VERSION = "1.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Postprocess ecephys data")

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = "Duration of clipped recording in debug mode. Default is 30 seconds. Only used if debug is enabled"
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="0.8", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="0.8", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")


if __name__ == "__main__":
    args = parser.parse_args()

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    if N_JOBS_CO is not None:
        if isinstance(N_JOBS, float):
            N_JOBS = int(N_JOBS * int(N_JOBS_CO))
        elif N_JOBS == -1:
            N_JOBS = int(N_JOBS_CO)
        elif int(N_JOBS_CO) < N_JOBS:
            N_JOBS = int(N_JOBS_CO)

    print(f"N_JOBS: {N_JOBS}")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_postprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    postprocessing_params = processing_params["postprocessing"]
    sparsity_params = processing_params["sparsity"]
    quality_metrics_names = processing_params["quality_metrics_names"]
    quality_metrics_params = processing_params["quality_metrics"]

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
            assert (
                preprocessed_folder / f"preprocessed_{recording_name}"
            ).is_dir(), f"Preprocessed folder for {recording_name} not found!"
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

        try:
            recording = si.load_extractor(preprocessed_folder / f"preprocessed_{recording_name}")
        except ValueError as e:
            print(f"Spike sorting skipped on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        # make sure we have spikesorted output for the block-stream
        sorted_folder = spikesorted_folder / f"spikesorted_{recording_name}"
        if not sorted_folder.is_dir():
            raise FileNotFoundError(f"Spike sorted data for {recording_name} not found!")

        try:
            sorting = si.load_extractor(sorted_folder)
        except ValueError as e:
            print(f"Spike sorting failed on {recording_name}. Skipping postprocessing")
            # create an empty result file (needed for pipeline)
            postprocessing_output_folder.mkdir()
            mock_array = np.array([], dtype=bool)
            np.save(postprocessing_output_folder / "placeholder.npy", mock_array)
            continue

        print(f"\t\tCreating sorting analyzer")
        sorting_analyzer_full = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=True,
            return_scaled=postprocessing_params["return_scaled"],
            **sparsity_params
        )
        # compute templates for de-duplication
        # now postprocess
        analyzer_dict = postprocessing_params.copy()
        analyzer_dict.pop("duplicate_threshold")
        analyzer_dict.pop("return_scaled")
        random_spikes_params = analyzer_dict.pop("random_spikes")
        sorting_analyzer_full.compute("random_spikes", **random_spikes_params)
        sorting_analyzer_full.compute("templates")
        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(
            sorting_analyzer_full, duplicate_threshold=postprocessing_params["duplicate_threshold"]
        )
        print(
            f"\tNumber of original units: {len(sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
        )
        n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
        postprocessing_notes += f"\n- Removed {n_duplicated} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids

        sorting_analyzer = sorting_analyzer_full.select_units(sorting_deduplicated.unit_ids)

        # save
        sorting_analyzer = sorting_analyzer.save_as(
            format="binary_folder",
            folder=postprocessing_output_folder
        )

        # now re-organize entries and postprocess
        analyzer_dict_sorted = {}
        analyzer_dict_sorted["waveforms"] = analyzer_dict.pop("waveforms")
        analyzer_dict_sorted["templates"] = {}
        analyzer_dict_sorted.update(analyzer_dict)

        print(f"\tComputing all postprocessing extensions")
        sorting_analyzer.compute(analyzer_dict_sorted)

        print("\tComputing quality metrics")
        qm = sorting_analyzer.compute(
            "quality_metrics",
            metric_names=quality_metrics_names,
            qm_params=quality_metrics_params
        )

        t_postprocessing_end = time.perf_counter()
        elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

        # save params in output
        postprocessing_params["recording_name"] = recording_name
        postprocessing_outputs = dict(duplicated_units=n_duplicated)
        postprocessing_process = DataProcess(
            name="Ephys postprocessing",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_postprocessing,
            end_date_time=datetime_start_postprocessing + timedelta(seconds=np.floor(elapsed_time_postprocessing)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=postprocessing_params,
            outputs=postprocessing_outputs,
            notes=postprocessing_notes,
        )
        with open(postprocessing_output_process_json, "w") as f:
            f.write(postprocessing_process.model_dump_json(indent=3))

    t_postprocessing_end_all = time.perf_counter()
    elapsed_time_postprocessing_all = np.round(t_postprocessing_end_all - t_postprocessing_start_all, 2)
    print(f"POSTPROCESSING time: {elapsed_time_postprocessing_all}s")
