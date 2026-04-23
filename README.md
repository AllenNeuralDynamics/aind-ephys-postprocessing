# Postprocessing for AIND ephys pipeline
## aind-ephys-postprocessing


### Description

This capsule is designed to postprocess ephys data for the AIND pipeline.

This capsule first removes redundant/duplicate units (sharing >90% of the spikes). Then, it
computes *sparse* waveforms and then in computes several additional postprocessing data, including:

- spike amplitudes
- unit locations
- spike locations
- correlograms
- ISI histograms
- template similarity
- template metrics
- PCA scores

In addition, [quality metrics](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html) are also computed here. Postprocessed data are used downstream for spike sorting curation and visualization.


### Inputs

The `data/` folder must include the output of the [aind-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing) and the [aind-ephys-spikesort-pykilosort](https://github.com/AllenNeuralDynamics/aind-ephys-spiksort-pykilosort)/[aind-ephys-spikesort-kilosort25](https://github.com/AllenNeuralDynamics/aind-ephys-spiksort-kilosort25)capsules, including the `preprocessed_{recording_name}` and `spikesorted_{recording_name}` folders.

### Parameters

The `code/run` script takes the follwing arguments:

```bash
  --n-jobs N_JOBS       Number of jobs to use for parallel processing.
                        Default is -1 (all available cores). It can also be a float between 0 and 1 to use a fraction of available cores
  --params-file PARAMS_FILE
                        Optional json file with parameters
  --params-str PARAMS_STR
                        Optional json string with parameters

```

A full list of parameters used for postprocessing and quality metrics calculation can be found in the `code/params.json`:

```json
{
    "job_kwargs": {
        "chunk_duration": "1s",
        "progress_bar": false
    },
    "sparsity": {
        "method": "radius",
        "radius_um": 100
    },
    "postprocessing": {
        "duplicate_threshold": 0.9,
        "waveforms_deduplicate": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "max_spikes_per_unit": 100,
            "return_scaled": false,
            "dtype": null,
            "sparse": false,
            "precompute_template": ["average"],
            "use_relative_path": true
        },
        "waveforms": {
            "ms_before": 3.0,
            "ms_after": 4.0,
            "max_spikes_per_unit": 500,
            "return_scaled": true,
            "dtype": null,
            "precompute_template": ["average", "std"],
            "use_relative_path": true
        },
        "spike_amplitudes": {
            "peak_sign": "neg",
            "return_scaled": true,
            "outputs": "concatenated"
        },
        "similarity": {
            "method": "cosine_similarity"
        },
        "correlograms": {
            "window_ms": 50.0,
            "bin_ms": 1.0
        },
        "isis": {
            "window_ms": 100.0,
            "bin_ms": 5.0
        },
        "unit_locations": {
            "method": "monopolar_triangulation"
        },
        "spike_locations": {
            "method": "grid_convolution"
        },
        "template_metrics": {
            "upsampling_factor": 10,
            "sparsity": null,
            "include_multi_channel_metrics": true
        },
        "principal_components": {
            "n_components": 5,
            "mode": "by_channel_local",
            "whiten": true
        }
    },
    "quality_metrics_names": [
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violation",
        "rp_violation",
        "sliding_rp_violation",
        "amplitude_cutoff",
        "amplitude_median",
        "amplitude_cv",
        "synchrony",
        "firing_range",
        "drift",
        "isolation_distance",
        "l_ratio",
        "d_prime",
        "nearest_neighbor",
        "silhouette"
    ],
    "quality_metrics": {
        "presence_ratio": {
            "bin_duration_s": 60
        },
        "snr": {
            "peak_sign": "neg", 
            "peak_mode": "extremum", 
            "random_chunk_kwargs_dict": null
        },
        "isi_violation": {
            "isi_threshold_ms": 1.5, 
            "min_isi_ms": 0
        },
        "rp_violation": {
            "refractory_period_ms": 1, 
            "censored_period_ms": 0.0
        },
        "sliding_rp_violation": {
            "bin_size_ms": 0.25,
            "window_size_s": 1,
            "exclude_ref_period_below_ms": 0.5,
            "max_ref_period_ms": 10,
            "contamination_values": null
        },
        "amplitude_cutoff": {
            "peak_sign": "neg",
            "num_histogram_bins": 100,
            "histogram_smoothing_value": 3,
            "amplitudes_bins_min_ratio": 5
        },
        "amplitude_median": {
            "peak_sign": "neg"
        },
        "amplitude_cv": {
            "average_num_spikes_per_bin": 50,
            "percentiles": [5, 95],
            "min_num_bins": 10,
            "amplitude_extension": "spike_amplitudes"
        },
        "firing_range": {
            "bin_size_s": 5, 
            "percentiles": [5, 95]
        },
        "synchrony": {
            "synchrony_sizes": [2, 4, 8]
        },
        "nearest_neighbor": {
            "max_spikes": 10000, 
            "n_neighbors": 4
        },
        "nn_isolation": {
            "max_spikes": 10000, 
            "min_spikes": 10, 
            "n_neighbors": 4, 
            "n_components": 10, 
            "radius_um": 100
        },
        "nn_noise_overlap": {
            "max_spikes": 10000, 
            "min_spikes": 10, 
            "n_neighbors": 4, 
            "n_components": 10, 
            "radius_um": 100
        },
        "silhouette": {
            "method": ["simplified"]
        }
    }
}
```

### Output

The output of this capsule is the following:

- `results/postprocessed_{recording_name}` folder, containing the postprocessed data (as a [WaveformExtractor](https://spikeinterface.readthedocs.io/en/latest/modules/core.html#waveformextractor) folder)
- `results/postprocessed-sorting_{recording_name}` folder, containing the spike sorted data, saved by SpikeInterface, after de-duplication
- `results/data_process_postprocessing_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

