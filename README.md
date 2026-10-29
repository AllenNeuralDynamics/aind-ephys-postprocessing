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

Parameters are read from a `params.json` file that must be in the root folder of the capsule (next to `code/`, i.e. `../params.json` from `code/`), with a `logging` and a `postprocessing` section (keys such as `n_jobs` (default -1), `max_spikes_per_unit`, `additional_extensions`).

A full list of parameters can be found in the in the `aind-ephys-pipeline` repository, and are defined in this [schema](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline/blob/1.5.0/pipeline/default_params_schema.json).

### Output

The output of this capsule is the following:

- `results/postprocessed_{recording_name}` folder, containing the postprocessed data (as a [WaveformExtractor](https://spikeinterface.readthedocs.io/en/latest/modules/core.html#waveformextractor) folder)
- `results/postprocessed-sorting_{recording_name}` folder, containing the spike sorted data, saved by SpikeInterface, after de-duplication
- `results/data_process_postprocessing_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

