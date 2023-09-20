# Postprocessing for AIND ephys pipeline
## aind-capsule-ephys-postprocessing


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

The `data/` folder must include the output of the [aind-capsule-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-preprocessing) and the [aind-capsule-ephys-spikesort-pykilosort](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spiksort-pykilosort)/[aind-capsule-ephys-spikesort-kilosort25](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spiksort-kilosort25)capsules, including the `preprocessed_{recording_name}` and `spikesorted_{recording_name}` folders.

### Parameters

The `code/run` script takes no arguments. 
A full list of parameters used for postprocessing and quality metrics calculation can be found at the top of the `code/run_capsule.py` script.

### Output

The output of this capsule is the following:

- `results/postprocessed_{recording_name}` folder, containing the postprocessed data (as a [WaveformExtractor](https://spikeinterface.readthedocs.io/en/latest/modules/core.html#waveformextractor) folder)
- `results/postprocessed-sorting_{recording_name}` folder, containing the spike sorted data, saved by SpikeInterface, after de-duplication
- `results/data_process_postprocessing_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

