# NeXtSIM-4DVar

Source code to run the code developed for the paper 'Four-dimensional variational data assimilation with a sea-ice thickness emulator'.

## Installation
To install environment: 

```bash
conda env create -f environment.yml
```

To activate the environment:
```bash
conda activate env
```
## Build the dataset

The code to build the two datasets, the first one to train $f_{\theta}$ (dataset_evolution) and the second one to train $g_{\theta}$ and run the 4D--Var (dataset_full_state).


Original Files are download from nextsim NANUK outputs [[1]](#1). Those files are available through the SASIP github. [link to neXtSIM outputs](https://github.com/sasip-climate/catalog-shared-data-SASIP/blob/main/outputs/NANUK025.md). Forcings were dowloaded from ERA5 file [link](10.24381/cds.adbb2d47) 

Dataset is build and save under netCDF file with make_dataset script.
```bash
python make_dataset.py
```
To build the dataset with a TFRecord architecture, use make_tfrecord script
```bash
python make_tfrecord.py
```
Both datasets, need to be created.
