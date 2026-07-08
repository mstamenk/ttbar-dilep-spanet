# SPANET environment

This repository is set up for the first dileptonic ttbar SPANET calibration iteration:

- OS e-mu events with exactly two selected jets
- no required b-tag or jet-charge input
- ROOT ntuples converted to a fixed-shape HDF5 training file
- deterministic train/validation/test splits

## SPANET training environment

Create the training environment on an interactive node:

```bash
conda create -n ttbar-spanet python=3.9 -y
conda activate ttbar-spanet
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` prevents packages from `~/.local/lib/python*/site-packages` from leaking into the conda environment.

Install the SPANET-compatible PyTorch stack:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 "mkl<2024.1" -c pytorch -c conda-forge -y
```

Install SPANET and its training dependencies:

```bash
python -m pip install -r requirements_spanet.txt
```

Run the training-environment smoke test:

```bash
python environment/import_test_spanet.py
python -m pip check
```

## ROOT conversion environment

Use a separate environment for conversion if possible:

```bash
conda create -n ttbar-convert python=3.11 -y
conda activate ttbar-convert
export PYTHONNOUSERSITE=1
python -m pip install -r requirements_convert.txt
python environment/import_test.py
```

If the conversion must happen inside an older coffea analysis environment, use `requirements_convert_coffea_compat.txt` instead. It keeps `awkward<2`, `uproot==4.*`, `vector==0.11.0`, and `numpy==1.23.5`.

## Data and output paths

Use local scratch for large intermediate HDF5 files and model outputs. A typical layout is:

```text
$SCRATCH/ttbar-spanet/
  hdf5/
  checkpoints/
  logs/
  plots/
```

Keep immutable merged prod_v2 ROOT files on the shared filesystem, and write derived HDF5 outputs to scratch.

## Convert a small ROOT sample

```bash
python scripts/root_to_hdf5.py \
  --input "/path/to/prod_v2/merged/*.root" \
  --output "$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1_small.h5" \
  --max-events 50000
```

By default the converter uses all CPUs visible to the process, reads about `200000` input events per chunk, and writes `lzf`-compressed HDF5. Override the chunking with `--chunk-events` only if memory pressure or throughput requires it.

For a full 128-core OSCAR CPU job:

```bash
sbatch \
  --export=ALL,INPUT_GLOB="/path/to/prod_v2/merged/*.root",OUTPUT_H5="$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1.h5" \
  environment/oscar_convert.sbatch
```

If prod_v2 branch names differ from the defaults, pass comma-separated aliases, for example:

```bash
python scripts/root_to_hdf5.py \
  --input "/path/to/*.root" \
  --output "$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1.h5" \
  --jet-pt-branches Jet_pt,SelectedJet_pt \
  --electron-pt-branches Electron_pt,el_pt \
  --muon-pt-branches Muon_pt,mu_pt
```

## Sanity checks

```bash
python scripts/sanity_check_hdf5.py \
  --input "$SCRATCH/ttbar-spanet/hdf5/ttbar_dilep_v1_small.h5" \
  --output-dir "$SCRATCH/ttbar-spanet/plots/sanity_v1"
```
