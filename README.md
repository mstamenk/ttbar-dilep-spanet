# ttbar-dilep-spanet

SPANET workflow for dileptonic ttbar e-mu events with exactly two selected jets. The workflow converts the production ROOT ntuple to SPANET HDF5, trains a multitask SPANET model, evaluates the held-out test split, and writes an augmented ROOT file with predicted assignments, scores, neutrino/top regressions, and reconstructed W/top masses.

The commands below assume OSCAR-style paths and a checkout at:

```bash
cd ~/ttbar-spanet/ttbar-dilep-spanet
```

Use `PYTHONNOUSERSITE=1` in both environments. This prevents packages from `~/.local/lib/python*/site-packages` from leaking into conda environments, which has caused incompatible `torch`, `awkward`, `uproot`, and `numpy` versions on OSCAR.

## Repository Layout

- `configs/ttbar_dilep_event.yaml`: SPANET event definition.
- `configs/ttbar_dilep_options.json`: default SPANET model/training options.
- `requirements_spanet.txt`: training/evaluation environment packages.
- `requirements_convert.txt`: ROOT to HDF5 conversion packages.
- `requirements_convert_coffea_compat.txt`: optional converter stack for old coffea environments.
- `scripts/root_to_hdf5.py`: ROOT to SPANET HDF5 converter.
- `scripts/split_spanet_hdf5.py`: deterministic train/val/test HDF5 splitter.
- `scripts/validate_spanet_hdf5.py`: validates the HDF5 through SPANET's native dataset loader.
- `scripts/train_spanet.py`: compatibility wrapper around `spanet.train`.
- `scripts/evaluate_spanet_performance.py`: held-out test evaluation and plots.
- `scripts/infer_root_spanet.py`: single-process ROOT augmentation inference.
- `scripts/run_root_spanet_inference_8gpu.sh`: 8-GPU interactive ROOT augmentation launcher.

## 1. Create the SPANET Training Environment

Create this environment on a GPU node or a login node where conda can install the packages.

```bash
conda create -n ttbar-spanet python=3.9 -y
conda activate ttbar-spanet

export PYTHONNOUSERSITE=1

conda install \
  pytorch==1.12.1 \
  torchvision==0.13.1 \
  torchaudio==0.12.1 \
  cudatoolkit=11.6 \
  "mkl<2024.1" \
  -c pytorch -c conda-forge -y

python -m pip install -r requirements_spanet.txt
python -m pip install six requests

python environment/import_test_spanet.py
python -m pip check
```

Expected sanity-test output should include:

```text
spanet: ...
torch: 1.12.1
user site enabled: False
torch cuda available: True
```

If `torch cuda available` is `False` on a login or CPU node, rerun the import test on an allocated GPU node before training.

## 2. Create the ROOT Conversion Environment

Use a separate environment for conversion. The converter needs modern `uproot` and `awkward`, while some coffea workflows require older versions.

```bash
conda create -n ttbar-convert python=3.11 -y
conda activate ttbar-convert

export PYTHONNOUSERSITE=1

python -m pip install -r requirements_convert.txt
python environment/import_test.py
python -m pip check
```

If you must run conversion inside a coffea 0.7.x-style environment, install this instead:

```bash
python -m pip install -r requirements_convert_coffea_compat.txt
```

## 3. Optional Shell Aliases

These aliases are convenient but not required. Add them to `~/.bashrc` if desired:

```bash
alias ttbar-latest='export PYTHONNOUSERSITE=1; source ~/miniconda3/etc/profile.d/conda.sh; conda activate ttbar-spanet; cd ~/ttbar-spanet/ttbar-dilep-spanet'
alias ttbar-convert='export PYTHONNOUSERSITE=1; source ~/miniconda3/etc/profile.d/conda.sh; conda activate ttbar-convert; cd ~/ttbar-spanet/ttbar-dilep-spanet'
```

After editing:

```bash
source ~/.bashrc
```

## 4. Convert ROOT to SPANET HDF5

Use the conversion environment:

```bash
ttbar-convert
```

The current dileptonic merged ROOT file is:

```bash
/HEP/export/home/mstamenk/jet-charge-calibration/CMSSW_15_1_0_patch4/src/jet-charge-calib-miniaod/run/local_dilep_emu_os/merged_prod_v2/ttbar_dileptonic.root
```

Convert it to local scratch:

```bash
mkdir -p ~/scratch/ttbar-spanet/hdf5

python scripts/root_to_hdf5.py \
  --input /HEP/export/home/mstamenk/jet-charge-calibration/CMSSW_15_1_0_patch4/src/jet-charge-calib-miniaod/run/local_dilep_emu_os/merged_prod_v2/ttbar_dileptonic.root \
  --output ~/scratch/ttbar-spanet/hdf5/ttbar_dilep_v1.h5
```

Defaults:

- selects OS e-mu events with exactly two selected jets,
- keeps fully matched and partially matched events,
- writes `CLASSIFICATIONS/EVENT/reco_quality` with `1 = fully matched`, `0 = partial`, `-1 = unavailable`,
- writes `TARGETS/TopE/b` and `TARGETS/TopMu/b` for the electron-side and muon-side b-jet assignments,
- writes neutrino and top/tbar regression targets,
- uses all CPU cores visible to the process,
- uses event chunks, not byte chunks,
- writes `lzf` HDF5 compression by default.

## 5. Validate and Split the HDF5

Run the basic HDF5 checks in either environment with the needed packages:

```bash
python scripts/sanity_check_hdf5.py \
  --input ~/scratch/ttbar-spanet/hdf5/ttbar_dilep_v1.h5 \
  --output-dir ~/scratch/ttbar-spanet/plots/sanity_v1
```

Run the native SPANET loader validation in the SPANET environment:

```bash
ttbar-latest

python scripts/validate_spanet_hdf5.py \
  --input ~/scratch/ttbar-spanet/hdf5/ttbar_dilep_v1.h5 \
  --event-info configs/ttbar_dilep_event.yaml
```

Expected full-file validation should show about 3.85M selected events and these groups:

```text
inputs: ['Jets', 'Leptons', 'Met', 'Event']
assignments: ['TopE', 'TopMu']
classifications: ['EVENT/reco_quality']
```

Create explicit train/validation/test HDF5 files:

```bash
mkdir -p ~/scratch/ttbar-spanet/hdf5/splits

python scripts/split_spanet_hdf5.py \
  --input ~/scratch/ttbar-spanet/hdf5/ttbar_dilep_v1.h5 \
  --output-dir ~/scratch/ttbar-spanet/hdf5/splits \
  --prefix ttbar_dilep_v1
```

This produces:

```text
~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_train.h5
~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_val.h5
~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_test.h5
```

Use these split files for training and evaluation. Do not rely on SPANET's internal split when evaluating final test performance.

## 6. Train a SPANET Model

Run on an interactive 8-GPU node:

```bash
ttbar-latest

mkdir -p ~/scratch/ttbar-spanet/logs

python scripts/train_spanet.py \
  configs/ttbar_dilep_event.yaml \
  --options configs/ttbar_dilep_options.json \
  --training-file ~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_train.h5 \
  --validation-file ~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_val.h5 \
  --log-dir ~/scratch/ttbar-spanet/logs \
  --name ttbar_dilep_1M_8gpu_clean \
  --gpus 8 \
  --batch-size 4096 \
  --epochs 20 \
  -p 100
```

Notes:

- `scripts/train_spanet.py` is a wrapper that patches old SPANET imports for the installed PyTorch Lightning stack.
- `--gpus 8` uses Lightning distributed training.
- `--batch-size 4096` is per process in the SPANET/Lightning stack used here, so 8 GPUs produce fewer iterations per epoch than 1 GPU.
- `-p 100` means use 100 percent of the training sample.
- The default options are tuned to about 1M trainable parameters.

Training outputs are written under:

```text
~/scratch/ttbar-spanet/logs/ttbar_dilep_1M_8gpu_clean/version_0
```

## 7. Monitor Training with TensorBoard

On the GPU node:

```bash
tensorboard \
  --logdir ~/scratch/ttbar-spanet/logs/ttbar_dilep_1M_8gpu_clean \
  --host 0.0.0.0 \
  --port 6006
```

Then open an SSH tunnel from your laptop:

```bash
ssh -N -L 6006:localhost:6006 mstamenk@<login-host>
```

Open:

```text
http://localhost:6006
```

If the TensorBoard directory was copied to a laptop, run:

```bash
tensorboard --logdir /path/to/ttbar_dilep_1M_8gpu_clean --port 6006
```

## 8. Evaluate the Held-Out Test Split

Single-GPU evaluation is usually sufficient:

```bash
ttbar-latest

python scripts/evaluate_spanet_performance.py \
  --log-dir ~/scratch/ttbar-spanet/logs/ttbar_dilep_1M_8gpu_clean/version_0 \
  --input ~/scratch/ttbar-spanet/hdf5/splits/ttbar_dilep_v1_test.h5 \
  --event-info configs/ttbar_dilep_event.yaml \
  --output-dir ~/scratch/ttbar-spanet/eval/ttbar_dilep_1M_8gpu_clean_test \
  --batch-size 4096 \
  --gpus 1
```

The output directory contains:

- `summary.json`,
- assignment confusion plots for `TopE` and `TopMu`,
- assignment/detection probability plots,
- reco-quality ROC and confusion plots,
- regression residual plots.

## 9. Run ROOT Inference and Save an Augmented ROOT File

For final plotting, run inference on the original ROOT file and write one augmented ROOT file with original selected branches plus `spanet_*` branches.

On an interactive 8-GPU node:

```bash
ttbar-latest

./scripts/run_root_spanet_inference_8gpu.sh
```

This script:

- splits the ROOT file into 8 raw entry ranges,
- launches one Python process per GPU,
- writes shard ROOT files under `~/scratch/ttbar-spanet/root/ttbar_dileptonic_spanet_augmented_shards/`,
- merges shards with `hadd` into `~/scratch/ttbar-spanet/root/ttbar_dileptonic_spanet_augmented.root` when `hadd` is available.

Watch progress with:

```bash
tail -f ~/scratch/ttbar-spanet/root/ttbar_dileptonic_spanet_augmented_shards/ttbar_dileptonic_spanet_augmented_rank0.log
```

If the job is too slow, the bottleneck is usually ROOT I/O and copying all original branches, not GPU inference. For debugging, edit `scripts/run_root_spanet_inference_8gpu.sh` and set:

```bash
ORIGINAL_BRANCHES="minimal"
```

Then the output contains only event IDs/weights plus `spanet_*` outputs.

Important output branches include:

- `spanet_topE_b_index`, `spanet_topMu_b_index`,
- `spanet_topE_assignment_probability`, `spanet_topMu_assignment_probability`,
- `spanet_topE_detection_probability`, `spanet_topMu_detection_probability`,
- `spanet_reco_quality_prob_full`, `spanet_reco_quality_pred`,
- `spanet_nu_px`, `spanet_nu_py`, `spanet_nu_pz`,
- `spanet_nubar_px`, `spanet_nubar_py`, `spanet_nubar_pz`,
- `spanet_top_px`, `spanet_top_py`, `spanet_top_pz`, `spanet_top_e`,
- `spanet_tbar_px`, `spanet_tbar_py`, `spanet_tbar_pz`, `spanet_tbar_e`,
- `spanet_w_e_mass`, `spanet_w_mu_mass`,
- `spanet_topE_lnu_b_mass`, `spanet_topMu_lnu_b_mass`,
- `spanet_top_regressed_mass`, `spanet_tbar_regressed_mass`.

## 10. Common Problems

### User site packages leak into conda

Symptom: packages are imported from `~/.local/lib/python...`.

Fix:

```bash
export PYTHONNOUSERSITE=1
python -c "import site; print(site.ENABLE_USER_SITE)"
```

The printout should be `False`.

### `pkg_resources` deprecation warning

This warning is harmless if training runs:

```text
pkg_resources is deprecated as an API
```

The pinned `setuptools<81` avoids removal of `pkg_resources`.

### `No device id is provided via init_process_group or barrier`

This is a PyTorch distributed warning from the installed stack. It is harmless if all GPUs are active and training proceeds.

### `matplotlib` or `tensorboard` missing

Reinstall the SPANET requirements in the training environment:

```bash
ttbar-latest
python -m pip install -r requirements_spanet.txt
python -m pip check
```

### `awkward`, `uproot`, `vector`, or `numpy` conflicts

Do not mix the conversion requirements into the SPANET training environment unless necessary. Use:

- `ttbar-spanet` for training/evaluation,
- `ttbar-convert` for ROOT to HDF5 conversion.

## Commit Checklist

Before pushing:

```bash
python -m py_compile scripts/*.py
bash -n scripts/run_root_spanet_inference_8gpu.sh
git status --short
```

Review generated outputs and do not commit large files from `~/scratch`.
