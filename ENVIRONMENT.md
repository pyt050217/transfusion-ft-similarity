# Environment and Reproduction

## Tested environment

The experiments were run with:

- Python 3.10.22
- PyTorch 2.5.1+cu121
- NumPy 2.2.6
- Pandas 2.3.3
- scikit-learn 1.7.2
- NVIDIA GPU with CUDA 12.1-compatible PyTorch

The exact environment has been tested on the original server.
A clean installation from these files has not yet been validated.

## Create the Conda environment

```bash
conda env create -f environment.yml
conda activate transfusion-ft
```

## Install PyTorch and Python dependencies

Install the CUDA 12.1 build of PyTorch:

```bash
python -m pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## FTTransformer implementation

The experiment uses the source snapshot in:

`vendor/tab-transformer-pytorch/`

The package was reported as version 0.6.1 in the original
editable installation. The exact upstream commit and equivalence
to the published release have not been established.

The source snapshot retains its upstream LICENSE file.

## Verify imports

Run from the repository root:

```bash
PYTHONPATH="$PWD/vendor/tab-transformer-pytorch" \
python -c "import torch; \
from tab_transformer_pytorch import FTTransformer; \
print('PyTorch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('FTTransformer import OK')"
```

## Data requirements

Clinical data and processed patient-level datasets are not included
in this repository. The user must have authorized access to the
corresponding prepared datasets before running the experiments.

See the experiment-specific README files for data paths and
training commands.
