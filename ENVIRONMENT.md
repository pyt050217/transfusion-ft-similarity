# Environment

Primary environment: `tab310`

Tested setup:

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- NVIDIA GPU
- FTTransformer: lucidrains/tab-transformer-pytorch

## Recreate

```bash
conda env create -f environment.yml
conda activate tab310
The original training environment used a CUDA 12.1 compatible PyTorch build.
External implementation
FTTransformer implementation:
https://github.com/lucidrains/tab-transformer-pytorch
