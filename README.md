# HSGLM
The official Pytorch implementation of paper "Hierarchical Spatiotemporal Graph Learning with Mamba for Dynamic Brain Functional Connectivity Network Analysis" accepted by 

# Abstract

# Overall Pipeline

# Highlights

## Acknowledgements & Third-Party Code
We gratefully acknowledge the following open-source projects that helped shape this repository:
- **[alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py)** (MIT): Structure and implementation details in `mamba.py` follow alxndrTL's implementation.
Please refer to individual files for specific license headers.

# Dependencies & Environment Setup
This project is implemented in Python 3.9 with PyTorch 1.12.1 and CUDA 11.3. Follow the steps below to set up your environment:

```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install torch-scatter -f [https://data.pyg.org/whl/torch-1.12.1+cu113.html](https://data.pyg.org/whl/torch-1.12.1+cu113.html)
pip install torch-sparse -f [https://data.pyg.org/whl/torch-1.12.1+cu113.html](https://data.pyg.org/whl/torch-1.12.1+cu113.html)
pip install torch-geometric==2.6.1
pip install -r requirements.txt


