
This code is based on the CCA-SSG model proposed in the NeurIPS 2021 paper [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) and is modified to plugin in the $rLap$ augmentor.

Especially, we add the DGL version of our $rLap$ augmentor to run the experiments and change the following files:

- `aug.py`: We add support for 10 augmentors based on our PyG based experiments for the DGL design.
- `main.py`: A new argument parser option `--aug` is added to choose between the augmentors.
- `run.sh` Helper file to run all experiments
- `process_results.py`: Helper script to prepare the markdown table of results.

Rest of the code is same as the one available here: https://github.com/hengruizhang98/CCA-SSG

## Usage

We have included a helper notebook that can run on COLAB for convenience. To run it:
1. upload the `CCA_SSG_rLap.ipynb` to COLAB.
2. Now, upload the code provided in the anonymous url to COLAB. Make sure to rename the uploaded folder to `rlap`.
3. Execute the cells in order to get the results.

Alternatively, if you have an environment setup already, use the following bash script:

```bash
$ bash run.sh
```

This script will run multiple experiments for a dataset-augmentor combination and even print the results in a markdown table in stdout. 

_If you want to run the python scripts independently, please follow the instructions below:_


```python
# Cora with EdgeDropping (default CCA-SSG augmentor)
python main.py --aug ED --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4
# Cora with rLap
python main.py --aug RLAP --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4

# The same options can be used for the other datasets. By default, the ED augmentor is chosen
# Please set `--aug RLAP` explicitly to use the `rLap` augmentor.

# Citeseer
python main.py --dataname citeseer --epochs 20 --n_layers 1 --lambd 5e-4 --dfr 0.0 --der 0.4 --lr2 1e-2 --wd2 1e-2

# Pubmed
python main.py --dataname pubmed --epochs 100 --lambd 1e-3 --dfr 0.3 --der 0.5 --lr2 1e-2 --wd2 1e-4

# Amazon-Computer
python main.py --dataname comp --epochs 50 --lambd 5e-4 --dfr 0.1 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Amazon-Photo
python main.py --dataname photo --epochs 50 --lambd 1e-3 --dfr 0.2 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Coauthor-CS
python main.py --dataname cs --epochs 50 --lambd 1e-3 --dfr 0.2 --lr2 5e-3 --wd2 1e-4 --use_mlp

# Coauthor-Physics
python main.py --dataname physics --epochs 100 --lambd 1e-3 --dfr 0.5 --der 0.5 --lr2 5e-3 --wd2 1e-4
```
