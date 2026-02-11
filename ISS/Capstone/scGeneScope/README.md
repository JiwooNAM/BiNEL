# scGeneScope

scGeneScope: A Perturbationally-Paired Single Cell Imaging and Transcriptomics Dataset and Benchmark for Treatment Response Modeling


## Installation

1. Install poetry 1.8.5 

```shell
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.5
```

2. Clone the repo and cd into the directory:

```shell
git clone git@github.com:altoslabs/scGeneScope.git
cd scGeneScope
```

3. Installation with poetry native virtual environment 

Make sure python 3.11 is available in your PATH (system dependent)

If python 3.11 is not available, you can install it with [pyenv](https://github.com/pyenv/pyenv) or create a conda env with python 3.11 and point poetry to that env with `poetry env use /path/to/conda/env/bin/python`.

Run `poetry shell` to create/activate a virtual environment.

4. Install the dependencies

```shell
poetry install --with dev
```
5. To activate the project environment:
- Go to the project folder and run `poetry shell`

## Downloading the embedding data and original data from HuggingFace

1. Set the required env variables.
```shell
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN={your_token_here}
```
2. Download the two h5ad files per model that contains the embeddings to the data directory.
```shell
cd <path-to-repo-folder>
huggingface-cli download altoslabs/scGeneScope features/rnaseq/pca/n2000/round_1.h5ad --repo-type dataset --local-dir ./data/ --local-dir-use-symlinks False
huggingface-cli download altoslabs/scGeneScope features/rnaseq/pca/n2000/round_2.h5ad --repo-type dataset --local-dir ./data/ --local-dir-use-symlinks False
```

Available embeddings:
- features/rnaseq/pca/n2000
- features/rnaseq/scvi/n200
- features/rnaseq/scvi/scvi_1
- features/rnaseq/scvi/scvi_2
- features/rnaseq/geneformer
- features/rnaseq/scgpt
- features/rnaseq/UCE/4layer

- features/imaging/imagenet/vit-l
- features/imaging/imagenet/vit-h
- features/imaging/imagenet/resnet50
- features/imaging/imagenet/resnet152
- features/imaging/imagenet/resnet50_clip
- features/imaging/imagenet/vit-h_clip
- features/imaging/imagenet/openphenom


3. Download the original imaging and scRNAseq data. 
Note -- all paper results and scGeneScope operations and results can be generated from the precomputed embeddings above. Users can download the original imaging and scRNAseq data to generate new embeddings or run new experiments on the raw data.

To download the scRNAseq data (~80G), run:
```shell
./scripts/download_scRNAseq_data.sh
```

To download a single plate of the imaging data (~173M, 3405 files):
```shell
./scripts/download_single_imaging_plate.sh
```

To download all of the imaging data (~186G, ~4,200,000 files):
```shell
./scripts/download_all_imaging_data.sh
```

Caution, the full imaging data download takes a while due to the large number of files associated.

## Usage

### Hydra Training Script
To run end-to-end model training, we provide a training script that is integrated with the Hydra configuration system. This script can be found under `src/scgenescope/scripts/train.py` and can be executed as follows:

```python
cd <path-to-repo-folder>
python src/scgenescope/scripts/train.py <config-options>
```
The configuration options are discussed in the Configuration System section. The `train.py`` is also added to the environment as an executable script. As a result the above command can be shortened and called as follows:
```python
train <config-options>
```
You can run an experiment as follows. You can replace the experiment name with any experiment available under src/scgenescope/config/experiment
```python
train experiment=rnaseq/singleprofile/train_on_scgpt
```

### Configuration System

This repo uses Hydra for configuration system management. A configuration setup and default settings are stored in `src/scgenescope/config`. This folders and the contained files should not be modified unless there are updates to the model codebase (such as adding new models, datasets, loggers, ...).

Below we describe potential workflows to use the configuration system to scale your experimentation:

1) Override any configuration parameter from the commandline
```python
train trainer.max_epoch=100 model.classifier.hidden_dim=1024 model.classifier.depth=5
```
This command overrides the `cfg.trainer.max_epoch` value and sets it to `100`, overrides `cfg.model.classifier.hidden_dim` value and sets it to `1024` and , overrides `cfg.model.classifier.depth` value and sets it to `5` 

2) Add any additional parameters that were not defined in the configuration system
```python
python train.py +trainer.max_steps=5000
``` 
This will add an attribute field `gradient_clip_val` to the trainer.

## Acknowledgments

Some of the hydra configs and utility functions are inspired from the lightning-hydra-template repo ([https://github.com/ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)).




