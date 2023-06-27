# CODE REPOSITORY - Exploiting the structure of biochemical pathways to investigate dynamical properties with Neural Networks for graphs

## 1. Setup

Be sure to have the `conda` package manager installed on your machine.

### 1.1 Create virtual environment

```console
$ conda create -n petri-bio python=3.10 -y && conda activate petri-bio
```

### 1.2 Install package as editable

```console
$ pip install -e .
```

### 1.3 Install `pytorch`

Follow instructions based on your machine [here](https://pytorch.org/get-started/locally/).

### 1.4 Install `pytorch-geometric`

Follow instructions based on your machine [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### 1.5 Install `graphviz`

```console
$ conda install pygraphviz -c conda-forge -y
```

### 1.6 Restart the virtual environment

This will make the commands be available system-wise

```console
$ conda deactivate && conda activate petri-bio
```

### 1.7 Download the data
Data is hosted in this [this Zenodo repository](https://sandbox.zenodo.org/record/1206712). Download it and  place the `data` folder in the project directory.)

## 2. Reproduce preprocessing

Run:

```console
$ # conda activate petri-bio
$ pb-prep dataset=<DATASET>
```

Where `<DATASET>` is one among `robustness`, `sensitivity`, and `monotonicity`.

This will start the preprocessing (i.e. subgraphs creation, dataset creation in `pytorch-geometric` format) of the chosen property dataset.

**WARNING**: this process takes a long time, works best on multi-core machines.

## 3. Reproduce training experiments

Run:

```console
$ # conda activate petri-bio
$ pb-train --multirun \
    experiment=<EXPERIMENT> \
    data.name=<DATASET> \
    data.fold=<FOLD_INDEX> \
    model.optimizer.lr=0.01,0.001,0.0005 \
    model.embedder.num_layers=2,4,6,8,10 \
    model.embedder.dim_embed=128,256,512
```

Where:
- `<DATAMODULE>` is one among `robustness`, `sensitivity`, and `monotonicity`.
- `<EXPERIMENT>` is one of `0-baseline`, `1-connectivity`, `2-source-dest`, `3-node-type`, `4-edge-aware`. Each represents a model variant as developed in the paper.
-   `<FOLD_INDEX>` is one among `0 1 2 3 4`, and represents the fold that will be used for evaluation (the remaining folds will be used for training/validation).

**WARNING**: this command runs on a SLURM cluster, and works best on multi-GPU machines. If you don't have access to a SLURM cluster, you can run each configuration separately, e.g.:

```console
$ # conda activate petri-bio
$ pb-train \
    experiment=<EXPERIMENT> \
    data.name=<DATASET> \
    data.fold=<FOLD_INDEX> \
    model.optimizer.lr=0.01 \
    model.embedder.num_layers=2 \
    model.embedder.dim_embed=128
```

If you don't have GPUs, add `trainer=cpu` to the above command.

Results will be available in the `experiments/<DATAMODULE>/<EXPERIMENT>/train/fold_<FOLD_INDEX>/` directory. In the results directory you will find the following files:

- `exec_time.log`: time elapsed by the experiment.
- `checkpoints/epoch_XXX.ckpt`: checkpoint of the epoch with the best validation AUROC.
- `logs/metrics.csv`: metrics of the experiment (loss, AUROC, and accuracy).


## 3. Reproduce model evaluation

Once you have trained various hyper-parameter configurations, you can select the best model and use it for evaluating the model on a specific test fold. To do so, run:

```console
$ # conda activate petri-bio
$ pb-eval \
    experiment.name=<EXPERIMENT> \
    data.name=<DATAMODULE> \
    data.fold=<FOLD_INDEX>
```

where `<DATAMODULE>`, `<EXPERIMENT>`, and `<FOLD_INDEX>` are defined as above. The script will select the best model among all candidates and evaluate it on the `<FOLD_INDEX>`-th fold. Results will be available in the `experiments/<DATAMODULE>/<EXPERIMENT>/eval/fold_<FOLD_INDEX>` folder, where you wil find the following files:

- `exec_time.log`: time elapsed by the experiment.
- `logs/metrics.csv`: metrics of the experiment (loss, AUROC, and accuracy).
- `logs/hparams.yaml`: hyper-parameters of the model selected for the evaluation.
- `logs/test_predictions.csv`: a .csv file with predictions for all test examples.


## 4. Reproduce knockout experiments

Run:

```console
$ # conda activate petri-bio
$ pb-ko \
    data.name=<DATASET> \
    data.pathway_id=<PATHWAY_ID> \
    data.input_species=<INPUT_SPECIES> \
    data.output_species=<OUTPUT_SPECIES>
```

where:
- `<DATASET>` is one among `robustness`, `sensitivity`, and `monotonicity`.
- `<PATHWAY_ID>` is the BIOMD ID of the graph (check the `data/pathways` folder for the exact format). Be sure not to include the `.dot` extension, only the ID will suffice.
- `<INPUT_SPECIES>` is the input species. Be sure that it is a valid species for the pathway with ID `<PATHWAY_ID>`.
- `<OUTPUT_SPECIES>` is the output species. Be sure that it is a valid species for the pathway with ID `<PATHWAY_ID>`.

Results will be available in the `experiments/<DATASET>/4-edge-aware/knockout/<PATHWAY_ID>/<INPUT_SPECIES>__<OUTPUT_SPECIES>` folder, where you will find the following files:
- `predictions.csv`: a file containing a prediction for each version of the graph with a knocked-out edge.
- `knockout_graph.pdf`: a `.pdf` file to visualize how the knock-outs affect model predictions.


## 5. Reproduce time analysis experiments

Run:

```console
$ # conda activate petri-bio
$ pb-pred \
    data.name=<DATASET> \
    data.pathway_id=<PATHWAY_ID> \
    data.input_species=<INPUT_SPECIES> \
    data.output_species=<OUTPUT_SPECIES>
```

where:
- `<DATASET>` is one among `robustness`, `sensitivity`, and `monotonicity`.
- `<PATHWAY_ID>` is the BIOMD ID of the graph (check the `data/raw/pathways` folder for the exact format). Be sure not to include the `.dot` extension, only the ID will suffice.
- `<INPUT_SPECIES>` is the input species. Be sure that it is a valid species for the pathway with ID `<PATHWAY_ID>`.
- `<OUTPUT_SPECIES>` is the output species. Be sure that it is a valid species for the pathway with ID `<PATHWAY_ID>`.

Results will be available in the `experiments/<DATASET>/4-edge-aware/knockout/<PATHWAY_ID>/<INPUT_SPECIES>__<OUTPUT_SPECIES>` folder, where you will find the following files:
- `elapsed.txt`: the time taken to predict the example.

## 6. Notebooks

In the `notebooks` folder, you will find two ipython notebooks:
- `evaluation.ipynb` to reproduce Table 3 of the paper (results).
- `hparams.ipynb`, to reproduce the plots of validation AUROC as the hyper-parameters are varied (supplementary material).
