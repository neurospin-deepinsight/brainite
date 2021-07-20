# -*- coding: utf-8 -*-
"""
Pathway module factorized latent space
======================================

Credit: A Grigis

pmVAE leverages biological prior information in the form of pathway gene sets
to construct interpretable representations of single-cell data. It uses
pathway module sub networks to construct a latent space factorized by
pathway gene sets. Each pathway has a corresponding module which behaves as
a mini VAE acting only on the participating genes. These modules produce
latent representations that can be direcrly interpreted within the context
of a specific pathway. To account for overlapping pathway gene sets (due to
e.g. signaling hierarchies) a custom training procedure encourage module
independence.

The `test` variable must be set to False to run a full training.
"""

# Imports
import os
import sys
import time
import copy
from itertools import product
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import torch
from brainboard import Board
from dataify import SingleCellRNASeqDataset
from brainite.models import PMVAE
from brainite.losses import PMVAELoss

test = True
datasetdir = "/tmp/kang"
if not os.path.isdir(datasetdir):
    os.mkdir(datasetdir)
batch_size = 256
latent_dim = 4
n_epochs = 3 if test else 1200
learning_rate = 0.001
beta = 1e-5
if test:
    reductions = ["pca"]
else:
    reductions = ["tsne", "umap"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################################################
# Single-cell dataset
# -------------------
#
# Fetch & load the Kang dataset.

ds_train = SingleCellRNASeqDataset(root=datasetdir, train=True, seed=0)
ds_val = SingleCellRNASeqDataset(root=datasetdir, train=False, seed=0)
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ["train", "val"]}
gtpath = os.path.join(datasetdir, "kang_recons.h5ad")
membership_mask = pd.DataFrame(
    ds_train.data["membership_mask"], index=ds_train.data["membership_index"],
    columns=ds_train.data["membership_columns"])
print(membership_mask)

#############################################################################
# Training
# --------
#
# Create/train the model.

def train_model(dataloaders, model, device, criterion, optimizer,
                scheduler=None, n_epochs=100, checkpointdir=None,
                save_after_epochs=1, board=None, board_updates=None,
                load_best=False):
    """ General function to train a model and display training metrics.

    Parameters
    ----------
    dataloaders: dict of torch.utils.data.DataLoader
        the train & validation data loaders.
    model: nn.Module
        the model to be trained.
    device: torch.device
        the device to work on.
    criterion: torch.nn._Loss
        the criterion to be optimized.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    n_epochs: int, default 100
        the number of epochs.
    checkpointdir: str, default None
        a destination folder where intermediate models/histories will be
        saved.
    save_after_epochs: int, default 1
        determines when the model is saved and represents the number of
        epochs before saving.
    board: brainboard.Board, default None
        a board to display live results.
    board_updates: list of callable, default None
        update displayed item on the board.
    load_best: bool, default False
        optionally load the best model regarding the loss.
    """
    since = time.time()
    if board_updates is not None:
        board_updates = listify(board_updates)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    model = model.to(device)
    for epoch in range(n_epochs):
        print("Epoch {0}/{1}".format(epoch, n_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  
            else:
                model.eval()   
            running_loss = 0.0
            for batch_data, batch_labels in dataloaders[phase]:
                batch_data = batch_data.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward:
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs, layer_outputs = model(batch_data)
                    criterion.layer_outputs = layer_outputs
                    loss, extra_loss = criterion(outputs, batch_data)
                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * batch_data[0].size(0)
            if scheduler is not None and phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            print("{0} Loss: {1:.4f}".format(phase, epoch_loss))
            if board is not None:
                board.update_plot("loss_{0}".format(phase), epoch, epoch_loss)
            # Display validation classification results
            if board_updates is not None and phase == "val":
                for update in board_updates:
                    update(model, board, outputs, layer_outputs)
            # Deep copy the best model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        # Save intermediate results
        if checkpointdir is not None and epoch % save_after_epochs == 0:
            outfile = os.path.join(
                checkpointdir, "model_{0}.pth".format(epoch))
            checkpoint(
                model=model, outfile=outfile, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, epoch_loss=epoch_loss)
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:4f}".format(best_loss))
    # Load best model weights
    if load_best:
        model.load_state_dict(best_model_wts)


def listify(data):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    arr: list or array
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return data
    else:
        return [data]


def checkpoint(model, outfile, optimizer=None, scheduler=None,
               **kwargs):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: nn.Module
        the model to be saved.
    outfile: str
        the destination file name.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    kwargs: dict
        others parameters to be saved.
    """
    kwargs.update(model=model.state_dict())
    if optimizer is not None:
        kwargs.update(optimizer=optimizer.state_dict())
    if scheduler is not None:
        kwargs.update(scheduler=scheduler.state_dict())
    torch.save(kwargs, outfile)


model = PMVAE(
    membership_mask=membership_mask, latent_dim=latent_dim,
    hidden_layers=[12], add_auxiliary_module=True,
    terms=membership_mask.index, activation=None)
board = Board(env="pmvae")
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = PMVAELoss(beta=beta)
train_model(dataloaders, model, device, criterion, optimizer,
            n_epochs=n_epochs, board=board, board_updates=None,
            save_after_epochs=100)

#############################################################################
# Reduce the number of dimensions
# -------------------------------
#
# Use TSNE, UMAP or PCA to create a 2d representation of the results.

def test_model(dataloaders, model, device):
    """ General function to test a model.

    Parameters
    ----------
    dataloaders: dict of torch.utils.data.DataLoader
        the train & validation data loaders.
    model: nn.Module
        the trained model.
    device: torch.device
        the device to work on.
    """
    was_training = model.training
    model.eval()
    recons, latent_codes, labels = [], [], []
    with torch.no_grad():
        for idx, (batch_data, batch_labels) in enumerate(dataloaders["val"]):
            batch_data = batch_data.to(device)
            outputs, layer_outputs = model(batch_data)
            recons.append(outputs.detach().cpu().numpy())
            latent_codes.append(layer_outputs["z"].detach().cpu().numpy())
            labels.extend(batch_labels)
    model.train(mode=was_training)
    recons = np.concatenate(recons, axis=0)
    latent_codes = np.concatenate(latent_codes, axis=0)
    labels = np.asarray(labels)
    return recons, latent_codes, labels


def extract_pathway_cols(df, pathway):
    mask = df.columns.str.startswith(pathway + "-")
    return df.loc[:, mask]


def compute_reduction(recons, pathways, reduction="tsne"):
    if reduction not in ("tsne", "umap", "pca"):
        raise ValueError("Unexpected reduction type.")
    for key in pathways:
        if reduction == "tsne":
            reducer = TSNE(n_components=2)
        elif reduction == "umap":
            reducer = UMAP(n_components=2)
        else:
            reducer = PCA(n_components=2)
        codes = extract_pathway_cols(recons.obsm["codes"], key)
        embedding = pd.DataFrame(
            reducer.fit_transform(codes.values),
            index=recons.obs_names,
            columns=["{0}-0".format(key), "{0}-1".format(key)])
        yield embedding


result_file = os.path.join(datasetdir, "pmvae_kang_recons.h5ad")
generated_pathways = [
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
    "REACTOME_TCR_SIGNALING",
    "REACTOME_CELL_CYCLE"]
global_recon, z, labels = test_model(dataloaders, model, device)
print(" -- global recon:", global_recon.shape)
print(" -- z:", z.shape)
print(" -- labels:", labels.shape)
obs = np.asarray([item.split("-") for item in labels])
obs_df = pd.DataFrame(obs, columns=["index", "condition", "cell_type"])
obs_df.set_index("index", inplace=True)
print(obs_df)
recons = anndata.AnnData(
    pd.DataFrame(
        global_recon, index=obs[:, 0], columns=ds_val.data["var_names"]),
    obs=obs_df, varm=None)
recons.obsm["codes"] = pd.DataFrame(
    z, index=obs[:, 0], columns=model.latent_space_names())
for reduc_name in reductions:
    print("Computing {0} (may be long)...".format(reduc_name.upper()))
    recons.obsm["pathway_{0}".format(reduc_name)] = pd.concat(
        compute_reduction(recons, generated_pathways,
                          reduction=reduc_name),
        axis=1)
recons.write(result_file)


#############################################################################
# Display
# --------
#
# Display the results & the ground truth:

def extract_pathway_cols(df, pathway):
    mask = df.columns.str.startswith(pathway + "-")
    return df.loc[:, mask]


def tab20(arg):
    cmap = plt.get_cmap("tab20")
    return rgb2hex(cmap(arg))


generated_recons = anndata.read(result_file)
recons = anndata.read(gtpath)
recons.obsm["pathway_tsne"] = recons.obsm["pathway_tsnes"]
cmap = {
    "CD4 T": tab20(0),
    "CD8 T": tab20(1),
    "CD14 Mono": tab20(2),
    "CD16 Mono": tab20(3),
    "B": tab20(4),
    "DC": tab20(6),
    "NK": tab20(8),
    "T": tab20(10)}
pathways = [
    "INTERFERON_ALPHA_BETA_SIGNALIN",
    "CYTOKINE_SIGNALING_IN_IMMUNE_S",
    "TCR_SIGNALING",
    "CELL_CYCLE"]
conditions = [("GT", "tsne", recons, pathways)]
conditions += [("GENERATED", reduc, generated_recons, generated_pathways)
                for reduc in reductions]
for _name, _reduc, _recons, _pathways in conditions:
    print("--", _name, _reduc)
    fig, axes = plt.subplots(2, len(pathways), figsize=(6 * len(_pathways), 8))
    title = "{0} pathway factorized latent space results ({1})".format(
        _name, _reduc.upper())
    fig.suptitle(title, fontsize=15, y=0.99)
    pairs = product(["stimulated", "control"], _pathways)
    for ax, (active, key) in zip(axes.ravel(), pairs):
        mask = (_recons.obs["condition"] == active)
        codes = extract_pathway_cols(
            _recons.obsm["pathway_{0}".format(_reduc)], key)
        # plot non-active condition
        ax.scatter(*codes.loc[~mask].T.values, s=1, c="lightgrey", alpha=0.1) 
        # plot active condition
        ax.scatter(*codes.loc[mask].T.values,
                   c=list(map(cmap.get, _recons.obs.loc[mask, "cell_type"])),
                   s=1, alpha=0.5,)
        key = key.replace("REACTOME_", "")[:30]
        ax.set_title("{0} {1}".format(key, active), fontsize=10)
        ax.axis("off")
    fig.legend(
        handles=[Patch(color=c, label=l) for l,c in cmap.items()],
        ncol=4, loc=("lower center"), bbox_to_anchor=(0.5, 0.01),
        fontsize="xx-large", prop={"size": 10})
    plt.tight_layout()
    fig.subplots_adjust(bottom=.1)

plt.show()

