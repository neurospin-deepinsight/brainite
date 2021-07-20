# -*- coding: utf-8 -*-
"""
Multi Channels VAE (MCVAE)
==========================

Credit: A Grigis & C. Ambroise

The Multi Channel VAE (MCVAE) is an extension of the variational autoencoder
able to jointly model multiple data source named channels.

The `test` variable must be set to False to run a full training.
"""

import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from brainboard import Board
from brainite.models import MCVAE
from brainite.losses import MCVAELoss

test = True
n_samples = 500
n_channels = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 10
adam_lr = 2e-3
n_epochs = 3 if test else 5000
if test:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################################
# Create synthetic data
# ---------------------
#
# Generate multiple sources (channels) of data through a linear generative
# model

class GeneratorUniform(nn.Module):
    """ Generate multiple sources (channels) of data through a linear
    generative model:
    z ~ N(0,I)
    for c_idx in n_channels:
        x_ch = W_ch(c_idx)
    where 'W_ch' is an arbitrary linear mapping z -> x_ch
    """
    def __init__(self, lat_dim=2, n_channels=2, n_feats=5, seed=100):
        super(GeneratorUniform, self).__init__()
        self.lat_dim = lat_dim
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.seed = seed
        np.random.seed(self.seed)

        W = []
        for c_idx in range(n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim))
            u, s, vt = np.linalg.svd(w_, full_matrices=False)
            w = (u if self.n_feats >= lat_dim else vt)
            W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
            W[c_idx].weight.data = torch.FloatTensor(w)

        self.W = torch.nn.ModuleList(W)

    def forward(self, z):
        if isinstance(z, list):
            return [self.forward(_) for _ in z]
        if type(z) == np.ndarray:
            z = torch.FloatTensor(z)
        assert z.size(1) == self.lat_dim
        obs = []
        for ch in range(self.n_channels):
            x = self.W[ch](z)
            obs.append(x.detach())
        return obs


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=500, lat_dim=2, n_feats=5, n_channels=2,
                 generatorclass=GeneratorUniform, snr=1, train=True):
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.snr = snr
        self.train = train
        seed = (7 if self.train is True else 14)
        np.random.seed(seed)
        self.z = np.random.normal(size=(self.n_samples, self.lat_dim))
        self.generator = generatorclass(
            lat_dim=self.lat_dim, n_channels=self.n_channels,
            n_feats=self.n_feats)
        self.x = self.generator(self.z)
        self.X, self.X_noisy = preprocess_and_add_noise(self.x, snr=snr)
        self.X = [np.expand_dims(x.astype(np.float32), axis=1) for x in self.X]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return [x[item] for x in self.X]

    @property
    def shape(self):
        return (len(self), len(self.X))


def preprocess_and_add_noise(x, snr, seed=0):
    if not isinstance(snr, list):
        snr = [snr] * len(x)
    scalers = [StandardScaler().fit(c_arr) for c_arr in x]
    x_std = [scalers[c_idx].transform(x[c_idx]) for c_idx in range(len(x))]
    # seed for reproducibility in training/testing based on prime number basis
    seed = (seed + 3 * int(snr[0] + 1) + 5 * len(x) + 7 * x[0].shape[0] +
            11 * x[0].shape[1])
    np.random.seed(seed)
    x_std_noisy = []
    for c_idx, arr in enumerate(x_std):
        sigma_noise = np.sqrt(1. / snr[c_idx])
        x_std_noisy.append(arr + sigma_noise * np.random.randn(*arr.shape))
    return x_std, x_std_noisy


ds_train = SyntheticDataset(
    n_samples=n_samples, lat_dim=true_lat_dims, n_feats=n_feats,
    n_channels=n_channels, train=True, snr=snr)
ds_val = SyntheticDataset(
    n_samples=n_samples, lat_dim=true_lat_dims, n_feats=n_feats,
    n_channels=n_channels, train=False, snr=snr)
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=n_samples, shuffle=True, num_workers=1)
        for x in ["train", "val"]}

############################################################################
# Sparse vs non sparse
# --------------------
#
# Train a sparse and a non sparse MCVAE.

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
            for batch_data in dataloaders[phase]:
                batch_data = to_device(batch_data, device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward:
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs, layer_outputs = model(batch_data)
                    criterion.layer_outputs = layer_outputs
                    loss, extra_loss = criterion(outputs)
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


def to_device(data, device):
    """ Transfer data to device.

    Parameters
    ----------
    data: tensor or list of tensor
        the data to be transfered.
    device: torch.device
        the device to work on.

    Returns
    -------
    out: tensor or list of tensor
        the transfered data.
    """
    if isinstance(data, list):
        return [tensor.to(device) for tensor in data]
    else:
        return data.to(device)


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


def update_dropout_rate(model, board, outputs, layer_outputs=None):
    """ Display the dropout rate.
    """
    if model.log_alpha is not None:
        do = np.sort(model.dropout.numpy().reshape(-1))
        board.update_hist("dropout_probability", do)


models = {}
torch.manual_seed(42)
models["mcvae"] = MCVAE(
    latent_dim=fit_lat_dims, n_channels=n_channels,
    n_feats=[n_feats] * n_channels, vae_model="dense",
    vae_kwargs={}, sparse=False)
torch.manual_seed(42)
models["smcvae"] = MCVAE(
        latent_dim=fit_lat_dims, n_channels=n_channels,
    n_feats=[n_feats] * n_channels, vae_model="dense",
    vae_kwargs={}, sparse=True)
for model_name, model in models.items():
    print("- model:", model_name)
    print(model)
    board = Board(env=model_name)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    criterion = MCVAELoss(n_channels, beta=1., sparse=False)
    train_model(dataloaders, model, device, criterion, optimizer,
                n_epochs=n_epochs, board=board,
                board_updates=update_dropout_rate)

############################################################################
# Display results

pred = {}  # Prediction
z = {}     # Latent Space
g = {}     # Generative Parameters
x_hat = {}  # Reconstructed channels
for model_name, model in models.items():
    model.eval()
    X = [torch.from_numpy(x).to(device) for x in datasets["val"].X]
    print("--", model_name)
    print("-- X", [x.size() for x in X])

    with torch.no_grad():
        q = model.encode(X)  # encoded distribution q(z|x)
    print("-- encoded distribution q(z|x)", [n for n in q])

    z[model_name] = model.p_to_prediction(q)
    print("-- z", [e.shape for e in z[model_name]])

    if model.sparse:
        z[model_name] = model.apply_threshold(z[model_name], 0.2)
    z[model_name] = np.array(z[model_name]).reshape(-1) # flatten
    print("-- z", z[model_name].shape)

    g[model_name] = [
        model.vae[c_idx].encode.w_mu.weight.detach().cpu().numpy()
        for c_idx in range(n_channels)]
    g[model_name] = np.array(g[model_name]).reshape(-1)  #flatten


############################################################################
# With such a simple dataset, mcvae and sparse-mcvae gives the same results in
# terms of latent space and generative parameters.
# However, only with the sparse model is able to easily identify the
# important latent dimensions.

plt.figure()
plt.subplot(1,2,1)
plt.hist([z["smcvae"], z["mcvae"]], bins=20, color=["k", "gray"])
plt.legend(["Sparse", "Non sparse"])
plt.title("Latent dimensions distribution")
plt.ylabel("Count")
plt.xlabel("Value")
plt.subplot(1,2,2)
plt.hist([g["smcvae"], g["mcvae"]], bins=20, color=["k", "gray"])
plt.legend(["Sparse", "Non sparse"])
plt.title(r"Generative parameters $\mathbf{\theta} = \{\mathbf{\theta}_1 "
          r"\ldots \mathbf{\theta}_C\}$")
plt.xlabel("Value")

do = np.sort(models["smcvae"].dropout.numpy().reshape(-1))
plt.figure()
plt.bar(range(len(do)), do)
plt.suptitle("Dropout probability of {0} fitted latent dimensions in Sparse "
             "Model".format(fit_lat_dims))
plt.title("{0} true latent dimensions".format(true_lat_dims))

plt.show()
