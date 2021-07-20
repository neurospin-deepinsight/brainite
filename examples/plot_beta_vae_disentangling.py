# -*- coding: utf-8 -*-
"""
Beta VAE disentangling
======================

Credit: A Grigis

Learning an interpretable factorised representation of the independent
data generative factors of the world without supervision is an important
precursor for the development of artificial intelligence that is able to
learn and reason in the sameway that humans do. This tutorial we use
beta-VAE to automated the discovery of interpretable factorised latent
representations from raw image data in a completely unsupervised manner.
To do so, we use the disentanglement test Sprites dataset where image data
have been generated from 6 disentangled latent factors.

The `test` variable must be set to False to run a full training.
"""

# Imports
import os
import sys
import glob
import time
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from dataify import DSpritesDataset
from brainboard import Board
from brainite.models import VAE
from brainite.losses import BetaHLoss, BetaBLoss, BtcvaeLoss
from brainite.utils import reconstruct_traverse, make_mosaic_img, add_labels

test = True
datasetdir = "/tmp/beta_vae_disentangling"
if not os.path.isdir(datasetdir):
    os.mkdir(datasetdir)
batch_size = 64
dataset_size = 100 if test else 737280
n_epochs = 30
adam_lr = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# Sprites dataset
# ---------------
#
# Fetch & load the dataset.

dataset = DSpritesDataset(root=datasetdir, size=dataset_size)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=1)

#############################################################################
# Disentanglement
# ---------------
#
# Create/train the model for different losses & plot traverse through all
# latent dimensions.

def train_model(dataloader, model, device, criterion, optimizer,
                scheduler=None, n_epochs=100, checkpointdir=None,
                save_after_epochs=1, board=None, board_updates=None,
                load_best=False):
    """ General function to train a model and display training metrics.

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        the data loader.
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
    dataset_size = len(dataloader)
    model = model.to(device)
    for epoch in range(n_epochs):
        print("Epoch {0}/{1}".format(epoch, n_epochs - 1))
        print("-" * 10)
        model.train()
        running_loss = 0.0
        for batch_data, batch_latent_vals in dataloader:
            batch_data = batch_data.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward:
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs, layer_outputs = model(batch_data)
                criterion.layer_outputs = layer_outputs
                loss, extra_loss = criterion(outputs, batch_data)
                # Backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
            # Statistics
            running_loss += loss.item() * batch_data[0].size(0)
        if scheduler is not None:
            scheduler.step()
        epoch_loss = running_loss / dataset_size
        print("Loss: {:.4f}".format(epoch_loss))
        if board is not None:
            board.update_plot("loss", epoch, epoch_loss)
        # Display validation classification results
        if board_updates is not None:
            for update in board_updates:
                update(model, board, outputs, layer_outputs)
        # Deep copy the best model
        if epoch_loss < best_loss:
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


def plot_losses(cache, filename):
    """ Display KL, LL, iterations plots.
    """
    if "kl" not in cache or "ll" not in cache:
        return
    ll = np.asarray(cache["ll"]).squeeze()
    kl = np.asarray(cache["kl"]).squeeze()
    fig, axs = plt.subplots(nrows=1, ncols=2)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for idx, dim_kl in enumerate(kl.T):
        axs[0].plot(
            dim_kl, color=colors[idx], label="dim{0}".format(idx + 1))
        axs[0].set_xlabel("Training iterations")
        axs[0].set_ylabel("KL")
        axs[1].plot(
            ll, dim_kl, color=colors[idx], label="dim{0}".format(idx + 1))
        axs[1].set_xlabel("Log Likelihood")
        axs[1].set_ylabel("KL")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(filename)


def plot_reconstructions(model, data, checkpointdir, filename=None):
    """ Display the reconstructed data over epochs.
    """
    weights_files = glob.glob(os.path.join(checkpointdir, "*.pth"))
    n_plots = len(weights_files)
    original = data.cpu().numpy()
    original = np.expand_dims(original, axis=0)
    stages = [original]
    labels = ["orig"]
    for idx, path in enumerate(sorted(
            weights_files, key=lambda x: int(os.path.basename(x)[6:-4]))):
        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        reconstruction = model.reconstruct(data, sample=False)
        reconstruction = np.expand_dims(reconstruction, axis=0)
        stages.append(reconstruction)
        labels.append("rec epoch {0}".format(epoch))
    concatenated = np.concatenate(stages, axis=0)
    mosaic = make_mosaic_img(concatenated)
    concatenated = Image.fromarray(mosaic)
    concatenated = add_labels(concatenated, labels)
    if filename is not None:
        concatenated.save(filename)
    return concatenated

losses = {
    "betah": BetaHLoss(beta=4, steps_anneal=0, use_mse=True),
    "betab": BetaBLoss(C_init=0.5, C_fin=25, gamma=100, steps_anneal=100000,
                       use_mse=True),
    "btcvae": BtcvaeLoss(dataset_size=len(dataset), alpha=1, beta=1, gamma=6,
                         is_mss=True, steps_anneal=0, use_mse=True)}
for loss_name in ("betah", "betab", "btcvae", ):

    # Train the model
    checkpointdir = os.path.join(datasetdir, "checkpoints", loss_name)
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)
    model = VAE(
        input_channels=1, input_dim=DSpritesDataset.img_size,
        conv_flts=[32, 32, 32, 32], dense_hidden_dims=[256, 256],
        latent_dim=10, noise_out_logvar=-3, noise_fixed=False,
        act_func=None, dropout=0, sparse=False)
    board = Board(env="betavae")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    criterion = losses[loss_name]
    train_model(dataloader, model, device, criterion, optimizer,
                n_epochs=n_epochs, board=board, board_updates=None,
                checkpointdir=checkpointdir, save_after_epochs=5)
    plot_losses(criterion.cache,
                os.path.join(datasetdir, "loss_{0}.png".format(loss_name)))

    # Display traverse
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    data = torch.unsqueeze(torch.from_numpy(
        dataset.X[index][:100].astype(np.float32)), dim=1).to(device)
    model.eval()
    name = "traverse_posteriror_{0}".format(loss_name)
    filename = os.path.join(datasetdir, "{0}.png".format(name))
    mosaic_traverse = reconstruct_traverse(
        model, data, n_per_latent=8, n_latents=None, is_posterior=True,
        filename=filename)
    filename = os.path.join(
        datasetdir, "reconstruction_stages_{0}.png".format(loss_name))
    recons = plot_reconstructions(
        model, data[:8], checkpointdir, filename=filename)

    plt.figure()
    plt.imshow(np.asarray(recons))
    plt.title("reconstruction_{0}".format(loss_name))
    plt.axis("off")

    plt.figure()
    plt.imshow(np.asarray(mosaic_traverse))
    plt.title(name)
    plt.axis("off")

plt.show()
