# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
MCVAE loss.
"""

# Imports
import torch
from torch.nn import functional as func
from torch.distributions import Normal, kl_divergence


class MCVAELoss(object):
    """ MCVAE loss.

    Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Luigi Antelmi, Nicholas Ayache, Philippe Robert,
    Marco Lorenzi Proceedings of the 36th International Conference on Machine
    Learning, PMLR 97:302-311, 2019.

    MCVAE consists of two loss functions:

    1. KL divergence loss: how off the distribution over the latent space is
       from the prior. Given the prior is a standard Gaussian and the inferred
       distribution is a Gaussian with a diagonal covariance matrix,
       the KL-divergence becomes analytically solvable.
    2. log-likelihood LL

    loss = beta * KL_loss + LL_loss.
    """
    def __init__(self, n_channels, beta=1., enc_channels=None,
                 dec_channels=None, sparse=False, nodecoding=False):
        """ Init class.

        Parameters
        ----------
        n_channels: int
            the number of channels.
        beta, float, default 1.
            for beta-VAE.
        enc_channels: list of int, default None
            encode only these channels (for kl computation).
        dec_channels: list of int, default None
            decode only these channels (for ll computation).
        sparse: bool, default False
            use sparsity contraint.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(MCVAELoss, self).__init__()
        self.n_channels = n_channels
        self.beta = beta
        self.sparse = sparse
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        if enc_channels is None:
            self.enc_channels = list(range(n_channels))
        else:
            assert(len(enc_channels) <= n_channels)
        if dec_channels is None:
            self.dec_channels = list(range(n_channels))
        else:
            assert(len(dec_channels) <= n_channels)
        self.n_enc_channels = len(self.enc_channels)
        self.n_dec_channels = len(self.dec_channels)
        self.nodecoding = nodecoding
        self.layer_outputs = None

    def __call__(self, p):
        """ Compute loss.

        Parameters
        ----------
        p: list of Normal distributions (C,) -> (N, F)
            reconstructed channels data.
        x: list of Tensor, (C,) -> (N, F)
            inputs channels data.
        """
        if self.nodecoding:
            return -1
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        x = self.layer_outputs["x"]
        q = self.layer_outputs["q"]
        kl = self.compute_kl(q, self.beta)
        ll = self.compute_ll(p, x)

        total = kl - ll
        return total, {"kl": kl, "ll": ll}

    def compute_kl(self, q, beta):
        kl = 0
        if not self.sparse:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += kl_divergence(qi, Normal(
                        0, 1)).sum(-1, keepdim=True).mean(0)
        else:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += self._kl_log_uniform(qi).sum(
                            -1, keepdim=True).mean(0)
        return beta * kl / self.n_enc_channels

    def compute_ll(self, p, x):
        # p[x][z]: p(x|z)
        ll = 0
        for c_idx1 in range(self.n_channels):
            for c_idx2 in range(self.n_channels):
                if c_idx1 in self.dec_channels and c_idx2 in self.enc_channels:
                    ll += self._compute_ll(
                        p=p[c_idx1][c_idx2], x=x[c_idx1]).mean(0)
        return ll / self.n_enc_channels / self.n_dec_channels

    def compute_log_alpha(self, mu, logvar):
        # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
        return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(
            min=-8, max=8)

    def _compute_ll(self, p, x):
        ll = p.log_prob(x).view(len(x), -1)
        return ll.sum(-1, keepdim=True)

    def _kl_log_uniform(self, normal):
        """
        Paragraph 4.2 from:
        Variational Dropout Sparsifies Deep Neural Networks
        Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
        https://arxiv.org/abs/1701.05369
        https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
        blob/master/KL%20approximation.ipynb
        """
        mu = normal.loc
        logvar = normal.scale.pow(2).log()
        log_alpha = self.compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
                  torch.log1p(torch.exp(-log_alpha)) - k1)
        return - neg_kl
