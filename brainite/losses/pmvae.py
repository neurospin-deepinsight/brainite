# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
PMVAE loss.
"""

# Imports
import numpy as np
import torch
from torch.nn import functional as func


class PMVAELoss(object):
    """ PMVAE loss.

    Compute a global & a local (per pathway) reconstruction loss and a
    KL divergence regularization loss with beta weighting.
    """
    def __init__(self, beta=1):
        """ Init class.

        Parameters
        ----------
        beta: float, default 1
            the weight of KL term regularization.
        """
        super(PMVAELoss, self).__init__()
        self.layer_outputs = None
        self.beta = beta

    def __call__(self, global_recon, target, *args, **kwargs):
        """ Compute the loss.
        """
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters z, mu, logvar.")
        mu = self.layer_outputs["mu"]
        logvar = self.layer_outputs["logvar"]
        z = self.layer_outputs["z"]
        model = self.layer_outputs["model"]
        module_outputs = self.layer_outputs["module_outputs"]
        device = global_recon.device

        def weighted_mse(y_true, y_pred, sample_weight):
            sample_weight = torch.from_numpy(sample_weight.astype(np.float32))
            sample_weight = sample_weight.to(device)
            diff = torch.square(y_true - y_pred) * sample_weight
            wmse = torch.sum(diff, dim=-1) / torch.sum(sample_weight)
            return wmse

        kl = torch.exp(logvar) + mu ** 2 - logvar - 1
        kl = 0.5 * torch.sum(kl, dim=1)
        kl = kl.mean()

        global_recon_loss = func.mse_loss(global_recon, target,
                                          reduction="mean")
        # global_recon_loss = torch.sum(global_recon_loss, dim=1).mean()

        local_recon_losses = []
        for feat_mask, module_mask in model.get_masks_for_local_losses():
            # Dropout other modules & reconstruct
            module_mask = torch.from_numpy(module_mask.astype(np.float32))
            module_mask = module_mask.to(device)
            only_active_module = torch.mul(module_outputs, module_mask)
            local_recon = model.merger(only_active_module)

            # Only compute the loss with participating genes
            wmse = weighted_mse(target, local_recon, feat_mask)

            local_recon_losses.append(wmse)

        local_recon_losses = torch.stack(local_recon_losses, dim=1)
        local_recon_loss = torch.sum(local_recon_losses, dim=1)
        local_recon_loss = local_recon_loss / model.n_annotated_modules
        local_recon_loss = local_recon_loss.mean()

        loss = global_recon_loss + local_recon_loss + self.beta * kl

        return loss, {"global_recon_loss": global_recon_loss,
                      "local_recon_loss": local_recon_loss, "kl": kl}
