# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import unittest
import numpy as np
import torch
from brainite.losses import BetaHLoss, BetaBLoss, BtcvaeLoss
from brainite.models import VAE


class TestModelsVAE(unittest.TestCase):
    """ Test the VAE.
    """
    def setUp(self):
        """ Setup test.
        """
        self.model = VAE(
            input_channels=1, input_dim=[64, 64], conv_flts=[16, 16],
            dense_hidden_dims=[128], latent_dim=2, noise_out_logvar=-3,
            noise_fixed=True, log_alpha=None, act_func=None,
            final_activation=False, dropout=0, sparse=False, encoder=None,
            decoder=None)
        self.X = torch.from_numpy(np.ones((10, 1, 64, 64), dtype=np.float32))
        self.losses = {
            "betah": BetaHLoss(beta=4, steps_anneal=0, use_mse=True),
            "betab": BetaBLoss(
                C_init=0.5, C_fin=25, gamma=100, steps_anneal=100000,
                use_mse=True),
            "btcvae": BtcvaeLoss(
                dataset_size=len(self.X), alpha=1, beta=1, gamma=6,
                is_mss=True, steps_anneal=0, use_mse=True)}

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test VAE forward.
        """
        out, layer_out = self.model(self.X)
        for _loss in self.losses.values():
            _loss.layer_outputs = layer_out
            loss, extra_loss = _loss(out, self.X)


if __name__ == "__main__":

    unittest.main()
