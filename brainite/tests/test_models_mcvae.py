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
from brainite.losses import MCVAELoss
from brainite.models import MCVAE


class TestModelsMCVAE(unittest.TestCase):
    """ Test the MCVAE.
    """
    def setUp(self):
        """ Setup test.
        """
        self.model = MCVAE(
            latent_dim=2, n_channels=3, n_feats=[30, 4, 10],
            noise_init_logvar=-3, noise_fixed=False, sparse=False,
            vae_model="dense", vae_kwargs=None, nodecoding=False)
        self.loss = MCVAELoss(
            n_channels=1, beta=1., enc_channels=None,
            dec_channels=None, sparse=False, nodecoding=False)
        self.X = [
            torch.from_numpy(np.ones((10, 1, 30), dtype=np.float32)),
            torch.from_numpy(np.ones((10, 1, 4), dtype=np.float32)),
            torch.from_numpy(np.ones((10, 1, 10), dtype=np.float32))]

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test MCVAE forward.
        """
        out, layer_out = self.model(self.X)
        self.loss.layer_outputs = layer_out
        loss, extra_loss = self.loss(out)


if __name__ == "__main__":

    unittest.main()
