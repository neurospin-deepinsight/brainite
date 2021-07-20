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
from brainite.losses import PMVAELoss
from brainite.models import PMVAE


class TestModelsPMVAE(unittest.TestCase):
    """ Test the PMVAE.
    """
    def setUp(self):
        """ Setup test.
        """
        n_pathways = 4
        n_obs = 100
        membership_mask = np.random.randint(2, size=(n_pathways, n_obs))
        membership_mask = membership_mask.astype(bool)
        self.model = PMVAE(
            membership_mask, latent_dim=2, hidden_layers=[12],
            bias_last_layer=False, add_auxiliary_module=True,
            terms=["p{0}".format(idx) for idx in range(n_pathways)],
            activation=None)
        self.loss = PMVAELoss(beta=1)
        self.X = torch.from_numpy(np.ones((10, n_obs), dtype=np.float32))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test PMVAE forward.
        """
        out, layer_out = self.model(self.X)
        self.loss.layer_outputs = layer_out
        loss, extra_loss = self.loss(out, self.X)


if __name__ == "__main__":

    unittest.main()
