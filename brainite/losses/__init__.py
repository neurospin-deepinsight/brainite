# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common losses.
"""

# Imports
import sys
import inspect
from .vae import BetaBLoss
from .vae import BetaHLoss
from .vae import BtcvaeLoss
from .mcvae import MCVAELoss
from .pmvae import PMVAELoss


def get_losses():
    """ Get all available losses.

    Returns
    -------
    losses: dict
        a dictionnary containing all declared losses.
    """
    losses = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            losses[name] = obj
    return losses
