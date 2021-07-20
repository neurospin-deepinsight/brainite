# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common architectures.
"""

# Imports
import sys
import inspect
from .vae import VAE
from .mcvae import MCVAE
from .pmvae import PMVAE


def get_models():
    """ Get all available models.

    Returns
    -------
    models: dict
        a dictionnary containing all declared models.
    """
    models = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            models[name] = obj
    return models
