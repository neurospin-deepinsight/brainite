# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Module providing NeuroImaging & Genetic Data Integration Tools for PyTorch.
"""
SUMMARY = """
.. container:: summary-carousel

    `brainite` is a PyTorch toolbox that provides NeuroImaging & Genetic
    Data Integration methods.
"""
long_description = (
    "Module providing NeuroImaging & Genetic Data Integration Tools for "
    "PyTorch.\n")

# Main setup parameters
NAME = "brainite"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin webPage"
EXTRAURL = (
    "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/"
    "NeuroSpin.aspx")
LINKS = {"deepinsight": "https://github.com/neurospin-deepinsight/deepinsight"}
URL = "https://github.com/neurospin-deepinsight/brainite"
DOWNLOAD_URL = "https://github.com/neurospin-deepinsight/brainite"
LICENSE = "CeCILL-B"
AUTHOR = """
brainite developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["brainite"]
REQUIRES = [
    "numpy>=1.17.1",
    "scipy>=0.19.1",
    "pandas>=0.24.2",
    "Pillow>=6.2.1",
    "torch>=1.8.1",
    "torchvision>=0.9.1"
]
