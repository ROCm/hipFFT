# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from rocm_docs import ROCmDocs


os.system("sed -e 's/HIPFFT_EXPORT //g' ../library/include/hipfft.h > .doxygen/hipfft.h")

docs_core = ROCmDocs("hipFFT Documentation")
docs_core.run_doxygen()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
