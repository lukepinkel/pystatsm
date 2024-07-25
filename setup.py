# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:25:28 2021

@author: lukepinkel
"""

import setuptools
import numpy as np
from Cython.Build import cythonize

ext_modules = [
    setuptools.Extension(
        "pystatsm.utilities.ordered_indices", 
        sources=["pystatsm/utilities/ordered_indices.pyx"]),
    
    setuptools.Extension(
        "pystatsm.utilities.cython_wrappers",
        sources=["pystatsm/utilities/cython_wrappers.pyx", 
                 "pystatsm/utilities/src/coo_to_csc.c",
                 "pystatsm/utilities/src/cs_add_inplace.c",
                 "pystatsm/utilities/src/cs_add_inplace_complex.c",
                 "pystatsm/utilities/src/cs_kron_ss.c",
                 "pystatsm/utilities/src/cs_kron_ds.c",
                 "pystatsm/utilities/src/cs_kron_sd.c",
                 "pystatsm/utilities/src/cs_dot.c",
                 "pystatsm/utilities/src/cs_pattern_trace.c",
                 "pystatsm/utilities/src/cs_kron_id_sp.c",
                 "pystatsm/utilities/src/cs_matmul_inplace.c",
                 "pystatsm/utilities/src/cs_matmul_inplace_complex.c",
                 "pystatsm/utilities/src/naive_matmul_inplace.c",
                 "pystatsm/utilities/src/repeat_1d.c",
                 "pystatsm/utilities/src/tile_1d.c",
                 "pystatsm/utilities/src/tile_1d_complex.c",
                 ],
        include_dirs=[np.get_include(),
                      "pystatsm/utilities/src/"]),
    ]

setuptools.setup(
    name="pystatsm",
    version="0.1.1",
    url="https://github.com/lukepinkel/pystatsm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.17.2',
        'numba>=0.45.1',
        'scipy>=1.5.3',
        'tqdm>=4.36.1',
        'scikit-sparse>=0.4.4',
        'matplotlib>=3.3',
        'patsy>=0.5.1',
        'pandas>=1.2.1'
        ],
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"})
)