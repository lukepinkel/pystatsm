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
        "pystatsm.utilities.cs_kron_wrapper",
        sources=["pystatsm/utilities/cs_kron_wrapper.pyx", 
                 "pystatsm/utilities/cs_kron.c"],
        include_dirs=[np.get_include()]),
    
    setuptools.Extension(
        "pystatsm.utilities.coo_to_csc_wrapper",
        sources=["pystatsm/utilities/coo_to_csc_wrapper.pyx", 
                 "pystatsm/utilities/coo_to_csc.c"],
        include_dirs=[np.get_include()]),
    
    setuptools.Extension(
        "pystatsm.utilities.csc_matmul",
        sources=["pystatsm/utilities/csc_matmul_wrapper.pyx", 
                 "pystatsm/utilities/csc_matmul.c"],
        include_dirs=[np.get_include()]),
    
    setuptools.Extension(
        "pystatsm.utilities.cs_add_inplace_wrapper",
        sources=["pystatsm/utilities/cs_add_inplace_wrapper.pyx", 
                 "pystatsm/utilities/cs_add_inplace.c"],
        include_dirs=[np.get_include()]),
    
    setuptools.Extension(
        "pystatsm.utilities.tile_1d_wrapper",
        sources=["pystatsm/utilities/tile_1d_wrapper.pyx", 
                 "pystatsm/utilities/tile_1d.c"],
        include_dirs=[np.get_include()]),
    
    
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