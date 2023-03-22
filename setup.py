# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:25:28 2021

@author: lukepinkel
"""

import setuptools
from Cython.Build import cythonize

ext_modules = [setuptools.Extension("pystatsm.utilities.ordered_indices", 
                                    ["pystatsm/utilities/ordered_indices.pyx"]
                                    )
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
    ext_modules=cythonize(ext_modules)
)