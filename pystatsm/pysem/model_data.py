#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 05:23:39 2023

@author: lukepinkel
"""
import pandas as pd
import numpy as np

class ModelData(object):
    
    def __init__(self, data=None, sample_cov=None, sample_mean=None, 
                 n_obs=None, ddof=0):
        self.data = data
        self.sample_cov = sample_cov
        self.sample_mean = sample_mean
        self.n_obs = n_obs
        self.ddof = ddof

        if self.data is not None:
            self.data, self.data_df = self._to_dataframe_and_array(self.data)
            self.sample_cov = self.data_df.cov(ddof=self.ddof)
            self.sample_mean = self.data.mean(axis=0).reshape(1, -1)
            self.n_obs = self.data_df.shape[0]
            self.data_centered =  self.data - np.mean(self.data, axis=0)
        if self.sample_cov is not None:
            self.sample_cov, self.sample_cov_df = self._to_dataframe_and_array(self.sample_cov)
        
        if self.sample_mean is not None:
            self.sample_mean, self.sample_mean_df = self._to_dataframe_and_array(self.sample_mean)
        self.lndS = np.linalg.slogdet(self.sample_cov)[1]
        self.const = -self.lndS - self.sample_cov.shape[0]

    @staticmethod
    def _to_dataframe_and_array(data):
        if isinstance(data, pd.DataFrame):
            arr, df = data.values, data
        elif isinstance(data, np.ndarray):
            columns = [f"x{i}" for i in range(1, data.shape[1]+1)]
            arr, df = data, pd.DataFrame(data, columns=columns)
        return arr, df
    
    @staticmethod
    def augmented_covariance(sample_cov, sample_mean):
        p = sample_cov.shape[0]
        augmented_cov = np.zeros((p+1, p+1))
        augmented_cov[:p, :p] = sample_cov + np.outer(sample_mean, sample_mean)
        augmented_cov[:p, -1] = augmented_cov[-1, :p]   =  sample_mean    
        augmented_cov[-1, -1] = 1.0
        return augmented_cov
        