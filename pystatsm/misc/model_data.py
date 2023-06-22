import pandas as pd
import numpy as np

class ModelData(object):
    
    def __init__(self, data=None, sample_cov=None, sample_mean=None,
                 group_vec=None, group_indices=None, n_obs=None, ddof=0):
        if sample_cov is not None:
            if type(sample_cov) is pd.DataFrame:
                sample_cov = {0:sample_cov}
            if type(sample_mean) is pd.DataFrame:
                sample_cov = {0:sample_mean}
        self.data = data
        self.sample_cov = sample_cov
        self.sample_mean = sample_mean
        self.n_obs = n_obs
        self.ddof = ddof
        self.group_indices = group_indices
        if data is not None:
            self.data, self.data_df = self._to_dataframe_and_array(self.data)
        if sample_cov is None and group_vec is not None:
            self.n_groups = len(np.unique(group_vec))
        elif sample_cov is not None:
            self.n_groups = len(sample_cov)
        self.sample_cov_df = {}
        self.sample_mean_df = {}
        self.const = {}
        if self.sample_cov is not None:
            self.sample_cov = self.sample_cov.copy()
            ov_names = self.sample_cov[0].columns
            nov = len(ov_names)
            self.var_order = dict(zip(ov_names, np.arange(nov)))
            for i in range(self.n_groups):
                cov_i, cov_df_i = self._to_dataframe_and_array(self.sample_cov[i].copy())
                self.sample_cov[i], self.sample_cov_df[i] = cov_i, cov_df_i
                lndS = np.linalg.slogdet(self.sample_cov[i])[1]
                self.const[i] = -lndS - self.sample_cov[i].shape[0]
        if self.sample_mean is not None:
            self.sample_mean =  self.sample_mean.copy()
            for i in range(self.n_groups):
                mean_i, mean_df_i = self._to_dataframe_and_array(self.sample_mean[i].copy())
                self.sample_mean[i], self.sample_mean_df[i] = mean_i, mean_df_i
 

    @staticmethod
    def _to_dataframe_and_array(data):
        if isinstance(data, pd.DataFrame):
            arr, df = data.values, data
        elif isinstance(data, pd.Series):
            arr, df = data.values, data
            arr = arr.reshape(1, -1)
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
    
    
    @classmethod
    def from_dataframe(cls, data, group_col=None, ddof=0):
        n_obs = data.shape[0]
        if group_col is not None:
            group_vec = data[group_col]
            group_ids = np.unique(group_vec)
            n_groups = len(group_ids)
            remap = dict(zip(group_ids, np.arange(n_groups)))
            group_vec = group_vec.replace(remap) 
            group_indices = data.groupby(group_vec).indices
            data = data.copy()
            data = data.drop(group_col, axis=1)
        else:
            group_vec = pd.Series(np.zeros(n_obs, dtype=int), index=data.index)
            group_indices = {0:np.arange(data.shape[0])}
            n_groups = 1
            group_ids = np.array([0], dtype=int)
    
        sample_cov, sample_mean, n_obs = {}, {}, {}
        for i in range(n_groups):
            sample_cov[i] = data.iloc[group_indices[i]].cov(ddof=ddof)
            sample_mean[i] = data.iloc[group_indices[i]].mean()
            n_obs[i] = len(group_indices[i])
        return cls(data=data, sample_cov=sample_cov, sample_mean=sample_mean, 
                   group_vec=group_vec, group_indices=group_indices, ddof=ddof, 
                   n_obs=n_obs)
    
    
    
    
    
    
    