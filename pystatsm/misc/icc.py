#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:56:49 2022

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
from ..pylmm.lmm_mod import LMM


def icc_from_summary_stats(subject_var, rater_var, residual_var, 
                           n_subjects, n_raters):
    ss_table = pd.DataFrame([[subject_var], 
                             [rater_var], 
                             [residual_var], 
                             [subject_var + rater_var + residual_var]],
                            index=["id", "items", "resid", "total"], 
                            columns=["variance"])
    ss_table["percent"] = ss_table["variance"] / ss_table.loc["total", "variance"]
    
    stats = np.zeros((5, 3))
    
    msbs = n_raters * subject_var + residual_var
    msbr = n_subjects * rater_var + residual_var
    msws = rater_var + residual_var
    mswr = subject_var + residual_var
    msrs = residual_var
    
    dfbs = n_subjects - 1
    dfbr = n_raters - 1
    dfws = n_subjects * dfbr
    dfwr = n_raters * dfbs
    dfrs = dfbs * dfbr
    
    fbs = msbs / msrs
    fbr = msbr / msrs
    stats[0] = dfbs, dfbr, dfrs
    stats[1] = msbs * dfbs, msbr * dfbr, msrs * dfrs
    stats[2] = msbs, msbr, msrs
    stats[3] = fbs, fbr, np.nan
    stats[4] = sp.stats.f(dfbs, dfrs).sf(fbs), sp.stats.f(dfbr, dfrs).sf(fbr), np.nan
    
    stats_table = pd.DataFrame(stats,columns=["subs", "judges", "resids"],
                               index=["df", "ssq", "ms", "f", "p"])
    icc1 = (msbs - msws) / (msbs + dfbr * msws)
    icc2 = (msbs - msrs) / (msbs + dfbr * msrs + n_raters / n_subjects * (msbr - msrs))
    icc3 = (msbs - msrs) / (msbs + dfbr * msrs)
    icc12 = (msbs - msws) / msbs
    icc22 = (msbs - msrs) / (msbs + (msbr - msrs) / n_subjects)
    icc32 = (msbs - msrs) / msbs
    f11 = msbs / msws
    f21 = msbs / msrs
    icc = np.zeros((4, 6))
    icc[0] = icc1, icc2, icc3, icc12, icc22, icc32
    icc[1] = f11, f21, f21, f11, f21, f21
    icc[2] = dfbs, dfbs, dfbs, dfbs, dfbs, dfbs
    icc[3] = dfws, dfrs, dfrs ,dfws, dfrs, dfrs
    icc_table = pd.DataFrame(icc.T, index=["ICC1", "ICC2", "ICC3", 
                                     "ICC1k", "ICC2k", "ICC3k"],
                             columns=["ICC", "F", "df1", "df2"])
    return stats_table, icc_table, ss_table
    
        
        

class ICC(object):
    
    def __init__(self, values, raters, subjects, data, covariates_formula=None,
                 lmm_fit_kws=None):
        id_ = subjects
        items = raters
        lmm_fit_kws = {} if lmm_fit_kws is None else lmm_fit_kws
        if covariates_formula is None:
            formula = f"{values}~1+(1|{id_})+(1|{items})"
        else:
            formula = f"{values}~1+{covariates_formula}+(1|{id_})+(1|{items})"
        model = LMM(formula, data=data)
        model.fit(**lmm_fit_kws)
        ms_id = model.theta[0]
        ms_it = model.theta[1]
        mse = model.theta[-1]
        ms_df = pd.DataFrame([[ms_id], [ms_it], [mse], [ms_id+ms_it+mse]],
                             index=["id", "items", "resid", "total"], 
                             columns=["variance"])
        ms_df["percent"] = ms_df["variance"] / ms_df.loc["total", "variance"]
        
        nj = model.random_effects.group_sizes[1]
        nobs = model.random_effects.group_sizes[0]
        
        msb = nj * ms_id + mse
        msj = nobs * ms_it + mse
        msw = ms_it + mse
        stats = np.zeros((5, 3))
        dfb = nobs - 1
        dfj = nj - 1
        dfe = dfb * dfj
        fb = msb / mse
        fj = msj / mse
        stats[0] = dfb, dfj, dfe
        stats[1] = msb * dfb, msj * dfj, mse*dfe
        stats[2] = msb, msj, mse
        stats[3] = fb, fj, np.nan
        stats[4] = sp.stats.f(dfb, dfe).sf(fb), sp.stats.f(dfj, dfe).sf(fj), np.nan
        
        stats = pd.DataFrame(stats,columns=["subs", "judges", "resids"],
                             index=["df", "ssq", "ms", "f", "p"])
        icc1 = (msb - msw) / (msb + dfj * msw)
        icc2 = (msb - mse) / (msb + dfj * mse + nj / nobs * (msj - mse))
        icc3 = (msb - mse) / (msb + dfj * mse)
        icc12 = (msb - msw) / msb
        icc22 = (msb - mse) / (msb + (msj - mse) / nobs)
        icc32 = (msb - mse) / msb
        f11 = msb / msw
        f21 = msb / mse
        dfd = nobs * dfj
        icc = np.zeros((4, 6))
        icc[0] = icc1, icc2, icc3, icc12, icc22, icc32
        icc[1] = f11, f21, f21, f11, f21, f21
        icc[2] = dfb, dfb, dfb, dfb, dfb, dfb
        icc[3] = dfd, dfe, dfe ,dfd, dfe, dfe
        icc = pd.DataFrame(icc.T, index=["ICC1", "ICC2", "ICC3", 
                                         "ICC1k", "ICC2k", "ICC3k"],
                           columns=["ICC", "F", "df1", "df2"])
        
        self.res = icc
        self.stats = stats
        self.model = model

        
def icc(subject_var, rater_var, residual_var, n_subjects, n_raters):
    ms_subject = n_raters * subject_var + residual_var
    ms_rater = n_subjects * rater_var + residual_var
    ms_within = rater_var + residual_var
    
    df_subject  = n_subjects - 1
    df_rater    = n_raters - 1
    df_residual = df_subject * df_rater
    f_subject = ms_subject / residual_var
    f_rater = ms_rater / residual_var
    p_subject = sp.stats.f(df_subject, df_residual).sf(f_subject)
    p_rater = sp.stats.f(df_rater, df_residual).sf(f_rater)
    stats = np.zeros((5, 3))
    stats[0] = df_subject, df_rater, df_residual
    stats[1] = ms_subject * df_subject, ms_rater * df_rater, residual_var * df_residual
    stats[2] = f_subject, f_rater, np.nan
    stats[4] = p_subject, p_rater, np.nan
    stats = pd.DataFrame(stats,columns=["subs", "judges", "resids"],
                              index=["df", "ssq", "ms", "f", "p"])
    icc1 = (ms_subject - ms_within) / (ms_subject + df_rater * ms_within)
    icc2 = (ms_subject - residual_var) / (ms_subject + df_rater * residual_var + n_raters / n_subjects * (ms_rater - residual_var))
    icc3 = (ms_subject - residual_var) / (ms_subject + df_rater * residual_var)
    icc12 = (ms_subject - ms_within) / ms_subject
    icc22 = (ms_subject - residual_var) / (ms_subject + (ms_rater - residual_var) / n_subjects)
    icc32 = (ms_subject - residual_var) / ms_subject
    f11 = ms_subject / ms_within
    f21 = ms_subject / residual_var
    dfd = n_subjects * df_rater
    icc = np.zeros((4, 6))
    icc[0] = icc1, icc2, icc3, icc12, icc22, icc32
    icc[1] = f11, f21, f21, f11, f21, f21
    icc[2] = df_subject, df_subject, df_subject, df_subject, df_subject, df_subject
    icc[3] = dfd, df_residual, df_residual ,dfd, df_residual, df_residual
    icc = pd.DataFrame(icc.T, index=["ICC1", "ICC2", "ICC3", 
                                      "ICC1k", "ICC2k", "ICC3k"],
                        columns=["ICC", "F", "df1", "df2"])
    return icc, stats

        