import numpy as np
import pandas as pd
from pystatsm.pysem2.formula import FormulaParser
from pystatsm.pysem2.model_builder import ModelBuilder
from pystatsm.pysem2.param_table import ParameterTable
from pystatsm.pysem2.sem import SEM
from pystatsm.pysem2.model_simulator import SimulatedSEM
from pystatsm.utilities.numerical_derivs import fo_fc_cs, jac_cs, so_gc_cd

FORMULA0 = """  
  # measurement model
    ind60 =~ x1 + x2 + x3
    dem60 =~ y1 + y2 + y3 + y4
    dem65 =~ y5 + y6 + y7 + y8
  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60
  # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"""

FORMULA1 = """
   z11 =~ 1 * x111 +   x112 +   x113 + b4 * x114
    z12 =~ 1 * x121 +   x122 +   x123 + b4 * x124

    z21 =~ 1 * x211 +   x212 +   x213 + b4 * x214
    z22 =~ 1 * x221 +   x222 +   x223 + b4 * x224

    z31 =~ 1 * x311 +   x312 +   x313 + b4 * x314
    z32 =~ 1 * x321 +   x322 +   x323 + b4 * x324

    z1 =~ 1 * z11 +   z12
    z2 =~ 1 * z21 +   z22
    z3 =~ 1 * z31 +   z32

    z2 ~ z1 + x1 + b2 * x2
    z3 ~ z1 + z2 + x1 + b2 *x2 + x3

    y1 ~ x4 + x5 + x6
    y2 ~ z2 + x4 + x7

    x111 ~ 0*1
    x112 ~ 1
    x113 ~ 1
    x114 ~ 1

    x121 ~ 0*1
    x122 ~ 1
    x123 ~ 1
    x124 ~ 1


    x211 ~ 0*1
    x212 ~ 1
    x213 ~ 1
    x214 ~ 1

    x221 ~ 0*1
    x222 ~ 1
    x223 ~ 1
    x224 ~ 1


    x311 ~ 0*1
    x312 ~ 1
    x313 ~ 1
    x314 ~ 1

    x321 ~ 0*1
    x322 ~ 1
    x323 ~ 1
    x324 ~ 1

    z11 ~ 0*1
    z12 ~ 1
    z21 ~ 0*1
    z22 ~ 1
    z31 ~ 0*1
    z32 ~ 1
    z1 ~ 1
    z2 ~ 1
    z3 ~ 1
"""

FORMULA2 = """
    z11 =~ 1 * x111 +   x112 +   b3 * x113 + b4 * x114
    z12 =~ 1 * x121 +   x122 +        x123 + b4 * x124

    z21 =~ 1 * x211 +   x212 +        x213 + b4 * x214
    z22 =~ 1 * x221 +   x222 +        x223 + b4 * x224

    z31 =~ 1 * x311 +   x312 +        x313 + b4 * x314
    z32 =~ 1 * x321 +   x322 +   b3 * x323 + b4 * x324

    z1 =~ 1 * z11 +   b4 * z12
    z2 =~ 1 * z21 +   b4 * z22
    z3 =~ 1 * z31 +   b3 * z32

    z2 ~ z1 + b3 * x1 + b2 * x2
    z3 ~ z1 + z2 + x1 + b2 *x2 + x3

    y1 ~ x4 + x5 + x6
    y2 ~ z2 + b3 * x4 + x7

    x111 ~ 0*1
    x112 ~ 1
    x113 ~ 1
    x114 ~ 1

    x121 ~ 0*1
    x122 ~ 1
    x123 ~ 1
    x124 ~ 1


    x211 ~ 0*1
    x212 ~ 1
    x213 ~ 1
    x214 ~ 1

    x221 ~ 0*1
    x222 ~ 1
    x223 ~ 1
    x224 ~ 1


    x311 ~ 0*1
    x312 ~ 1
    x313 ~ 1
    x314 ~ 1

    x321 ~ 0*1
    x322 ~ 1
    x323 ~ 1
    x324 ~ 1

    z11 ~ 0*1
    z12 ~ 1
    z21 ~ 0*1
    z22 ~ 1
    z31 ~ 0*1
    z32 ~ 1
    z1 ~ 1
    z2 ~ 1
    z3 ~ 1


   z11 ~~ c1 * z21 + c2 * z31
   z12 ~~ c1 * z22 + c2 * z32
   z13 ~~ c1 * z23 + c2 * z33
"""


class ModelBuildTester(ModelBuilder):

    def __init__(self, formulas):
        self.formula_parser = FormulaParser(formulas)
        self.var_names = self.formula_parser.var_names
        self.param_df = self.formula_parser.param_df.copy()

    def reset_param_df(self):
        self.param_df = self.formula_parser.param_df.copy()

    def test_variance_addition(self):
        self.reset_param_df()
        self.add_variances(fix_lv_cov=False)
        missing_var = self.check_missing_variances(self.var_names["all"])
        mask = (
                (self.param_df["lhs"] == self.param_df["rhs"]) &
                self.param_df["lhs"].isin(self.var_names["nob"])
        )
        res = []
        res.append(len(missing_var) == 0)
        res.append(~np.all(self.param_df.loc[mask, "fixed"]))
        self.reset_param_df()
        self.add_variances(fix_lv_cov=True)
        missing_var = self.check_missing_variances(self.var_names["all"])
        mask = (
                (self.param_df["lhs"] == self.param_df["rhs"]) &
                self.param_df["lhs"].isin(self.var_names["nob"])
        )
        res.append(len(missing_var) == 0)
        res.append(np.all(self.param_df.loc[mask, "fixed"]))
        self.reset_param_df()
        return res

    def test_covariance_addition(self):
        res = []
        self.reset_param_df()
        lox_vars0 = self.check_missing_covs(self.var_names["lox"])
        lvx_vars0 = self.check_missing_covs(self.var_names["lvx"])
        end_vars0 = self.check_missing_covs(self.var_names["enx"])

        self.add_covariances(lvx_cov=True, y_cov=True)
        lox_vars = self.check_missing_covs(self.var_names["lox"])
        lvx_vars = self.check_missing_covs(self.var_names["lvx"])
        end_vars = self.check_missing_covs(self.var_names["enx"])
        if len(lox_vars0) > 0:
            res.append(len(lox_vars) == 0)
        else:
            res.append(True)
        if len(lvx_vars0) > 0:
            res.append(len(lvx_vars) == 0)
        else:
            res.append(True)
        if len(end_vars0) > 0:
            res.append(len(end_vars) == 0)
        else:
            res.append(True)

        self.reset_param_df()
        self.add_covariances(lvx_cov=False, y_cov=True)
        lox_vars = self.check_missing_covs(self.var_names["lox"])
        lvx_vars = self.check_missing_covs(self.var_names["lvx"])
        end_vars = self.check_missing_covs(self.var_names["enx"])
        if len(lox_vars0) > 0:
            res.append(len(lox_vars) == 0)
        else:
            res.append(True)
        if len(lvx_vars0) > 0:
            res.append(len(lvx_vars) == 0)
        else:
            res.append(True)
        if len(end_vars0) > 0:
            res.append(len(end_vars) == 0)
        else:
            res.append(True)

        self.reset_param_df()
        self.add_covariances(lvx_cov=False, y_cov=False)
        lox_vars = self.check_missing_covs(self.var_names["lox"])
        lvx_vars = self.check_missing_covs(self.var_names["lvx"])
        end_vars = self.check_missing_covs(self.var_names["enx"])
        if len(lox_vars0) > 0:
            res.append(len(lox_vars) == 0)
        else:
            res.append(True)
        if len(lvx_vars0) > 0:
            res.append(len(lvx_vars) == 0)
        else:
            res.append(True)
        if len(end_vars0) > 0:
            res.append(len(end_vars) == len(end_vars0))
        else:
            res.append(True)
        self.reset_param_df()
        return res

    def test_fix_first(self):
        self.reset_param_df()
        self.fix_first()
        res = []
        var_names, param_df = self.var_names, self.param_df
        ind1 = (param_df["rel"] == "=~") & \
               (param_df["lhs"].isin(var_names["nob"]))
        ltable = param_df.loc[ind1]
        for v in var_names["nob"]:
            ix = ltable["lhs"] == v
            if len(ltable.index[ix]) > 0:
                res.append(np.any(ltable.loc[ix, "fixed"]))
        return res

    def test_add_means(self):
        res = []

        self.reset_param_df()
        vars_to_add0 = self.check_missing_means(self.var_names["all"])
        self.add_means(fix_lv_mean=False)
        vars_to_add = self.check_missing_means(self.var_names["all"])
        res.append(len(vars_to_add) == 0)
        ix = (
                (self.param_df["rel"] == "~") & (self.param_df["rhs"] == "1") &
                self.param_df["lhs"].isin(vars_to_add0 & self.var_names["nob"])
        )
        if np.any(ix):
            fixed = self.param_df.loc[ix, "fixed"]
            res.append(~np.all(fixed))
        else:
            res.append(True)

        self.reset_param_df()
        self.add_means(fix_lv_mean=True)
        vars_to_add = self.check_missing_means(self.var_names["all"])
        res.append(len(vars_to_add) == 0)
        ix = (self.param_df["rel"] == "~") & (self.param_df["rhs"] == "1")
        fixed = self.param_df.loc[ix & self.param_df["lhs"].isin(vars_to_add0 & self.var_names["nob"]), "fixed"]
        res.append(np.all(fixed))
        return res

    def test_latent_dummies(self):
        self.reset_param_df()
        self.add_latent_dummies()
        res = self.check_latent_dummies(self.var_names["lvo"])
        return res


def test_model_builder():
    model_builder = ModelBuildTester(FORMULA0)
    assert (all(model_builder.test_variance_addition()))
    assert (all(model_builder.test_covariance_addition()))
    assert (all(model_builder.test_add_means()))
    assert (all(model_builder.test_fix_first()))
    assert (all(model_builder.test_latent_dummies()))

    model_builder = ModelBuildTester(FORMULA1)
    assert (all(model_builder.test_variance_addition()))
    assert (all(model_builder.test_covariance_addition()))
    assert (all(model_builder.test_add_means()))
    assert (all(model_builder.test_fix_first()))
    assert (all(model_builder.test_latent_dummies()))

    model_builder = ModelBuildTester(FORMULA2)
    assert (all(model_builder.test_variance_addition()))
    assert (all(model_builder.test_covariance_addition()))
    assert (all(model_builder.test_add_means()))
    assert (all(model_builder.test_fix_first()))
    assert (all(model_builder.test_latent_dummies()))
    
    
def test_param_table():
    parameter_table = ParameterTable(FORMULA0)
    table = parameter_table.get_table()
    assert (np.all(table.loc[table["fixed"], "ind"] == 0))

    parameter_table = ParameterTable(FORMULA1)
    table = parameter_table.get_table()
    assert (np.all(table.loc[table["fixed"], "ind"] == 0))

    parameter_table = ParameterTable(FORMULA2)
    table = parameter_table.get_table()
    assert (np.all(table.loc[table["fixed"], "ind"] == 0))
    
def test_sem():
    sim = SimulatedSEM(FORMULA2, 5)
    data = sim.simulate()
    model = SEM(FORMULA2, data, group_col="group", group_kws=dict(shared=[True, False, True, False, False, False]))
    model.fit(minimize_kws=dict(method="tnc"), minimize_options=dict(disp=100))
    x = model.theta.copy() * 1.2 + 0.02
    model._check_complex = True
    grad_exact = model.gradient(x)
    grad_approx = fo_fc_cs(model.fit_func, x)
    # np.abs(grad_exact-grad_approx).max()
    assert (np.allclose(grad_exact, grad_approx, atol=1e-4, rtol=1e-4))

    grad_exact = model.gradient(x, per_group=True)
    grad_approx = jac_cs(lambda x: model.fit_func(x, per_group=True), x)
    assert (np.allclose(grad_exact, grad_approx, atol=1e-4, rtol=1e-4))

    hess_exact = model.hessian(x)
    hess_approx = so_gc_cd(model.gradient, x)
    assert (np.allclose(hess_exact, hess_approx, atol=1e-4, rtol=1e-4))
    # np.abs(hess_exact-hess_approx).max()

    sim = SimulatedSEM(FORMULA0, 1)
    data = sim.simulate()
    model = SEM(FORMULA0, data)
    x = model.theta.copy() * 1.2 + 0.02
    model._check_complex = True
    dsigma_exact = model.dsigma_mu(x)
    dsigma_approx = jac_cs(lambda x: model.implied_sample_stats(x)[0], x)
    # np.abs(dsigma_exact[0] - dsigma_approx).max()
    assert (np.allclose(dsigma_exact, dsigma_approx, atol=1e-4, rtol=1e-4))
