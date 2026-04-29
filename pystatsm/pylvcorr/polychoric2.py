import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from functools import cached_property
from ..utilities import (func_utils, output, param_transforms,
                                         random, linalg_operations, indexing_utils)


class _PairWorkspace:
    """At fixed (i1, i2, r) lazily caches the r-dependent quantities used by
    qml: cell probabilities, dprob/dr, and d2prob/dr2. Lifetime is the caller's
    scope (one Newton iteration); a fresh workspace is built per r."""

    def __init__(self, est, i1, i2, r):
        self.est = est
        self.i1, self.i2 = i1, i2
        self.r = r
        self.vech_ind = est.II_ind[i1, i2]
        self.counts = est.counts[self.vech_ind]

    @cached_property
    def _corners(self):
        ti1 = self.est.tauo[self.i1]
        ti2 = self.est.tauo[self.i2]
        i, j = self.est.indices[self.vech_ind]['prob']
        return ti1[i], ti2[j], ti1[i + 1], ti2[j + 1]

    @cached_property
    def prob(self):
        t1, t2, t3, t4 = self._corners
        return np.maximum(func_utils.binorm_cdf_region((t1, t2), (t3, t4), self.r), 1e-16)

    @cached_property
    def dp_dr(self):
        t1, t2, t3, t4 = self._corners
        return func_utils.binorm_pdf_region((t1, t2), (t3, t4), self.r)

    @cached_property
    def d2p_dr2(self):
        t1, t2, t3, t4 = self._corners
        return func_utils.dbinorm_pdf_region((t1, t2), (t3, t4), self.r)

    def loglike(self):
        return -np.sum(self.counts * np.log(self.prob))

    def score(self):
        return -np.sum((self.counts / self.prob) * self.dp_dr)

    def hessian(self):
        u = self.counts / self.prob
        v = self.counts / np.maximum(self.prob ** 2, 1e-16)
        return np.sum(self.dp_dr ** 2 * v) - np.sum(u * self.d2p_dr2)




class Polychor(object):

    def __init__(self, data):
        self.data = data
        self.cols = self.data.columns.tolist()
        self.X = data.values.astype(int)
        self.X = self.X - np.min(self.X, axis=0)
        self.n, self.p = self.X.shape
        self.N_cats = np.array([len(np.unique(self.X[:, i])) for i in range(self.p)]).astype(int)
        self.transform_list = [param_transforms.OrderedTransform,
                               param_transforms.OrderedTransform,
                               param_transforms.TanhTransform]
        self.tanh_transform = param_transforms.TanhTransform()
        self.process_data()
        self.initialize_interactions()

    def process_data(self):
        self.n, self.p = self.X.shape
        self.uniq = {}
        self.catt = {}
        self.cprp = {}
        self.tauc = {}
        self.tauo = {}
        self.ncat = {}
        for i in range(self.p):
            self.uniq[i], self.catt[i] = np.unique(self.X[:, i], return_counts=True)
            self.cprp[i] = (np.cumsum(self.catt[i])/np.sum(self.catt[i]))[:-1]
            self.tauc[i] = sp.special.ndtri(self.cprp[i])
            self.tauo[i] = np.r_[-1e10, self.tauc[i], 1e10]
        self.ncat = np.array([len(self.catt[i]) for i in range(self.p)])

    def initialize_interactions(self):
        p = self.p
        p2 = int(p * (p - 1) // 2)
        p3 = int(p2 * (p2 + 1) // 2)
        inds1 = indexing_utils.generate_indices((p,)*2, first_indices_change_fastest=False,
                                 ascending=False, strict=True)
        inds2 = indexing_utils.generate_indices((p2,)*2, first_indices_change_fastest=False,
                                 ascending=False, strict=False)
        inds3 = []
        for i in range(p3):
            inds3.append(inds1[inds2[i][0]]+inds1[inds2[i][1]])
        self.inds1, self.inds2, self.inds3 = inds1, inds2, inds3
        counts = {}
        props = {}
        indices = {}
        params = {}
        trns = {}
        for i, (i1, i2) in enumerate(self.inds1):
            counts[i] , props[i], indices[i], params[i], trns[i] = self._xtabs(i1, i2)
        self.counts = counts
        self.props = props
        self.indices = indices
        self.params = params
        self.trns = trns
        self.II_ind = indexing_utils.inv_tril_indices(p, -1)
        self.p2 = p2
        self.p3 = p3

    def _xtabs(self, i1, i2):
        _, counts = sp.stats.contingency.crosstab(self.X[:, i1], self.X[:, i2])
        ni1, ni2 = counts.shape
        mi1, mi2 = ni1 - 1, ni2 - 1
        n_obs = np.sum(counts)
        params = np.r_[self.tauc[i1], self.tauc[i2],
                       np.corrcoef(self.X[:, i1], self.X[:, i2])[0, 1]]
        props = counts / n_obs

        ind_ni1_ni2 = np.meshgrid(np.arange(ni1), np.arange(ni2), indexing='ij')
        ind_mi1_ni2 = np.meshgrid(np.arange(mi1), np.arange(ni2), indexing='ij')
        ind_mi2_ni1 = np.meshgrid(np.arange(mi2), np.arange(ni1), indexing='ij')
        ind_mi1_mi2 = np.meshgrid(np.arange(mi1), np.arange(mi2), indexing='ij')

        indices = {"prob":ind_ni1_ni2,
                    "taui1":ind_mi1_ni2,
                    "taui2":ind_mi2_ni1,
                    "tau_i1i2":ind_mi1_mi2}
        transform = param_transforms.CombinedTransform(self.transform_list,
                                                       [mi1, mi2, 1])
        return counts, props, indices, params, transform

    def unpack_params_full(self, params, i1, i2, padval=1e12):
        ti1 = np.r_[-padval, params[:self.ncat[i1]-1], padval]
        ti2 = np.r_[-padval, params[self.ncat[i1]-1:self.ncat[i1]-1+self.ncat[i2]-1], 1e10]
        ri1i2 = params[-1]
        return ti1, ti2, ri1i2

    def predicted_probs(self, i1, i2, params=None, r=None):
        if params is not None:
           ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        else:
            ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        return p

    def loglike(self, params, i1, i2):
        ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        counts = self.counts[vech_ind]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        p = np.maximum(p, 1e-16)
        ll = -np.sum(counts * np.log(p))
        return ll

    def qloglike(self, r, i1, i2):
        ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        counts = self.counts[vech_ind]
        i, j = self.indices[vech_ind]["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        p = np.maximum(p, 1e-16)
        ll = -np.sum(counts * np.log(p))
        return ll

    def dprob_dparams(self, params, i1, i2, order=1):
        ti1, ti2, r = self.unpack_params_full(params, i1, i2)
        ni1, ni2 = self.ncat[i1], self.ncat[i2]
        mi1, mi2 = ni1 - 1, ni2 - 1
        vech_ind = self.II_ind[i1, i2]
        indices = self.indices[vech_ind]
        dp_dt1 = np.zeros((ni1, ni2, mi1))
        dp_dt2 = np.zeros((ni1, ni2, mi2))

        i, j = indices["prob"]
        p = func_utils.binorm_cdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)
        dp_dr = func_utils.binorm_pdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)

        k, j = indices["taui1"]
        t1, t2, t3 = ti1[k+1], ti2[j], ti2[j+1]
        tmp1 = func_utils.dbinorm_cdf_du(t1, t3, r) - func_utils.dbinorm_cdf_du(t1, t2, r)
        dp_dt1[k, j, k] =  tmp1
        dp_dt1[k+1, j, k] = -tmp1
        k, i = indices["taui2"]

        t1, t2, t3 = ti2[k+1], ti1[i], ti1[i+1]

        tmp1 = func_utils.dbinorm_cdf_du(t1, t3, r) - func_utils.dbinorm_cdf_du(t1, t2, r)
        dp_dt2[i, k, k] =  tmp1
        dp_dt2[i, k+1, k] = -tmp1

        dp_dparams = np.concatenate([dp_dt1, dp_dt2, dp_dr[:, :, None]], axis=-1)

        if order < 2:
            return p, dp_dparams

        d2p_dr_dr = np.zeros((ni1, ni2, 1, 1))
        d2p_dt1_dt1 =  np.zeros((ni1, ni2, mi1, mi1))
        d2p_dt2_dt2 =  np.zeros((ni1, ni2, mi2, mi2))
        d2p_dt1_dt2 =  np.zeros((ni1, ni2, mi1, mi2))
        d2p_dt1_dr  =  np.zeros((ni1, ni2, mi1, 1))
        d2p_dt2_dr  =  np.zeros((ni1, ni2, mi2, 1))

        i, j = indices["prob"]
        d2p_dr_dr[i, j, 0, 0] = func_utils.dbinorm_pdf_region((ti1[i], ti2[j]), (ti1[i+1], ti2[j+1]), r)

        k, j =  indices["taui1"]
        t1, t2, t3 = ti1[k+1], ti2[j], ti2[j+1]
        tmp1 = func_utils.d2binorm_cdf_du2(t1, t3, r) - func_utils.d2binorm_cdf_du2(t1, t2, r)
        tmp2 = func_utils.d2binorm_cdf_dur(t1, t3, r) - func_utils.d2binorm_cdf_dur(t1, t2, r)
        d2p_dt1_dt1[k, j, k, k] = tmp1
        d2p_dt1_dt1[k+1, j, k, k] = -tmp1
        d2p_dt1_dr[k, j, k, 0] = tmp2
        d2p_dt1_dr[k+1, j, k, 0] = -tmp2

        k, i = indices["taui2"]
        t1, t2, t3 = ti2[k+1], ti1[i], ti1[i+1]
        tmp1 = func_utils.d2binorm_cdf_du2(t1, t3, r) - func_utils.d2binorm_cdf_du2(t1, t2, r)
        tmp2 = func_utils.d2binorm_cdf_dur(t1, t3, r) - func_utils.d2binorm_cdf_dur(t1, t2, r)
        d2p_dt2_dt2[i, k, k, k]   = tmp1
        d2p_dt2_dt2[i, k+1, k, k] = -tmp1
        d2p_dt2_dr[i, k, k, 0]    = tmp2
        d2p_dt2_dr[i, k+1, k, 0]  = -tmp2

        k, j = indices["tau_i1i2"]
        t1, t2 = ti1[k+1], ti2[j+1]
        tmp1 = func_utils.d2binorm_cdf_duv(t1, t2, r)
        d2p_dt1_dt2[k, j, k, j] = d2p_dt1_dt2[k+1, j+1, k, j] = tmp1
        d2p_dt1_dt2[k+1, j, k, j] = d2p_dt1_dt2[k, j+1, k, j] = -tmp1

        d2p_dt2_dt1 = np.swapaxes(d2p_dt1_dt2, 2, 3)
        d2p_dr_dt1 = np.swapaxes(d2p_dt1_dr, 2, 3)
        d2p_dr_dt2 = np.swapaxes(d2p_dt2_dr, 2, 3)
        d2p_dparams2 = np.block([[d2p_dt1_dt1, d2p_dt1_dt2,  d2p_dt1_dr],
                        [d2p_dt2_dt1, d2p_dt2_dt2,  d2p_dt2_dr],
                        [d2p_dr_dt1,  d2p_dr_dt2,   d2p_dr_dr]])

        return p, dp_dparams, d2p_dparams2

    def dprob_dr(self, r, i1, i2, order=1):
        ti1, ti2 = self.tauo[i1], self.tauo[i2]
        vech_ind = self.II_ind[i1, i2]
        i, j = self.indices[vech_ind]["prob"]
        t1, t2, t3, t4 = ti1[i], ti2[j], ti1[i+1], ti2[j+1]
        p = func_utils.binorm_cdf_region((t1, t2), (t3, t4), r)
        p = np.maximum(p, 1e-16)
        dp_dr = func_utils.binorm_pdf_region((t1, t2), (t3, t4), r)
        if order<2:
            return p, dp_dr
        d2p_dr_dr = func_utils.dbinorm_pdf_region((t1, t2), (t3, t4), r)
        return p, dp_dr, d2p_dr_dr


    def gradient(self, params, i1, i2):
        prob, dprob = self.dprob_dparams(params, i1, i2, order=1)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        g = -np.einsum("ij,ijl->l", counts / prob, dprob)
        return g


    def qgradient(self, r, i1, i2):
        prob, dprob = self.dprob_dr(r, i1, i2, order=1)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        g = -np.einsum("ij,ij->", counts / prob, dprob)
        return g

    def qhessian(self, r, i1, i2):
        prob, dp, dp2 = self.dprob_dr(r, i1, i2, order=2)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        u = counts / prob
        v = counts / np.maximum(prob**2, 1e-16)
        H = np.einsum("ij,ij,ij->", dp, v, dp) - np.einsum("ij,ij->", u, dp2)
        return H

    def hessian(self, params, i1, i2):
        prob, dp, dp2 = self.dprob_dparams(params, i1, i2, order=2)
        prob = np.maximum(prob, 1e-16)
        counts = self.counts[self.II_ind[i1, i2]]
        u = counts / prob
        v = counts / np.maximum(prob**2, 1e-16)
        H = np.einsum("ijl,ij,ijk->lk",dp, v, dp) - np.einsum("ij,ijkl->kl", u, dp2)
        return H


    def loglike_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params.copy())
        ll = self.loglike(x, i1, i2)
        return ll

    def gradient_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params.copy())
        J = self.trns[vech_ind].jac_rvs(params.copy())
        g = self.gradient(x, i1, i2)
        g = np.dot(g, J)
        return g

    def hessian_transformed(self, params, i1, i2):
        vech_ind = self.II_ind[i1, i2]
        x = self.trns[vech_ind].rvs(params)
        J = self.trns[vech_ind].jac_rvs(params)
        D = self.trns[vech_ind].hess_rvs(params)
        g = self.gradient(x, i1, i2)
        H = self.hessian(x, i1, i2)
        H = J.T.dot(H).dot(J) + np.einsum("ijk,i->jk", D, g)
        return H

    def qloglike_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        ll = np.atleast_1d(self.qloglike(x, i1, i2))
        return ll

    def qgradient_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        J = self.tanh_transform.jac_rvs(r)
        g = self.qgradient(x, i1, i2)
        g = np.reshape(np.dot(g, J), -1)
        return g


    def qhessian_transformed(self, r, i1, i2):
        x = self.tanh_transform.rvs(r)
        J = self.tanh_transform.jac_rvs(r)
        D = self.tanh_transform.hess_rvs(r)
        g = self.qgradient(x, i1, i2)
        H = self.qhessian(x, i1, i2)
        H = np.atleast_2d(H*J**2 + D*g)
        return H

    def _fit_fml(self, i1, i2, opt_kws=None):
        default_opt_kws = dict(method="trust-constr")
        opt_kws = func_utils.handle_default_kws(opt_kws, default_opt_kws)
        func = lambda x: self.loglike_transformed(x, i1, i2)
        grad = lambda x: self.gradient_transformed(x, i1, i2)
        hess = lambda x: self.hessian_transformed(x, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        y = self.trns[vech_ind].fwd(self.params[vech_ind])
        opt_res = sp.optimize.minimize(func, y, jac=grad, hess=hess, **opt_kws)
        param = self.trns[vech_ind].rvs(opt_res.x)
        return opt_res, param

    def _fit_qml(self, i1, i2, opt_kws=None):
        # Legacy scipy trust-constr path; retained for comparison/fallback.
        default_opt_kws = dict(method="trust-constr")
        opt_kws = func_utils.handle_default_kws(opt_kws, default_opt_kws)
        func = lambda x: self.qloglike_transformed(x, i1, i2)
        grad = lambda x: self.qgradient_transformed(x, i1, i2)
        hess = lambda x: self.qhessian_transformed(x, i1, i2)
        vech_ind = self.II_ind[i1, i2]
        y = self.tanh_transform.fwd(self.params[vech_ind][-1])
        opt_res = sp.optimize.minimize(func, y, jac=grad, hess=hess, **opt_kws)
        param = self.tanh_transform.rvs(opt_res.x)
        return opt_res, param

    def _fit_qml_newton(self, i1, i2, r0=None, tol=1e-8, max_iter=50, r_clip=0.9999):
        # Hand-rolled Newton on the 1D score in y = arctanh(r) coordinates.
        # The tanh parameterization keeps r in (-1, 1) by construction and
        # stretches the boundary so Newton remains well-conditioned even when
        # the MLE is near +/- 1. Chain rule:
        #   g_y = g_r (1 - r^2)
        #   H_y = H_r (1 - r^2)^2 - 2 g_r r (1 - r^2)
        # At each step we build a fresh _PairWorkspace at the current r so
        # prob, dp_dr, and d2p_dr2 are computed once per iteration.
        if r0 is None:
            r0 = self.params[self.II_ind[i1, i2]][-1]
        r = float(np.clip(r0, -r_clip, r_clip))
        y = np.arctanh(r)
        ws = _PairWorkspace(self, i1, i2, r)
        ll0 = ws.loglike()
        for it in range(max_iter):
            g_r = ws.score()
            H_r = ws.hessian()
            one_minus_r2 = 1.0 - r * r
            g_y = g_r * one_minus_r2
            H_y = H_r * one_minus_r2 ** 2 - 2.0 * g_r * r * one_minus_r2
            if abs(g_y) < tol:
                break
            if not np.isfinite(H_y) or H_y <= 1e-12:
                H_y = max(abs(g_y), 1e-6)
            step = -g_y / H_y
            alpha = 1.0
            for _ in range(20):
                y_new = y + alpha * step
                r_new = float(np.tanh(y_new))
                ws_new = _PairWorkspace(self, i1, i2, r_new)
                ll_new = ws_new.loglike()
                if ll_new < ll0:
                    y, r, ws, ll0 = y_new, r_new, ws_new, ll_new
                    break
                alpha *= 0.5
            else:
                break  # backtrack exhausted — accept current point
        return r, {'iters': it + 1, 'final_g': float(g_y), 'success': abs(g_y) < tol,
                   'loglike': float(ll0)}

    def _fit(self, i1, i2, method="twostep", opt_kws=None):
        i = self.II_ind[i1, i2]
        if method.lower() in ["two-step", "twostep", "two step", "qml"]:
            r, info = self._fit_qml_newton(i1, i2)
            self.params[i][-1:] = r
            self.opt_res[i] = info
            self._opt[i] = (float(info['success']), info['iters'], info['iters'],
                            0.0, abs(info['final_g']), abs(info['final_g']))
            return
        elif method.lower() in ["one-step", "onestep", "one step", "ml", "fml"]:
            self.opt_res[i], self.params[i] = self._fit_fml(i1, i2, opt_kws)
        t = self.opt_res[i]
        self._opt[i] = (t.success, t.nit, t.nfev, t.execution_time, t.optimality,
                       np.max(np.abs(t.grad)))


    def _get_acm_comp(self, n, xi, xj, xk, xl, gij, gkl, wij, wkl):
        v = np.sum(gij[xi, xj] * gkl[xk, xl]) / n
        v = v - wij * wkl
        return v

    def _build_g_W(self):
        # Per-pair G matrices and W scalars; shared by both get_acov variants.
        A_mats, B_mats, G_mats = {}, {}, {}
        W_arr = np.zeros(self.p2)
        for i in range(self.p):
            ni = self.ncat[i]
            ti = self.tauo[i]
            rind, cind = np.diag_indices(ni - 1)
            Ai = np.zeros((ni, ni - 1))
            phi = func_utils.norm_pdf(ti[1:-1])
            Ai[(rind, cind)] = phi
            Ai[(rind + 1, cind)] = -phi
            p = func_utils.norm_cdf(ti[1:]) - func_utils.norm_cdf(ti[:-1])
            A_mats[i] = Ai
            ADAi = np.dot(Ai.T * 1 / p, Ai)
            Bi = np.linalg.solve(ADAi, Ai.T * 1 / p)
            B_mats[i] = Bi
        for counter, (i1, i2) in enumerate(self.inds1):
            params = self.params[counter]
            counts = self.counts[counter]
            prob, dprob1, _ = self.dprob_dparams(params, i1, i2, order=2)
            mi1, mi2 = self.ncat[[i1, i2]] - 1
            dp_dti1 = dprob1[:, :, :mi1]
            dp_dti2 = dprob1[:, :, mi1:mi1 + mi2]
            dp_dr = dprob1[:, :, -1]
            bi1 = -np.einsum("ij,ijk->k", dp_dr / prob, dp_dti1)
            bi2 = -np.einsum("ij,ijk->k", dp_dr / prob, dp_dti2)
            Bi1, Bi2 = B_mats[i1], B_mats[i2]
            alpha = dp_dr / prob
            D = np.sum((dp_dr ** 2) / prob)
            G = alpha + np.dot(bi1, Bi1).reshape(-1, 1) + np.dot(bi2, Bi2).reshape(1, -1)
            G = G / D
            W_arr[counter] = np.einsum("ij,ij->", G, counts)
            G_mats[counter] = G
        return G_mats, W_arr

    def get_acov(self):
        # Vectorized: build (p2, n) per-observation g matrix, then
        # Acov = ((g_obs @ g_obs.T) / n - outer(W, W)) / n.
        # Matches the inds3 loop form cell-for-cell but collapses the
        # p2*(p2+1)/2 Python iterations into one matmul.
        G_mats, W_arr = self._build_g_W()
        n = self.n
        g_obs = np.empty((self.p2, n))
        for k in range(self.p2):
            i1, i2 = self.inds1[k]
            g_obs[k] = G_mats[k][self.X[:, i1], self.X[:, i2]]
        Acov = np.matmul(g_obs, g_obs.T) / n - np.outer(W_arr, W_arr)
        return Acov / n  # match the asymptotic 1/n scaling

    def get_acov_loop(self):
        # Original quadruple-Python-loop version, kept for verification.
        G_mats, W_arr = self._build_g_W()
        Acov = np.zeros((self.p2, self.p2))
        for counter, (a, b, c, d) in enumerate(self.inds3):
            ab, cd = self.inds2[counter]
            Acov[ab, cd] = self._get_acm_comp(self.n, self.X[:, a], self.X[:, b],
                                              self.X[:, c], self.X[:, d],
                                              G_mats[ab], G_mats[cd],
                                              W_arr[ab], W_arr[cd])
            Acov[cd, ab] = Acov[ab, cd]
        return Acov / self.n


    def fit(self, method="twostep", opt_kws=None):
        self.opt_res = {}
        self._opt = np.zeros((self.p2, 6))
        for ii in range(self.p2):
            self._fit(*self.inds1[ii], method=method, opt_kws=opt_kws)
        self.opt_df = pd.DataFrame(self._opt, columns=["success", "nit", "nfev","time" ,"optimality", "maxabsgrad"])

        self.rho_hat = np.array([self.params[i][-1] for i in range(len(self.params))])
        self.acov = self.get_acov()
        self.rho_hat = self.rho_hat
        self.rho_se = np.sqrt(np.diag(self.acov))
        self.rho_labels = [f"r({self.cols[i]}, {self.cols[j]})" for i, j in self.inds1]
        self.degfrees = np.array([self.n-self.params[i].shape[0] for i in range(self.p2)])
        self.R_hat = linalg_operations.invecl(self.rho_hat)
        self.R_df = pd.DataFrame(self.R_hat, columns=self.data.columns, index=self.data.columns)

        self.res =  output.get_param_table(self.rho_hat,self.rho_se, self.data.shape[0], self.rho_labels)




def threshold_dict_to_array(t_dict):
    p = len(t_dict)
    nt_extended = np.zeros(p, dtype=int)
    nmax = max([len(x) for x in t_dict.values()])
    t_arr = np.zeros((p, nmax))
    for i, (key, val) in enumerate(t_dict.items()):
        nt_extended[i] = len(val)
        t_arr[i, :nt_extended[i]] = val
    return t_arr



def convert_acm(acm, p):
    """
    Primarily for testing purposes for converting a matrix indexed using
    descending co-lexicographic order to one with ascending
    lexicographic order.

    Args:
        acm (numpy.ndarray): Input matrix of size p * (p - 1) // 2 by p * (p - 1) // 2.
            This is the matrix to be converted.
        p (int): Size of the correlation matrix whose lower half below diagonal elements
                 the acm is the covariance matrix for

    Returns:
        numpy.ndarray: The converted matrix with indices in the new order.

    """
    p1 = p
    p2 = p1 * (p1 - 1) // 2

    ix1 = indexing_utils.generate_indices((p1,)*2, first_indices_change_fastest=False, ascending=False, strict=True)
    ix2 = indexing_utils.generate_indices((p2,)*2, first_indices_change_fastest=False, ascending=False, strict=False)
    ix1, ix2 = np.array(ix1), np.array(ix2)

    iy1 = indexing_utils.generate_indices((p1,)*2, first_indices_change_fastest=False, ascending=True, strict=True)
    iy2 = indexing_utils.generate_indices((p2,)*2, first_indices_change_fastest=False, ascending=True, strict=False)
    iy1, iy2 = np.array(iy1), np.array(iy2)

    ij = ix1[ix2[:, 0]]
    kl = ix1[ix2[:, 1]]
    a, b = np.sort(ij, axis=1).T
    c, d = np.sort(kl, axis=1).T

    ab_to_ii = np.zeros((p1, p1), dtype=int)
    ab_to_ii[(iy1[:, 0], iy1[:, 1])] = ab_to_ii[(iy1[:, 1], iy1[:,0])] = np.arange(p2)

    ab, cd = np.sort(np.vstack([ab_to_ii[(a, b)], ab_to_ii[(c, d)]]), axis=0)
    (a, b), (c, d)  = iy1[ab].T, iy1[cd].T
    abcd = np.argsort(np.lexsort(np.vstack((a, b, c, d))))
    acm_flat =  acm[np.tril_indices(len(acm))]
    acm2 = np.zeros_like(acm)
    acm2[ix2[:, 0], ix2[:, 1]] = acm2[ix2[:, 1], ix2[:, 0]] = acm_flat[abcd]
    return acm2


def random_thresholds(nvar, ncat=None, min_cat=None, max_cat=None, rng=None):
    rng = np.random.default_rng(123) if rng is None else rng
    if ncat is None:
        if type(min_cat) in [int, float]:
            min_cat = np.repeat(min_cat, nvar)
        if type(max_cat) in [int, float]:
            max_cat = np.repeat(max_cat, nvar)
        ncat = np.zeros(nvar, dtype=int)
        for i in range(nvar):
            ncat[i] = rng.choice(np.arange(min_cat[i]+2, max_cat[i]+2), 1)
    elif type(ncat) in [int, float]:
        ncat = np.repeat(ncat, nvar)
    elif type(ncat) is np.ndarray:
        ncat = ncat
    quantiles = {}
    taus = {}
    for i in range(nvar):
        q = np.r_[0, rng.dirichlet(20*np.ones(ncat[i]-2)).cumsum()]
        q[-1] = 1.0
        quantiles[i] = q
        taus[i] = sp.special.ndtri(q)
    return quantiles,taus


class PolycoricSim(object):

    def __init__(self, R, taus, rng=None):
        self.rng = np.random.default_rng(123) if rng is None else rng
        self.R = R
        self.taus = taus
        self.p = R.shape[0]
        self.cat_sizes = [len(self.taus[i]) for i in range(len(self.taus))]

    def simulate_data(self, size=1000, exact=False):
        Z = self.rng.multivariate_normal(mean=np.zeros(self.p), cov=self.R, size=size)
        Y = np.zeros_like(Z)
        for i in range(self.p):
            Y[..., i] = np.digitize(Z[..., i], self.taus[i])
        return Z, Y



# for k1 in [True, False]:
#     for k2 in [True, False]:

#         print(f"First Fastest:{k1}; ascending:{k2}",
#               indexing_utils.generate_indices((4,4), first_indices_change_fastest=k1,
#                                               ascending=k2, strict=True))
# import time
# import tqdm
# import seaborn as sns
# import matplotlib.pyplot as plt

# rng = np.random.default_rng(1234)
# n_vars = 6
# n_cats = rng.choice(np.arange(4, 8), n_vars)
# R = random.r_lkj(eta=1.0, n=1, dim=n_vars, rng=rng).squeeze()
# qs, taus = random_thresholds(n_vars, n_cats)

# sim = PolycoricSim(R, taus, rng=rng)


# Z, Y = sim.simulate_data()
# df = pd.DataFrame(Y, columns=[f"x{i}" for i in range(1, n_vars+1)])
# start = time.time()
# mod = Polychor(data=df)
# init_dur = time.time() - start
# mod.fit(method="onestep")
# fit_dur = time.time() - init_dur - start

# n_sim = 10_000

# pbar = tqdm.tqdm(total=n_sim, smoothing=0.001)
# n_r = mod.rho_hat.shape[0]
# sim_arr = np.zeros((n_sim,n_r*2))
# rho_sim = linalg_operations.vecl(sim.R)
# mods = {}
# for i in range(n_sim):
#     Z, Y = sim.simulate_data(size=200)
#     df = pd.DataFrame(Y, columns=[f"x{i}" for i in range(1, n_vars+1)])
#     mods[i] = Polychor(data=df)
#     mods[i].fit(method="onestep")
#     pbar.update(1)
# pbar.close()

# rho_hats = np.stack([mods[i].rho_hat for i in range(n_sim)])
# acm_hats = np.stack([mods[i].acov for i in range(n_sim)])
# acm_mean = np.mean(acm_hats, axis=0)
# acm_medn = np.median(acm_hats, axis=0)
# rho_devs = pd.DataFrame(rho_hats - rho_sim)

# rho_cov = rho_devs.cov()



# g = sns.PairGrid(rho_devs, diag_sharey=False)
# g.map_upper(sns.regplot, scatter_kws=dict(s=3,zorder=0, alpha=0.5), line_kws=dict(zorder=10, color="C1"))
# g.map_diag(sns.histplot, kde=True, bins=25)
# g.axes[0,0].axvline(0, color='k')
# g.axes[1,1].axvline(0, color='k')
# g.axes[2,2].axvline(0, color='k')

# g.axes[0,0].axvline(rho_devs[0].mean(), color='red', ls='--', alpha=0.5)
# g.axes[1,1].axvline(rho_devs[1].mean(), color='red', ls='--', alpha=0.5)
# g.axes[2,2].axvline(rho_devs[2].mean(), color='red', ls='--', alpha=0.5)
# g.map_lower(sns.kdeplot, fill=True, thresh=0, levels=100)



# rho_hats = pd.DataFrame(rho_hats)

# desc = rho_hats.describe(percentiles=[0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95])
# desc.loc["Skew"] = rho_hats.skew()
# desc.loc["Kurtosis"] = rho_hats.kurtosis()


# rho_hats = pd.DataFrame(rho_hats)
# dev =  (rho_hats-rho_sim)
# desc = (rho_hats-rho_sim).describe(percentiles=[0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95])
# desc.loc["Skew"] = rho_hats.skew()
# desc.loc["Kurtosis"] = rho_hats.kurtosis()
# desc=desc.T






