"""Cross-validation of the complexdesign estimators against R survey.

Not a test: requires R with the survey package.  Run as
`PYTHONPATH=. python pystatsm/tests/complexdesign_rcheck.py`; it rebuilds
the exact frame used by complexdesign_test.py, computes every estimator on
both sides (linearization, JKn, a shared-replicate-weight bootstrap, fpc,
domains, svyby groups, gaussian and binomial svyglm), prints the max
absolute deviations, and emits the R numbers formatted for pasting into the
golden-value tests.  The bootstrap comparison feeds R the same replicate
weights via svrepdesign, so it checks the variance assembly algebra rather
than RNG agreement.
"""
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd

from pystatsm.tests.complexdesign_test import make_survey_frame, N_PSU_PER_STR
from pystatsm.utilities.complexdesign.sample_design import SampleDesign
from pystatsm.utilities.complexdesign.repweights import (
    bootstrap_multipliers, replicate_variance)
from pystatsm.utilities.complexdesign.descriptives import (
    survey_total, survey_mean, survey_mean_by)
from pystatsm.utilities.complexdesign.survey_glm import SurveyGLM
from pystatsm.pyglm2.families import Gaussian, Binomial

N_BOOT = 100
FPC_POP_PSU = np.array([20, 15, 10, 8, 25, 12, 6, 40])
GAUSS_LABELS = ["Intercept", "C(x2)[T.1]", "C(x2)[T.2]", "x1"]
R_NAMES = {"Intercept": "X.Intercept.", "x1": "x1",
           "C(x2)[T.1]": "as.factor.x2.1", "C(x2)[T.2]": "as.factor.x2.2"}

R_CODE = """
suppressMessages(library(survey))
df <- read.csv("%(frame)s")
rw <- as.matrix(read.csv("%(repw)s"))
des <- svydesign(id=~psu, strata=~strata, weights=~w, data=df, nest=TRUE)
out <- data.frame(key=character(0), value=numeric(0))
add <- function(out, keys, vals) rbind(out, data.frame(key=keys, value=as.numeric(vals)))

tot <- svytotal(~yg+x1, des)
out <- add(out, c("total_yg", "total_x1"), coef(tot))
out <- add(out, c("total_se_yg", "total_se_x1"), SE(tot))
mn <- svymean(~yg+x1, des)
out <- add(out, c("mean_yg", "mean_x1"), coef(mn))
out <- add(out, c("mean_se_yg", "mean_se_x1"), SE(mn))

dsub <- subset(des, dom==1)
mdom <- svymean(~yg, dsub)
out <- add(out, c("dommean_yg", "dommean_se_yg"), c(coef(mdom), SE(mdom)))
out <- add(out, "degf_dom", degf(dsub))

by <- svyby(~yg, ~x2, des, svymean)
out <- add(out, paste0("by_mean_", by$x2), by$yg)
out <- add(out, paste0("by_se_", by$x2), by$se)

mg <- svyglm(yg ~ x1 + as.factor(x2), design=des)
cf <- summary(mg)$coefficients
out <- add(out, paste0("gg_coef_", make.names(rownames(cf))), cf[, 1])
out <- add(out, paste0("gg_se_", make.names(rownames(cf))), cf[, 2])
out <- add(out, "gg_df_resid", mg$df.residual)
out <- add(out, "degf_des", degf(des))

mb <- svyglm(yb ~ x1 + as.factor(x2), design=des, family=quasibinomial())
cf <- summary(mb)$coefficients
out <- add(out, paste0("gb_coef_", make.names(rownames(cf))), cf[, 1])
out <- add(out, paste0("gb_se_", make.names(rownames(cf))), cf[, 2])

rdes <- as.svrepdesign(des, type="JKn")
mnj <- svymean(~yg, rdes)
out <- add(out, "jkn_mean_se_yg", SE(mnj))
mgj <- svyglm(yg ~ x1 + as.factor(x2), design=rdes)
cf <- summary(mgj)$coefficients
out <- add(out, paste0("ggjk_se_", make.names(rownames(cf))), cf[, 2])

bdes <- svrepdesign(variables=df, repweights=rw, weights=df$w,
                    type="bootstrap", scale=1/%(n_boot)d,
                    rscales=rep(1, %(n_boot)d), combined.weights=TRUE)
mnb <- svymean(~yg, bdes)
totb <- svytotal(~yg, bdes)
out <- add(out, c("boot_mean_se_yg", "boot_total_se_yg"),
           c(SE(mnb), SE(totb)))
mgb <- svyglm(yg ~ x1 + as.factor(x2), design=bdes)
cf <- summary(mgb)$coefficients
out <- add(out, paste0("ggbt_se_", make.names(rownames(cf))), cf[, 2])

desf <- svydesign(id=~psu, strata=~strata, weights=~w, data=df, nest=TRUE,
                  fpc=~fpcN)
totf <- svytotal(~yg, desf)
out <- add(out, "fpc_total_se_yg", SE(totf))

write.csv(out, "%(out)s", row.names=FALSE)
cat(as.character(packageVersion("survey")), "\\n")
"""


def run_r(workdir, df, w_rep):
    frame = os.path.join(workdir, "frame.csv")
    repw = os.path.join(workdir, "repw.csv")
    outp = os.path.join(workdir, "rvals.csv")
    rpath = os.path.join(workdir, "check.R")
    df.to_csv(frame, index=False)
    pd.DataFrame(w_rep.T).to_csv(repw, index=False)
    with open(rpath, "w") as fh:
        fh.write(R_CODE % dict(frame=frame, repw=repw, out=outp,
                               n_boot=N_BOOT))
    proc = subprocess.run(["Rscript", rpath], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{proc.stderr}")
    print(f"R survey version: {proc.stdout.strip()}")
    tab = pd.read_csv(outp)
    return dict(zip(tab["key"], tab["value"]))


def python_side(des, des_fpc, mult):
    out = {}
    rt, Vt = survey_total(des, ["yg", "x1"])
    rm, Vm = survey_mean(des, ["yg", "x1"])
    out["total"] = (rt["total"].to_numpy(), rt["SE"].to_numpy())
    out["mean"] = (rm["mean"].to_numpy(), rm["SE"].to_numpy())
    sub = des.subset(des.df["dom"].to_numpy() == 1)
    rd, _ = survey_mean(sub, "yg")
    out["dommean"] = (rd["mean"].to_numpy()[0], rd["SE"].to_numpy()[0])
    out["degf_dom"] = sub.degf()
    rby = survey_mean_by(des, "yg", "x2")
    out["by"] = (rby["mean"].to_numpy(), rby["SE"].to_numpy())
    mg = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian).fit()
    out["gg"] = (mg.params, mg.params_se, mg.param_labels)
    out["gg_df_resid"] = mg.ddf
    out["degf_des"] = des.degf()
    mb = SurveyGLM("yb ~ x1 + C(x2)", des, family=Binomial).fit()
    out["gb"] = (mb.params, mb.params_se, mb.param_labels)
    rj, Vj = survey_mean(des, "yg", vcov="jackknife")
    out["jkn_mean_se"] = rj["SE"].to_numpy()[0]
    mgj = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian)
    mgj.fit(vcov="jackknife", progress=False)
    out["ggjk_se"] = mgj.params_se
    # bootstrap with the exported multipliers, assembled by hand so the R
    # comparison uses byte-identical replicate weights
    starts = des.layout.ind_psu[:-1]
    x = des.df["yg"].to_numpy()
    tx = np.add.reduceat(des.w * x, starts)
    tw = np.add.reduceat(des.w, starts)
    tot_rep = mult.dot(tx)
    mean_rep = tot_rep / mult.dot(tw)
    out["boot_total_se"] = np.sqrt(replicate_variance(
        tot_rep, np.ones(N_BOOT), 1.0 / N_BOOT))[0, 0]
    out["boot_mean_se"] = np.sqrt(replicate_variance(
        mean_rep, np.ones(N_BOOT), 1.0 / N_BOOT))[0, 0]
    mgb = SurveyGLM("yg ~ x1 + C(x2)", des, family=Gaussian)
    mgb.fit(vcov="bootstrap", n_rep=N_BOOT, rng=np.random.default_rng(42),
            progress=False)
    out["ggbt_se"] = mgb.params_se
    rtf, _ = survey_total(des_fpc, "yg")
    out["fpc_total_se"] = rtf["SE"].to_numpy()[0]
    return out


def _rvec(rv, prefix, labels):
    return np.array([rv[f"{prefix}_{R_NAMES[k]}"] for k in labels])


def compare(py, rv):
    rows = []

    def rel(name, a, b):
        a, b = np.atleast_1d(np.asarray(a, np.float64)), \
            np.atleast_1d(np.asarray(b, np.float64))
        rows.append((name, np.max(np.abs(a - b) / np.maximum(np.abs(b),
                                                             1e-12))))

    rel("total est", py["total"][0], [rv["total_yg"], rv["total_x1"]])
    rel("total SE", py["total"][1], [rv["total_se_yg"], rv["total_se_x1"]])
    rel("mean est", py["mean"][0], [rv["mean_yg"], rv["mean_x1"]])
    rel("mean SE", py["mean"][1], [rv["mean_se_yg"], rv["mean_se_x1"]])
    rel("domain mean", py["dommean"][0], rv["dommean_yg"])
    rel("domain mean SE", py["dommean"][1], rv["dommean_se_yg"])
    rel("domain degf", py["degf_dom"], rv["degf_dom"])
    rel("by-group means", py["by"][0], [rv[f"by_mean_{k}"] for k in range(3)])
    rel("by-group SEs", py["by"][1], [rv[f"by_se_{k}"] for k in range(3)])
    labels = py["gg"][2]
    rel("svyglm gauss coef", py["gg"][0], _rvec(rv, "gg_coef", labels))
    rel("svyglm gauss SE", py["gg"][1], _rvec(rv, "gg_se", labels))
    rel("svyglm gauss df_resid", py["gg_df_resid"], rv["gg_df_resid"])
    rel("degf(design)", py["degf_des"], rv["degf_des"])
    rel("svyglm binom coef", py["gb"][0], _rvec(rv, "gb_coef", labels))
    rel("svyglm binom SE", py["gb"][1], _rvec(rv, "gb_se", labels))
    rel("JKn mean SE", py["jkn_mean_se"], rv["jkn_mean_se_yg"])
    rel("JKn svyglm SE", py["ggjk_se"], _rvec(rv, "ggjk_se", labels))
    rel("boot mean SE (shared w)", py["boot_mean_se"], rv["boot_mean_se_yg"])
    rel("boot total SE (shared w)", py["boot_total_se"],
        rv["boot_total_se_yg"])
    rel("boot svyglm SE (shared w)", py["ggbt_se"],
        _rvec(rv, "ggbt_se", labels))
    rel("fpc total SE", py["fpc_total_se"], rv["fpc_total_se_yg"])
    width = max(len(r[0]) for r in rows)
    print("\nmax relative deviation vs R survey")
    for name, val in rows:
        print(f"  {name:<{width}}  {val:.3e}")
    return rows


def emit_goldens(rv, labels):
    def fmt(vals):
        return "np.array([" + ", ".join(f"{v:.12g}" for v in vals) + "])"

    print("\n# golden values for complexdesign_test.py "
          "(R survey, complexdesign_rcheck.py)")
    print("R_TOTAL_EST =", fmt([rv["total_yg"], rv["total_x1"]]))
    print("R_TOTAL_SE =", fmt([rv["total_se_yg"], rv["total_se_x1"]]))
    print("R_MEAN_EST =", fmt([rv["mean_yg"], rv["mean_x1"]]))
    print("R_MEAN_SE =", fmt([rv["mean_se_yg"], rv["mean_se_x1"]]))
    print("R_DOMAIN_MEAN =", fmt([rv["dommean_yg"], rv["dommean_se_yg"]]))
    print("R_BY_MEAN =", fmt([rv[f"by_mean_{k}"] for k in range(3)]))
    print("R_BY_SE =", fmt([rv[f"by_se_{k}"] for k in range(3)]))
    print("R_GLM_GAUSS_COEF =", fmt(_rvec(rv, "gg_coef", labels)))
    print("R_GLM_GAUSS_SE =", fmt(_rvec(rv, "gg_se", labels)))
    print("R_GLM_BINOM_COEF =", fmt(_rvec(rv, "gb_coef", labels)))
    print("R_GLM_BINOM_SE =", fmt(_rvec(rv, "gb_se", labels)))
    print("R_GLM_GAUSS_JKN_SE =", fmt(_rvec(rv, "ggjk_se", labels)))
    print("R_MEAN_JKN_SE =", fmt([rv["jkn_mean_se_yg"]]))
    print("R_FPC_TOTAL_SE =", fmt([rv["fpc_total_se_yg"]]))
    print("R_BOOT_MEAN_SE =", fmt([rv["boot_mean_se_yg"]]))
    print("R_BOOT_TOTAL_SE =", fmt([rv["boot_total_se_yg"]]))
    print("R_GLM_DF_RESID =", rv["gg_df_resid"])


def main():
    df = make_survey_frame(1234)
    des = SampleDesign(df, "strata", "psu", "w")
    fpc = N_PSU_PER_STR / FPC_POP_PSU
    des_fpc = SampleDesign(df, "strata", "psu", "w", fpc=fpc)
    df_out = des.df.copy()
    df_out["fpcN"] = FPC_POP_PSU[df_out["strata"].to_numpy()]
    mult = bootstrap_multipliers(des.layout, N_BOOT,
                                 rng=np.random.default_rng(42))
    w_rep = des.replicate_weights(mult)
    py = python_side(des, des_fpc, mult)
    workdir = tempfile.mkdtemp(prefix="complexdesign_rcheck_")
    rv = run_r(workdir, df_out, w_rep)
    compare(py, rv)
    emit_goldens(rv, py["gg"][2])


if __name__ == "__main__":
    main()
