"""Tests for the basis-free ModelGraph IR (pysem2.model_graph)."""
import pytest

from pystatsm.pysem2.model_graph import (ModelGraph, Parameter, CONSTANT,
                                         LOADING, REGRESSION)


def political_democracy():
    """The Bollen industrialization/democracy model, built basis-free.

    Mirrors FORMULA0 in tests/sem2_test.py:
        ind60 =~ x1 + x2 + x3
        dem60 =~ y1 + y2 + y3 + y4
        dem65 =~ y5 + y6 + y7 + y8
        dem60 ~ ind60
        dem65 ~ ind60 + dem60
        y1 ~~ y5; y2 ~~ y4 + y6; y3 ~~ y7; y4 ~~ y8; y6 ~~ y8
    """
    g = ModelGraph()
    g.measure("ind60", ["x1", "x2", "x3"])
    g.measure("dem60", ["y1", "y2", "y3", "y4"])
    g.measure("dem65", ["y5", "y6", "y7", "y8"])
    g.regress("dem60", ["ind60"])
    g.regress("dem65", ["ind60", "dem60"])
    g.covary("y1", "y5")
    g.covary("y2", "y4")
    g.covary("y2", "y6")
    g.covary("y3", "y7")
    g.covary("y4", "y8")
    g.covary("y6", "y8")
    return g


def test_parameter_free_vs_fixed():
    p = Parameter()
    assert p.free and not p.is_fixed and p.value is None

    p = Parameter(value=1.0)            # supplying a value implies fixed
    assert p.is_fixed and p.value == 1.0

    p = Parameter.fixed(0.0)
    assert p.is_fixed and p.value == 0.0

    p = Parameter.estimated(start=0.5, bounds=(0.0, None))
    assert p.free and p.start == 0.5 and p.lb == 0.0 and p.ub is None

    with pytest.raises(ValueError):
        Parameter(free=True, value=2.0)  # cannot be free and fixed at once


def test_node_and_edge_construction():
    g = political_democracy()
    assert set(g.latent_variables) == {"ind60", "dem60", "dem65"}
    assert set(g.observed_variables) == {f"x{i}" for i in (1, 2, 3)} | \
                                        {f"y{i}" for i in range(1, 9)}
    # CONSTANT is never a registered variable.
    assert CONSTANT not in g.variables
    # Loadings are directed latent -> indicator; regressions are src -> dst.
    pairs = {(s, d) for s, d, _, _ in g.directed_edges()}
    assert ("ind60", "x1") in pairs
    assert ("ind60", "dem60") in pairs


def test_roles_are_derived_not_cached():
    g = political_democracy()
    # Latent vs observed.
    assert g.is_latent("ind60") and g.is_observed("y1")
    # Indicators: observed variables with a latent parent.
    for v in ["x1", "x2", "x3"] + [f"y{i}" for i in range(1, 9)]:
        assert g.is_indicator(v), v
    # ind60 is exogenous (no non-constant parent); dem60/dem65 endogenous.
    assert g.is_exogenous("ind60")
    assert g.is_endogenous("dem60") and g.is_endogenous("dem65")
    # Parents exclude the constant by default.
    assert g.parents("dem65") == ["dem60", "ind60"]
    # Role tag sets compose the primitives.
    assert g.roles("ind60") == {"latent", "exogenous"}
    assert g.roles("dem60") == {"latent", "endogenous"}
    assert g.roles("y1") == {"observed", "endogenous", "indicator"}


def test_intercept_does_not_make_endogenous():
    g = ModelGraph()
    g.measure("f", ["a", "b", "c"])
    g.intercept("a")               # mean structure only
    assert g.has_intercept("a")
    assert g.is_endogenous("a")    # endogenous via the loading, not the mean
    g2 = ModelGraph()
    g2.add_node("z")
    g2.intercept("z")
    assert g2.is_exogenous("z")    # an intercept alone is not a structural cause


def test_equality_classes():
    g = ModelGraph()
    g.measure("f", ["x1", "x2", "x3"], label="lam")   # all three loadings tied
    g.covary("x1", "x2", label="th")
    g.covary("x2", "x3", label="th")
    classes = g.equality_classes()
    assert set(classes) == {"lam", "th"}
    assert len(classes["lam"]) == 3
    assert set(classes["th"]) == {("x1", "x2"), ("x2", "x3")}


def test_validate_rejects_cross_kind_labels():
    g = ModelGraph()
    g.measure("f", ["x1", "x2"], label="shared")
    g.covary("x1", "x2", label="shared")  # tying a loading to a covariance
    with pytest.raises(ValueError):
        g.validate()


def test_validate_rejects_fixed_without_value():
    g = ModelGraph()
    g.add_directed("f", "x", free=False)  # fixed but no value
    with pytest.raises(ValueError):
        g.validate()


def test_copy_is_independent():
    g = political_democracy()
    h = g.copy()
    h.regress("dem65", ["dem65_extra"] if False else [])  # no-op safety
    h.covary("y1", "y2")
    assert ("y1", "y2") not in g._bidirected
    assert ("y1", "y2") in h._bidirected


def test_to_frame_roundtrip_shapes():
    g = political_democracy()
    df = g.to_frame()
    # 3+4+4 loadings + (1+2) regressions + 6 covariances = 11 + 3 + 6 = 20 rows.
    assert len(df) == 20
    assert set(df["op"]) == {"=~", "~", "~~"}
    loadings = df[df["kind"] == LOADING]
    assert (loadings["op"] == "=~").all()


if __name__ == "__main__":
    test_parameter_free_vs_fixed()
    test_node_and_edge_construction()
    test_roles_are_derived_not_cached()
    test_intercept_does_not_make_endogenous()
    test_equality_classes()
    test_validate_rejects_cross_kind_labels()
    test_validate_rejects_fixed_without_value()
    test_copy_is_independent()
    test_to_frame_roundtrip_shapes()
    print("ok")
