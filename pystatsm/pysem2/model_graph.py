"""
Basis-free intermediate representation (IR) for structural equation models.

The :class:`ModelGraph` is a labeled directed/bidirected graph that captures
*what a model says* without committing to any matrix parameterization, variable
ordering, or flat parameter layout.  It is the well-founded object that sits
between a surface syntax (an R-style infix formula, a prefix/S-expression
reader, or a programmatic builder) and a concrete, "lowered" representation
(RAM, reticular all-y, full LISREL, ...).

Design contract
---------------
1. The graph carries **no** notion of matrix, row, column, or flat offset.  All
   of that is produced later by a separate lowering pass ``lower(graph, basis)``.
2. Parameter identity lives on the *edges* of this graph.  A choice of basis may
   permute or relocate parameters in a compiled model, but it never changes the
   edges here, so an ordered basis stops being load-bearing for correctness.
3. Variable *roles* (latent/observed, endogenous/exogenous, indicator, ...) are
   **derived** from graph topology plus a single per-node ``latent`` flag.  They
   are never stored as a collection of overlapping sets that must be kept
   mutually consistent.

The three kinds of model statements all become edges:

==================  ============================  ===========================
statement           meaning                       edge
==================  ============================  ===========================
``f =~ x``          ``x`` loads on latent ``f``    directed ``f -> x``
``y ~ x``           ``y`` regressed on ``x``       directed ``x -> y``
``y ~ 1``           intercept / mean of ``y``      directed ``CONSTANT -> y``
``a ~~ b``          (co)variance of ``a``, ``b``   bidirected ``a -- b``
==================  ============================  ===========================

Loadings, regressions and intercepts share the *same shape* (a directed arrow,
RAM's asymmetric ``A`` together with the mean structure), and (co)variances are
bidirected arrows (RAM's symmetric ``S``).  The measurement/structural (``=~``
vs ``~``) distinction is kept as a semantic edge ``kind`` because topology alone
cannot recover it -- a latent-to-latent arrow may be a higher-order loading
(``z1 =~ z11``) or a structural path (``z2 ~ z1``).  The purely topological roles
(latent/observed, endogenous/exogenous) never consult ``kind``; only
measurement-aware queries such as :meth:`ModelGraph.is_indicator` and layouts
that separate ``Lambda`` from ``Beta`` do.
"""
from __future__ import annotations

import copy as _copy

import pandas as pd

#: Name of the constant ("1") node used to carry the mean structure.  Means and
#: intercepts are directed edges from this node.
CONSTANT = "1"

#: Semantic edge kinds for directed edges.  ``LOADING`` (measurement) and
#: ``REGRESSION`` (structural) are genuinely distinct: a latent-to-latent arrow
#: can be either, so this is not recoverable from topology.  ``INTERCEPT`` marks
#: an edge from the constant node.  The topological roles ignore ``kind``; only
#: measurement-aware queries (:meth:`ModelGraph.is_indicator`) and layouts use it.
LOADING = "loading"
REGRESSION = "regression"
INTERCEPT = "intercept"


class Parameter:
    """
    The parameter attached to a single edge.

    A parameter is either *free* (estimated, with an optional ``start`` value) or
    *fixed* (held at ``value``).  Parameters that share a non-``None`` ``label``
    are constrained to be equal; this is the only equality mechanism, and it is
    expressed over edges rather than over flat positions.

    Parameters
    ----------
    free : bool, optional
        Whether the parameter is estimated.  Defaults to ``True`` unless a
        ``value`` is supplied, in which case it defaults to ``False``.
    value : float, optional
        The fixed value.  Only meaningful when ``free`` is ``False``.
    start : float, optional
        A starting value for estimation.  Only meaningful when ``free`` is
        ``True``.
    label : str, optional
        An equality-class label.  Edges sharing a label share a single estimated
        parameter.
    bounds : tuple of (float or None, float or None), optional
        ``(lower, upper)`` bounds for the estimated parameter.
    """

    __slots__ = ("free", "value", "start", "label", "lb", "ub")

    def __init__(self, free=None, value=None, start=None, label=None, bounds=None):
        if free is None:
            free = value is None
        if free and value is not None:
            raise ValueError("A free parameter cannot also have a fixed value.")
        lb, ub = (None, None) if bounds is None else bounds
        self.free = bool(free)
        self.value = value
        self.start = start
        self.label = label
        self.lb = lb
        self.ub = ub

    @classmethod
    def fixed(cls, value, label=None):
        """Construct a parameter fixed at ``value``."""
        return cls(free=False, value=value, label=label)

    @classmethod
    def estimated(cls, start=None, label=None, bounds=None):
        """Construct a free parameter with an optional ``start`` and ``bounds``."""
        return cls(free=True, start=start, label=label, bounds=bounds)

    @property
    def is_fixed(self):
        """``True`` if the parameter is held at a fixed value."""
        return not self.free

    def copy(self):
        """Return an independent copy of this parameter."""
        return Parameter(free=self.free, value=self.value, start=self.start,
                         label=self.label, bounds=(self.lb, self.ub))

    def __repr__(self):
        if self.free:
            bits = ["free"]
            if self.label is not None:
                bits.append(f"label={self.label!r}")
            if self.start is not None:
                bits.append(f"start={self.start}")
        else:
            bits = [f"={self.value}"]
            if self.label is not None:
                bits.append(f"label={self.label!r}")
        return f"Parameter({', '.join(bits)})"


class ModelGraph:
    """
    A basis-free graph representation of a structural equation model.

    The graph stores nodes (variables plus the implicit :data:`CONSTANT`),
    directed edges (loadings, regressions, intercepts), and bidirected edges
    ((co)variances).  Each edge owns a :class:`Parameter`.  Nodes carry a single
    ``latent`` flag; everything else about a variable's role is computed on
    demand from the graph.

    The builder methods (:meth:`measure`, :meth:`regress`, :meth:`covary`,
    :meth:`variance`, :meth:`intercept`) return ``self`` so that construction can
    be chained.  They are thin wrappers over the primitives
    :meth:`add_directed` and :meth:`add_bidirected`.
    """

    def __init__(self):
        # name -> {"latent": bool}.  The CONSTANT node is implicit and excluded.
        self._nodes = {}
        # (src, dst) -> {"param": Parameter, "kind": provenance tag}
        self._directed = {}
        # (a, b) with a <= b -> {"param": Parameter}
        self._bidirected = {}

    # ------------------------------------------------------------------ nodes
    def add_node(self, name, latent=False):
        """
        Register a variable node, creating it if necessary.

        Parameters
        ----------
        name : str
            The variable name.  May not be :data:`CONSTANT`.
        latent : bool, optional
            Whether the variable is latent.  If the node already exists, a
            ``True`` value promotes it to latent (latent-ness is sticky), but a
            ``False`` value never demotes an existing latent node.
        """
        if name == CONSTANT:
            raise ValueError(f"{CONSTANT!r} is the reserved constant node.")
        node = self._nodes.setdefault(name, {"latent": False})
        if latent:
            node["latent"] = True
        return self

    def _ensure(self, name, latent=False):
        if name != CONSTANT:
            self.add_node(name, latent=latent)
        return name

    # ------------------------------------------------------------ primitives
    def add_directed(self, src, dst, **param_kws):
        """
        Add a directed edge ``src -> dst`` (``dst`` is regressed on ``src``).

        Parameters
        ----------
        src, dst : str
            Source (cause) and destination (effect).  ``src`` may be
            :data:`CONSTANT` for an intercept/mean; ``dst`` may not.
        kind : str, optional
            Provenance tag (one of :data:`LOADING`, :data:`REGRESSION`,
            :data:`INTERCEPT`).  Descriptive only.
        **param_kws
            Forwarded to :class:`Parameter` (``free``, ``value``, ``start``,
            ``label``, ``bounds``).
        """
        kind = param_kws.pop("kind", REGRESSION)
        if dst == CONSTANT:
            raise ValueError("The constant node cannot be a destination.")
        if src == dst:
            raise ValueError(f"Directed self-loop on {src!r} is not allowed.")
        self._ensure(src)
        self._ensure(dst)
        self._directed[(src, dst)] = {"param": Parameter(**param_kws), "kind": kind}
        return self

    def add_bidirected(self, a, b, **param_kws):
        """
        Add a bidirected edge ``a -- b`` (covariance, or variance when ``a == b``).

        Neither endpoint may be :data:`CONSTANT`.  The endpoints are stored in a
        canonical order so that ``a -- b`` and ``b -- a`` are the same edge.

        Parameters
        ----------
        a, b : str
            The two variables.  Equal names denote a variance.
        **param_kws
            Forwarded to :class:`Parameter`.
        """
        if CONSTANT in (a, b):
            raise ValueError("The constant node has no (co)variance edges.")
        self._ensure(a)
        self._ensure(b)
        key = (a, b) if a <= b else (b, a)
        self._bidirected[key] = {"param": Parameter(**param_kws)}
        return self

    # ----------------------------------------------------- statement builders
    def measure(self, latent, indicators, **param_kws):
        """
        Declare a measurement model: ``latent =~ indicators``.

        Marks ``latent`` as a latent node and adds a directed loading edge from
        ``latent`` to each indicator.  Any modifier keywords (``value``,
        ``label``, ``start``, ``bounds``) are applied uniformly to every loading;
        per-indicator modifiers belong in a front-end, not here.
        """
        self.add_node(latent, latent=True)
        for ind in _as_list(indicators):
            self.add_directed(latent, ind, kind=LOADING, **param_kws)
        return self

    def regress(self, dst, srcs, **param_kws):
        """Declare a regression ``dst ~ srcs`` as directed edges ``src -> dst``."""
        for src in _as_list(srcs):
            self.add_directed(src, dst, kind=REGRESSION, **param_kws)
        return self

    def intercept(self, var, **param_kws):
        """Declare an intercept/mean ``var ~ 1`` as ``CONSTANT -> var``."""
        return self.add_directed(CONSTANT, var, kind=INTERCEPT, **param_kws)

    #: ``mean`` is an alias for :meth:`intercept`.
    mean = intercept

    def covary(self, a, b, **param_kws):
        """Declare a covariance ``a ~~ b`` (``a != b``)."""
        if a == b:
            raise ValueError("Use variance() for a == b.")
        return self.add_bidirected(a, b, **param_kws)

    def variance(self, var, **param_kws):
        """Declare a variance ``var ~~ var``."""
        return self.add_bidirected(var, var, **param_kws)

    # --------------------------------------------------------------- accessors
    @property
    def variables(self):
        """Sorted list of variable names (excluding :data:`CONSTANT`)."""
        return sorted(self._nodes)

    @property
    def latent_variables(self):
        """Sorted list of latent variable names."""
        return sorted(n for n, a in self._nodes.items() if a["latent"])

    @property
    def observed_variables(self):
        """Sorted list of observed (non-latent) variable names."""
        return sorted(n for n, a in self._nodes.items() if not a["latent"])

    def directed_edges(self):
        """Iterate ``(src, dst, Parameter, kind)`` over directed edges."""
        for (src, dst), d in self._directed.items():
            yield src, dst, d["param"], d["kind"]

    def bidirected_edges(self):
        """Iterate ``(a, b, Parameter)`` over bidirected edges."""
        for (a, b), d in self._bidirected.items():
            yield a, b, d["param"]

    def is_latent(self, var):
        """``True`` if ``var`` is a latent variable."""
        return self._nodes[var]["latent"]

    def is_observed(self, var):
        """``True`` if ``var`` is an observed (non-latent) variable."""
        return not self._nodes[var]["latent"]

    def parents(self, var, include_constant=False):
        """
        Sources of directed edges into ``var``.

        By default the :data:`CONSTANT` node is excluded so that the result
        reflects only structural/measurement causes (the mean structure does not
        make a variable endogenous).
        """
        out = [s for (s, d) in self._directed if d == var]
        if not include_constant:
            out = [s for s in out if s != CONSTANT]
        return sorted(out)

    def children(self, var):
        """Destinations of directed edges out of ``var``."""
        return sorted(d for (s, d) in self._directed if s == var)

    # -------------------------------------------------------- derived roles
    #
    # These predicates replace the ad-hoc collection of overlapping name sets.
    # Each has a single crisp topological definition; compose them as needed
    # rather than caching new sets.

    def is_endogenous(self, var):
        """
        ``True`` if ``var`` has any incoming directed edge from a non-constant
        node.  In RAM terms an indicator counts as endogenous (it is caused by
        its latent); use :meth:`is_indicator` to distinguish that case.
        """
        return len(self.parents(var)) > 0

    def is_exogenous(self, var):
        """``True`` if ``var`` has no non-constant parents."""
        return not self.is_endogenous(var)

    def is_indicator(self, var):
        """
        ``True`` if ``var`` is the destination of a measurement (loading) edge.

        This is the one role query that consults edge ``kind``: a variable
        regressed on a latent (a structural path) is *not* an indicator even
        though it has a latent parent, so topology alone is insufficient.
        """
        return any(kind == LOADING and dst == var
                   for _, dst, _, kind in self.directed_edges())

    def has_intercept(self, var):
        """``True`` if a directed edge from :data:`CONSTANT` into ``var`` exists."""
        return (CONSTANT, var) in self._directed

    def roles(self, var):
        """
        Return the set of role tags that ``var`` satisfies.

        The tags are derived live from the graph: ``"latent"``/``"observed"``,
        ``"endogenous"``/``"exogenous"``, and (when applicable) ``"indicator"``.
        """
        tags = set()
        tags.add("latent" if self.is_latent(var) else "observed")
        tags.add("endogenous" if self.is_endogenous(var) else "exogenous")
        if self.is_indicator(var):
            tags.add("indicator")
        return tags

    # --------------------------------------------------------- equality classes
    def equality_classes(self):
        """
        Map each equality label to the list of edge keys that carry it.

        A directed edge key is ``(src, dst)``; a bidirected key is ``(a, b)``.
        This is the basis-free description of the model's equality constraints.
        """
        classes = {}
        for (src, dst), d in self._directed.items():
            label = d["param"].label
            if label is not None:
                classes.setdefault(label, []).append((src, dst))
        for (a, b), d in self._bidirected.items():
            label = d["param"].label
            if label is not None:
                classes.setdefault(label, []).append((a, b))
        return classes

    # ----------------------------------------------------------- validation
    def validate(self):
        """
        Check structural well-formedness, raising ``ValueError`` on the first
        problem found.

        This checks only graph-level integrity -- it deliberately says nothing
        about identification or metric setting, which are lowering-time concerns.
        Returns ``self`` so it can be chained.
        """
        # Equality labels must not be shared across the directed/bidirected
        # divide: a covariance cannot be constrained equal to a loading.
        directed_labels = {d["param"].label for d in self._directed.values()
                           if d["param"].label is not None}
        bidirected_labels = {d["param"].label for d in self._bidirected.values()
                             if d["param"].label is not None}
        clash = directed_labels & bidirected_labels
        if clash:
            raise ValueError(
                f"Equality label(s) {sorted(clash)} are shared between directed "
                f"and bidirected edges, which constrain incomparable quantities.")
        # A fixed parameter must actually have a value.
        for key, d in self._directed.items():
            p = d["param"]
            if p.is_fixed and p.value is None:
                raise ValueError(f"Directed edge {key} is fixed but has no value.")
        for key, d in self._bidirected.items():
            p = d["param"]
            if p.is_fixed and p.value is None:
                raise ValueError(f"Bidirected edge {key} is fixed but has no value.")
        return self

    # ------------------------------------------------------------- utilities
    def copy(self):
        """Return a deep, independent copy of the graph."""
        new = ModelGraph()
        new._nodes = _copy.deepcopy(self._nodes)
        new._directed = {k: {"param": v["param"].copy(), "kind": v["kind"]}
                         for k, v in self._directed.items()}
        new._bidirected = {k: {"param": v["param"].copy()}
                           for k, v in self._bidirected.items()}
        return new

    def to_frame(self):
        """
        Render the edges as a tidy :class:`pandas.DataFrame` for inspection.

        The frame is a *view* for debugging and reporting; it is not the source
        of truth and carries no matrix/index information.  Columns mirror the
        familiar lavaan layout (``lhs``, ``op``, ``rhs``) augmented with the
        parameter fields.
        """
        rows = []
        op_for = {LOADING: "=~", REGRESSION: "~", INTERCEPT: "~"}
        for src, dst, p, kind in self.directed_edges():
            if kind == LOADING:
                lhs, rhs = src, dst
            else:  # regression / intercept: dst ~ src (rhs is the predictor)
                lhs, rhs = dst, src
            rows.append(_edge_row(lhs, op_for[kind], rhs, kind, p))
        for a, b, p in self.bidirected_edges():
            rows.append(_edge_row(a, "~~", b, "covariance", p))
        cols = ["lhs", "op", "rhs", "kind", "free", "value", "start", "label",
                "lb", "ub"]
        return pd.DataFrame(rows, columns=cols)

    def __repr__(self):
        return (f"ModelGraph({len(self._nodes)} vars, "
                f"{len(self._directed)} directed, "
                f"{len(self._bidirected)} bidirected)")


def _edge_row(lhs, op, rhs, kind, p):
    return {"lhs": lhs, "op": op, "rhs": rhs, "kind": kind, "free": p.free,
            "value": p.value, "start": p.start, "label": p.label,
            "lb": p.lb, "ub": p.ub}


def _as_list(x):
    if isinstance(x, str):
        return [x]
    return list(x)
