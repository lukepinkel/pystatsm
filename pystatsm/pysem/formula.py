import re
import pandas as pd
import numpy as np
from ..utilities.indexing_utils import  tril_indices

def _default_sort_key(item):
        match = re.match(r"([a-zA-Z]+)(\d+)", item)
        if match:
            alphabetic_part = match.group(1)
            numeric_part = int(match.group(2))
        else:
            alphabetic_part = item
            numeric_part = 0
        return (alphabetic_part, numeric_part)    


class ModelSpecification:
    matrix_names = ["L", "B", "F", "P"]
    matrix_order = dict(L=0, B=1, F=2, P=3)
    is_symmetric = dict(L=False, B=False, F=True, P=True)
    def __init__(self, formulas, var_order=None, extension_kws=None):
        formulas = re.sub(r'\s*#.*', '', formulas)
        self.var_order = var_order
        self.parameters= []
        self.equations = formulas.strip().split('\n')
        for eq in self.equations:
            if eq.strip():
                self.unpack_equation(eq)
        self.ptable = pd.DataFrame(self.parameters)
        self.ptable["rel"] = self.ptable["rel"].astype(str)
        self.classify_parameters()
        extension_kws = {} if extension_kws is None else extension_kws
        self.extend_model(**extension_kws)
        self.assign_matrices()
        self.sort_table(self.lv_order, self.ov_order)
        self.ptable["free"] = 0
        ix = ~self.ptable["fixed"]
        ix2 = ~self.ptable["label"].isnull()
        links = {}
        eqc = np.unique(self.ptable.loc[ix2, "label"] )
        for c in eqc:
            ixc = self.ptable["label"]==c
            index = self.ptable.loc[ixc, "label"].index.values
            i = np.min(index)
            links[c] = i, index
            ix2[i] = False
        ix = ix & ~ix2    
        n = len(self.ptable[ix])
        self.ptable.loc[ix, "free"] = np.arange(1, 1+n)
        for c in eqc:
            i, index = links[c]
            self.ptable.loc[index, "free"] = self.ptable.loc[i, "free"]
        self.make_model_mats()
        
        
    @staticmethod  
    def get_var_pair(ls, rs, rel):
        comps = rs.strip().split('*')
        ls = ls.strip()
        if len(comps) > 1:
            name = comps[1].strip()
            mod = comps[0].strip()
            try:
                fixedval = float(mod)
                fixed = True
                label = None
            except ValueError:
                fixedval = None
                fixed = False
                label = mod
        else:
            mod = None
            fixed = False
            name = comps[0].strip()
            label= None#f"{ls}{rel}{name}"
            fixedval = None
        row = {"lhs":ls,"rel":rel ,"rhs":name, "mod":mod,
               "label":label, "fixedval":fixedval, 
               "fixed":fixed}
        return row
           
    def unpack_equation(self, eq):
        if "=~" in eq:
            rel = "=~"
        elif "~~" in eq:
            rel = "~~"
        else:
            rel = "~"
        lhss, rhss = eq.split(rel)
        for ls in lhss.split('+'):
            for rs in rhss.split('+'):
                row = self.get_var_pair(ls, rs, rel)
                self.parameters.append(row)
                
    def map_rc(self, df, rmap, cmap):
        df["r"] = df["r"].map(rmap)
        df["c"] = df["c"].map(cmap)
        return df
    
    def sort_symmetric(self, df):
        df = df.sort_values(["c", "r"])
        ix = df["r"] < df["c"]
        df.loc[ix, "r"], df.loc[ix, "c"] = df.loc[ix, "c"], df.loc[ix, "r"]
        df = df.sort_values(["c", "r"])
        return df
    
    def sort_nonsymmetric(self, df):
        df = df.sort_values(["c", "r"])
        return df

    def sort_table(self, lv_order, ov_order):
        ptable = self.ptable
        L =  ptable.loc[ptable["mat"]==0]
        B =  ptable.loc[ptable["mat"]==1]
        F =  ptable.loc[ptable["mat"]==2]
        P =  ptable.loc[ptable["mat"]==3]
        with pd.option_context('mode.chained_assignment', None):
            L = self.map_rc(L, ov_order, lv_order)
            B = self.map_rc(B, lv_order, lv_order)
            F = self.map_rc(F, lv_order, lv_order)
            P = self.map_rc(P, ov_order, ov_order)

        L = self.sort_nonsymmetric(L)
        B = self.sort_nonsymmetric(B)
        F = self.sort_symmetric(F)
        P = self.sort_symmetric(P)
        
        ptable = pd.concat([L, B, F, P], axis=0)
        self.ptable = ptable.reset_index(drop=True)
        
    def classify_parameters(self):
        ptable = self.ptable
        lv_names = set(ptable.loc[ptable["rel"]=="=~", "lhs"])
        v_names = set(ptable.loc[ptable["rel"]=="=~", "rhs"])
        y_names = set(ptable.loc[ptable["rel"]=="~", "lhs"])
        x_names = set(ptable.loc[ptable["rel"]=="~", "rhs"])
        
        ov_ind_names = v_names.difference(lv_names)
        ov_y_names = y_names.difference(lv_names.union(v_names))
        ov_x_names = x_names.difference(lv_names.union(v_names).union(ov_y_names))
        ov_cov_names = set(ptable.loc[(ptable["rel"]=="~~") & ~(ptable["lhs"].isin(lv_names)), "lhs"])
        ov_cov_names = ov_cov_names.union(set(ptable.loc[(ptable["rel"]=="~~") & ~(ptable["rhs"].isin(lv_names)), "rhs"]))
        ov_names = set.union(ov_ind_names, ov_y_names, ov_x_names, ov_cov_names)
        if self.var_order is None:
            ov_ordered = sorted(ov_names, key=_default_sort_key)
            self.ov_order = dict(zip(ov_ordered, np.arange(len(ov_names))))
        else:
            ov_ordered = sorted(ov_names, key=lambda x:self.var_order[x])
            self.ov_order = dict(zip(ov_ordered, np.arange(len(ov_names))))
        
        ov_y_names2 = y_names.difference(set.union(v_names, x_names, lv_names))
        ov_nx_names = ov_names.difference(ov_x_names)
        lv_x_names = lv_names.difference(set.union(v_names, y_names))
        lv_y_names = set.intersection(y_names, lv_names).difference(set.union(v_names, x_names))

        lvov_y_names = set.union(lv_y_names, ov_y_names2)
        
        self.names = dict(lv=lv_names,v=v_names, y=y_names, x=x_names, 
                          ov=ov_names, lvov_y=lvov_y_names)
        self.ov_names = dict(ind=ov_ind_names,y=ov_y_names, x=ov_x_names,
                             cov=ov_cov_names, nx=ov_nx_names, y2=ov_y_names2)
        self.lv_names = dict(x=lv_x_names,y=lv_y_names, nx=lv_names.difference(lv_x_names))
    
    def add_covs(self, var_names, lhs, rhs, cov=True):
        n = len(var_names)
        var_names = list(var_names)
        if cov:
            for i, j in list(zip(*tril_indices(n, k=-1))):
                l, r = var_names[i], var_names[j]
                lhs.append(l)
                rhs.append(r)
        else:
            lhs.extend(var_names)
            rhs.extend(var_names)
        return lhs, rhs
    
    def extend_model(self, auto_var=True, auto_lvx_cov=True, auto_y_cov=True,
                     fix_lv_var=True, fix_first=True, fixed_x=True):
        lv_names = self.names["lv"]
        lvx_names = self.lv_names["x"]
        ovx_names = self.ov_names["x"] 
        lvovy_names = self.names["lvov_y"]
        
        lv_names = sorted(self.names["lv"], key=_default_sort_key)
        ix = self.ptable["rel"]=="~"
        reg_names = np.unique(self.ptable.loc[ix, ["lhs", "rhs"]]).tolist()
        ovr_names = list(set(reg_names).difference(lv_names))
        ovr_names = sorted(ovr_names, key=lambda x: self.ov_order[x])
        lv_names_extended = lv_names + ovr_names # analysis:ignore
        self.names["lv_extended"] = lv_names_extended
        self.n_lv = len(lv_names_extended)
        self.n_ov = len(self.names["ov"])
        lhs, rhs = [], []
        if auto_var:
            n1 = set(np.unique(self.ptable["lhs"]))
            n2 = set(np.unique(self.ptable["rhs"]))
            names = (n1.union(n2))#.difference(set.union(lvx_names, lvovy_names))
            names = list(names)
            lhs, rhs = self.add_covs(names, lhs, rhs, cov=False)
            names = list(ovx_names)
            lhs, rhs = self.add_covs(names, lhs, rhs, cov=True)
        if auto_lvx_cov:
            lhs, rhs = self.add_covs(lvx_names, lhs, rhs)
        if auto_y_cov:
            lhs, rhs = self.add_covs(lvovy_names, lhs, rhs)  
        
        ptable = self.ptable.to_dict(orient="records")
        for v1, v2 in list(zip(lhs, rhs)): 
            row = dict(lhs=v1, rel="~~", rhs=v2, mod=None, fixed=False)
            ptable.append(row)
        ptable = pd.DataFrame(ptable)
        if fix_lv_var:
            ix = ((ptable["rel"]=="~~") &
                 (ptable["lhs"].isin(lv_names)) & 
                 (ptable["lhs"]==ptable["rhs"]))
            ptable.loc[ix, "fixed"] = True
            ptable.loc[ix, "fixedval"] = 1.0
        if fix_first:
            ind1 = (ptable["rel"]=="=~") & (ptable["lhs"].isin(lv_names))
            ltable = ptable[ind1]
            ltable.groupby("lhs")
            for v in lv_names:
                ix = ltable["lhs"]==v
                if len(ltable.index[ix])>0:
                    if ~np.any(ltable.loc[ix, "fixed"]):
                        ptable.loc[ltable.index[ix][0], "fixed"] = True
                        ptable.loc[ltable.index[ix][0], "fixedval"] = 1.0
        if fixed_x:
            ix = (ptable["lhs"].isin(ovx_names) &
                  ptable["rhs"].isin(ovx_names) &
                  (ptable["rel"]=="~~"))
            ptable.loc[ix, "fixed"] = True
            ptable.loc[ix, "fixedval"] = 0
            ptable.loc[ix & (ptable["rhs"]==ptable["lhs"]), "fixedval"] = 1
        self.ptable = ptable


    def assign_matrices(self):
        ptable = self.ptable
        ov_names =  sorted(self.names["ov"], key=lambda x: self.ov_order[x])
        lv_names = sorted(self.names["lv"], key=_default_sort_key)
        ix = ptable["rel"]=="~"
        reg_names = np.unique(ptable.loc[ix, ["lhs", "rhs"]]).tolist()
        ovr_names = list(set(reg_names).difference(lv_names))
        ovr_names = sorted(ovr_names, key=lambda x: self.ov_order[x])
        lv_names = lv_names + ovr_names
        lv_names = sorted(lv_names, key=_default_sort_key)
        self.lv_order = dict(zip(lv_names, np.arange(len(lv_names))))
        mes = ptable["rel"]== "=~"
        reg = ptable["rel"]== "~"
        cov = ptable["rel"]== "~~"      
        
        ix = {}
        rvl = ptable["rhs"].isin(lv_names)
        rvo = ptable["rhs"].isin(ov_names)
        lvl = ptable["lhs"].isin(lv_names)
        rvb = rvl & rvo
        ix[0] = mes & ~rvl
        ix[1] = (mes & rvb) | (mes & ~rvo) | reg
        ix[2] = (cov & ~rvl) | (cov & lvl)
        ix[3] = cov & ~lvl
        ptable.loc[ix[0], "mat"] = 0
        ptable.loc[ix[1], "mat"] = 1
        ptable.loc[ix[2], "mat"] = 2
        ptable.loc[ix[3], "mat"] = 3
        ptable["mat"] = ptable["mat"].astype(int)
        ptable.loc[ptable["mat"]==0, "r"] =  ptable.loc[ptable["mat"]==0, "rhs"]
        ptable.loc[ptable["mat"]==0, "c"] =  ptable.loc[ptable["mat"]==0, "lhs"]
        ptable.loc[ptable["mat"]==1, "r"] =  ptable.loc[ptable["mat"]==1, "lhs"]
        ptable.loc[ptable["mat"]==1, "c"] =  ptable.loc[ptable["mat"]==1, "rhs"]
        ix = (ptable["mat"]==1) & (ptable["rel"]=="=~")
        ptable.loc[ix, "r"], ptable.loc[ix, "c"] = ptable.loc[ix, "c"],  ptable.loc[ix, "r"]
        ptable.loc[ptable["mat"]==2, "r"] =  ptable.loc[ptable["mat"]==2, "lhs"]
        ptable.loc[ptable["mat"]==2, "c"] =  ptable.loc[ptable["mat"]==2, "rhs"]
        ptable.loc[ptable["mat"]==3, "r"] =  ptable.loc[ptable["mat"]==3, "lhs"]
        ptable.loc[ptable["mat"]==3, "c"] =  ptable.loc[ptable["mat"]==3, "rhs"]
        self.ptable = ptable
        
    def make_model_mats(self):
        ptable = self.ptable
        n_lv, n_ov = self.n_lv, self.n_ov

        lv_names = sorted(self.lv_order.keys(), key=lambda x: self.lv_order[x])
        ov_names = sorted(self.ov_order.keys(), key=lambda x: self.ov_order[x])
        mat_dims = {0:(n_ov, n_lv), 1:(n_lv, n_lv), 2:(n_lv, n_lv), 3:(n_ov, n_ov)}
        mat_rows = {0:ov_names, 1:lv_names, 2:lv_names, 3:ov_names}
        mat_cols = {0:lv_names, 1:lv_names, 2:lv_names, 3:ov_names}
        
        fixed_mats, free_mats = {}, {}
        for i in range(4):
            subtable =  ptable.loc[ptable["mat"]==i]
            free = subtable.loc[~subtable["fixed"]]
            free_mat = np.zeros(mat_dims[i])
            free_mat[(free["r"], free["c"])] = free["free"]
            
            free_mats[i] = pd.DataFrame(free_mat, index=mat_rows[i], columns=mat_cols[i])
            
            fixed = subtable.loc[subtable["fixed"]]
            fixed_mat = np.zeros(mat_dims[i])
            fixed_mat[(fixed["r"], fixed["c"])] = fixed["fixedval"]
            
            fixed_mats[i] = pd.DataFrame(fixed_mat, index=mat_rows[i], columns=mat_cols[i])
        lv_ov = set(self.names["lv_extended"]).difference(set(self.names["lv"]))
        for v in lv_ov:
            if (v in fixed_mats[0].index) and  (v in fixed_mats[0].columns):
                fixed_mats[0].loc[v, v] = 1.0
        self.free_mats = free_mats
        self.fixed_mats = fixed_mats
        

    
     

# model1 = """  
#   # measurement model
#     ind60 =~ x1 + x2 + x3
#     dem60 =~ y1 + y2 + y3 + y4
#     dem65 =~ y5 + y6 + y7 + y8
#   # regressions
#     dem60 ~ ind60
#     dem65 ~ ind60 + dem60
#   # residual correlations
#     y1 ~~ y5
#     y2 ~~ y4 + y6
#     y3 ~~ y7
#     y4 ~~ y8
#     y6 ~~ y8
# """

# model2 =  """
#   # latent variables
#     ses     =~ education + sei
#     alien67 =~ anomia67 + powerless67
#     alien71 =~ anomia71 + powerless71
#   # regressions
#     alien71 ~ alien67 + ses
#     alien67 ~ ses
#   # correlated residuals
#     anomia67 ~~ anomia71
#     powerless67 ~~ powerless71
# """
# model3 = """
#  visual  =~ x1 + x2 + x3 
#               textual =~ x4 + x5 + x6
#               speed   =~ x7 + x8 + x9 
#               """

# model4 = """ # direct effect
#              Y ~ c*X
#            # mediator
#              M ~ a*X
#              Y ~ b*M
#          """
# model5 = """

#     z11 =~ 1*x111 + x112 + x113 + b4 * x114
#     z12 =~ 1*x121 + x122 + x123 + b4 * x124
#     z21 =~ 1*x211 + x212 + x213 + b4 * x214
#     z22 =~ 1*x221 + x222 + x223 + b4 * x224
#     z31 =~ 1*x311 + x312 + x313 + b4 * x314
#     z32 =~ 1*x321 + x322 + x323 + b4 * x324
    
#     z1 =~ 1*z11 + z12
#     z2 =~ 1*z21 + z22
#     z3 =~ 1*z31 + z32

# """
# model6 = """
# y1 ~ x1 + x2 + x3 + x4 + x5
# y2 ~ x1 + x2 + x3 + x4 + x5
# y3 ~ x1 + x2 + x3 + x4 + x5

# """

# model7 ="""
#     z11 =~ 1*x111 + x112 + x113 + b4 * x114
#     z12 =~ 1*x121 + x122 + x123 + b4 * x124
#     z21 =~ 1*x211 + x212 + x213 + b4 * x214
#     z22 =~ 1*x221 + x222 + x223 + b4 * x224
#     z31 =~ 1*x311 + x312 + x313 + b4 * x314
#     z32 =~ 1*x321 + x322 + x323 + b4 * x324
    
#     z1 =~ 1*z11 + z12
#     z2 =~ 1*z21 + z22
#     z3 =~ 1*z31 + z32

#     z2 ~ z1 + x1
#     z3 ~ z1 + z2 + x1
    
#     y1 ~ x2 + x3 + x4
#     y2 ~ x5 + z3
# """

# m1 = ModelSpecification(model1,)
# m2 = ModelSpecification(model2,)
# m3 = ModelSpecification(model3,)
# m4 = ModelSpecification(model4,)
# m5 = ModelSpecification(model5,)
# m6 = ModelSpecification(model6,)
# m7 = ModelSpecification(model7,)

#m = ModelSpecification(model, var_order = dict(zip(["y1","y2","y3","y4","y5","y6","y7","y8","x1","x2","x3"], np.arange(11))))
