

class ModelMatrixMapper:

    @staticmethod
    def get_matrix_assignments(param_df, var_names):
        lav_names, obs_names = var_names["lav"], var_names["obs"]
        mats = {}
        mes = param_df["rel"] == "=~"
        reg = param_df["rel"] == "~"
        cov = param_df["rel"] == "~~"
        mst = param_df["rhs"] == "1"
        rvl = param_df["rhs"].isin(lav_names)
        rvo = param_df["rhs"].isin(obs_names)
        lvl = param_df["lhs"].isin(lav_names)
        lol = param_df["lhs"].isin(obs_names)
        dmy = param_df["dummy"]
        rvb = rvl & rvo
        mats[0] = (mes & ~rvl) | dmy
        mats[1] = ((mes & rvb) | (mes & ~rvo) | reg) & ~dmy
        mats[2] = (cov & ~rvl) | (cov & lvl)
        mats[3] = cov & ~ lvl
        mats[4] = lol & ~lvl & reg & mst
        mats[5] = lvl & reg & mst
        return mats

    @property
    def mat_assignments(self):
        """
        Returns
        -------
        dict
            dictionary of boolean indexing arrays corresponding to each matrix
        """
        param_df, var_names = self.get_table(), self.var_names
        return self.get_matrix_assignments(param_df, var_names)

    @staticmethod
    def _assign_matrices(param_df, ix):
        param_df["mat"] = 0
        param_df["mat"] = param_df["mat"].astype(int)
        for i in range(6):
            param_df.loc[ix[i], "mat"] = i
            if i == 0:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "rhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
            elif i < 4:
                param_df.loc[ix[i], "r"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "rhs"]
            else:
                param_df.loc[ix[i], "c"] = param_df.loc[ix[i], "lhs"]
                param_df.loc[ix[i], "r"] = 0
            if i == 1:
                j = (param_df["mat"] == 1) & (param_df["rel"] == "=~")
                param_df.loc[j, "r"], param_df.loc[j, "c"] \
                    = param_df.loc[j, "c"], param_df.loc[j, "r"]
        return param_df

    def assign_matrices(self):
        """
        Assigns parameters to matrices and assigns the row and column
        for each parameter.

        Returns
        -------
        None
        """
        param_df, var_names = self.get_table(), self.var_names
        mat_assignments = self.get_matrix_assignments(param_df, var_names)
        self.set_table(self._assign_matrices(param_df, mat_assignments))
