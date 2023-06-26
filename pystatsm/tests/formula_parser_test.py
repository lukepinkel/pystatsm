from pystatsm.pysem.formula_parser import FormulaParser
from pystatsm.pysem.test_data import formula_parser_test_data as test_data


def are_equal_dict_of_sets(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1.keys():
        if dict1[key] != dict2[key]:
            return False
    return True



def test_formula_parser():
    for i in range(21):
        var_names = getattr(test_data, f"var_names{i}")
        param_df = getattr(test_data, f"param_df{i}")
        formula = getattr(test_data, f"formula{i}")
        formula_parser = FormulaParser(formula)
        assert(are_equal_dict_of_sets(formula_parser.var_names, var_names))
        assert(param_df.equals(formula_parser.param_df))
    

