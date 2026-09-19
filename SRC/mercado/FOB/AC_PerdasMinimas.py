import pyomo.environ as pyo

def AC_PerdasMinimas(_self):
    perdas_minimas = sum(_self.var_lists['PGER_UTE'][g] for g in range(_self.NUTE))
    custo_deficit = sum(10000 * _self.var_lists['deficit'][b] for b in range(_self.NBAR))
    expr = perdas_minimas + custo_deficit 
    return expr

