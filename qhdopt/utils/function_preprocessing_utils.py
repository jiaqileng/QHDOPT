import numpy as np
from sympy import lambdify, expand, N, symbols


def gen_affine_transformation(scaling_factor, lb):
    return lambda x: scaling_factor * x + lb


def gen_new_func_with_affine_trans(affine_trans, func, syms):
    affine_transformation_vars = list(
        affine_trans(np.array(syms))
    )
    return func.subs(zip(syms, affine_transformation_vars)), syms


def decompose_function(func, syms):
    lambdify_numpy = lambda free_vars, f: lambdify(free_vars, f, "numpy")
    symbol_to_int = {
        syms[i]: i + 1 for i in range(len(syms))
    }
    func_expanded = expand(func)

    # Containers for different parts
    univariate_terms, bivariate_terms = {}, {}

    # Iterate over the terms in the expanded form
    for term in func_expanded.as_ordered_terms():
        # Check the variables in the term
        vars_in_term = term.free_symbols

        # Classify the term based on the number of variables it contains
        if len(vars_in_term) == 1:
            single_var_index = symbol_to_int[list(vars_in_term)[0]]
            univariate_terms.setdefault(single_var_index, []).append(term)
        elif len(vars_in_term) == 2:
            index1, index2 = sorted(
                [symbol_to_int[sym] for sym in list(vars_in_term)]
            )

            factors = term.as_ordered_factors()
            coefficient = 1
            i = 0
            while len(factors[i].free_symbols) == 0:
                coefficient *= float(N(factors[i]))
                i += 1

            reordered_factors = sorted(
                [factors[i] for i in range(i, len(factors))],
                key=lambda factor: symbol_to_int[list(factor.free_symbols)[0]],
            )

            symbol_labels = []
            f = [1, 1]
            sym0id = list(reordered_factors[0].free_symbols)[0]
            for factor in reordered_factors :
                syms = list(factor.free_symbols)
                if len(syms) > 1 :
                    raise Exception(f"Found undecomposable term: {factor}")
                ind = 0 if syms[0] == sym0id else 1
                f[ind] *= factor
                
            f1, f2 = f
            print(f1, f2)
            
            bivariate_terms.setdefault((index1, index2), []).append(
                (
                    coefficient,
                    lambdify_numpy(list(f1.free_symbols), f1),
                    lambdify_numpy(list(f2.free_symbols), f2),
                )
            )
        elif len(vars_in_term) > 2:
            raise Exception(
                f"The specified function has {len(vars_in_term)} variable term "
                f"which is currently unsupported by QHD."
            )

    # Combine the terms to form each part
    univariate_part = {
        var: (1, lambdify_numpy(list(terms[0].free_symbols), sum(terms)))
        for var, terms in univariate_terms.items()
    }

    return univariate_part, bivariate_terms

def quad_to_gen(Q, b):
    x = symbols(f"x:{len(Q)}")
    f = 0
    for i in range(len(Q)):
        qii = Q[i][i]
        bi = b[i]
        f += 0.5 * qii * x[i] * x[i] + bi * x[i]
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            if abs(Q[i][j] - Q[j][i]) > 1e-6:
                raise Exception(
                    "Q matrix is not symmetric."
                )
            f += Q[i][j] * x[i] * x[j]
    return f, list(x)


def generate_bounds(bounds, dimension):
    if bounds is None:
        lb = [0] * dimension
        scaling_factor = [1] * dimension
    elif isinstance(bounds, tuple):
        lb = [bounds[0]] * dimension
        scaling_factor = [bounds[1] - bounds[0]] * dimension
    elif isinstance(bounds, list):
        lb = [bounds[i][0] for i in range(dimension)]
        scaling_factor = [bounds[i][1] - bounds[i][0] for i in range(dimension)]
    else:
        raise Exception(
            "Unsupported bounds type. Try: (lb, ub) or [(lb1, ub1), ..., (lbn, ubn)]."
        )
    return lb, scaling_factor
