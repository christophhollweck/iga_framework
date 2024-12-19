import typing
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import splinepy
from Bernstein_polynomials import eval_BS


# define alias type
Vector = typing.List['float']
ElementVector = typing.List['Element']

def _evaluate_univariate_b_spline(x: float, knots: typing.Union[Vector, np.ndarray], degree: int,
                                  endpoint=True, r=0) -> float:
    """
    Evaluates a univariate BSpline corresponding to the given knot vector and polynomial degree at the point x.

    :param endpoint:
    :param x: point of evaluation
    :param knots: knot vector
    :param degree: polynomial degree
    :param r: derivative
    :return: B(x)
    """
    knots = np.array(knots)
    i = _find_knot_interval(x, knots, endpoint=endpoint)
    if i == -1:
        return 0
    t = _augment_knots(knots, degree)
    i += degree + 1

    c = np.zeros(len(t) - degree - 1)
    c[degree + 1] = 1
    c = c[i - degree: i + 1]

    for k in range(degree, degree - r, -1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]

        c = np.divide((c[1:] - c[:-1]), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

    for k in range(degree - r, 0, -1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]
        omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

        a = np.multiply((1 - omega), c[:-1])
        b = np.multiply(omega, c[1:])
        c = a + b

    return factorial(degree) * c.squeeze() / factorial(degree - r)

def _augment_knots(knots: Vector, degree: int) -> np.ndarray:
    """
    Adds degree + 1 values to either end of the knot vector, in order to facilitate matrix based evaluation.

    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """
    return np.pad(knots, (degree + 1, degree + 1), 'constant', constant_values=(knots[0] - 1, knots[-1] + 1))

def _find_knot_interval(x: float, knots: np.ndarray, endpoint=False) -> int:
    """
    Finds the index i such that knots[i] <= x < knots[i+1]

    :param endpoint:
    :param x: point of interest
    :param knots: knot vector
    :return: index i
    """

    # if we have requested end point, and are at the end, return corresponding index.
    if endpoint and (knots[-2] <= x <= knots[-1]):
        i = max(np.argmax(knots < x) - 1, 0)
        return len(knots) - i - 2

    # if we are outside the domain, return -1
    if x < knots[0] or x > knots[-1]:
        return -1
    # otherwise, return the corresponding index

    return np.max(np.argmax(knots > x) - 1, 0)


def get_coeffs_f_2D(gamma, coeffs_1D):

    coeff_u = coeffs_1D[0]
    coeff_v = coeffs_1D[1]

    results = []
    for v in coeff_v:
        for u in coeff_u:
            results.append(gamma*u*v)

    return results


def get_BEO(LR):
    
    p = LR.p
    C = []

    for el in LR.elements:
        el.AABB = [[el.u_min, el.u_max],[el.v_min, el.v_max]]
        C_el = np.zeros((len(el.supported_b_splines), (p+1)**2))
        supported_b_splines_list = list(el.supported_b_splines)
        supported_b_splines_list.sort(key=lambda bspline: bspline.id)
        el.supported_b_splines_list = supported_b_splines_list
        for fnum, f in enumerate(el.supported_b_splines_list):
            gamma = f.weight
            f.AABB = [f.knots_u, f.knots_v]
            coeffs_f_1D = []
            for f_kv, el_kv in zip(f.AABB, el.AABB):
                u_list_N = np.linspace(el_kv[0], el_kv[-1], p + 1)
                u_list_B = np.linspace(-1, 1, p + 1)
                N_eval = []
                B_eval = []
                for u_N, u_B in zip(u_list_N, u_list_B):
                    N_eval.append(_evaluate_univariate_b_spline(u_N, f_kv, p, r=0))
                    B_eval.append(eval_BS(u_B, p))
                coeffs_f_1D.append(np.linalg.solve(B_eval, N_eval))
            C_el[fnum,:] = get_coeffs_f_2D(gamma, coeffs_f_1D)
        el.BEO = C_el

