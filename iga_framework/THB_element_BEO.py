import time
import numpy as np
from Bernstein_polynomials import eval_BS



def compute_2d_basis_functions(B_1d_u, B_1d_v, p):
    meta_list = []

    for i in range(p + 1):
        for j in range(p + 1):
            basis_functions_2d = []
            for k in range(p + 1):
                for l in range(p + 1):
                    basis_value = B_1d_v[i][k] * B_1d_u[j][l]
                    basis_functions_2d.append(basis_value)
            meta_list.append(basis_functions_2d)

    return meta_list

# Standard
def get_BEO(T):

    p = T.p

    B_eval_1D = []
    u_list_B = np.linspace(-1, 1, p + 1)
    for u_B in u_list_B:
        B_eval_1D.append(eval_BS(u_B, p))

    B_eval_1D_u = B_eval_1D
    B_eval_1D_v = B_eval_1D

    B_eval_2D = compute_2d_basis_functions(B_eval_1D_u, B_eval_1D_v, p)


    for el in T.elements:
        C_el = np.zeros((len(el.supported_b_splines), (p+1)**2))
        supported_b_splines_list = list(el.supported_b_splines)
        supported_b_splines_list.sort(key=lambda bspline: bspline.id)
        el.supported_b_splines_list = supported_b_splines_list
        for fnum, f in enumerate(el.supported_b_splines_list):
            u_list_N = np.linspace(el.u_min, el.u_max, p + 1)
            v_list_N = np.linspace(el.v_min, el.v_max, p + 1)
            N_eval = []
            for v in v_list_N:
                for u in u_list_N:
                    N_eval.append(f(u,v))
            C_el[fnum,:] = np.linalg.solve(np.array(B_eval_2D), N_eval)

        el.BEO = C_el


#
# # BEO caching
# def get_BEO(T):
#
#     ps = np.unique([el.p for el in T.elements])
#     B_evals_2D = dict()
#
#     for p in ps:
#
#         B_eval_1D = []
#         u_list_B = np.linspace(-1, 1, p + 1)
#         for u_B in u_list_B:
#             B_eval_1D.append(eval_BS(u_B, p))
#
#         B_eval_1D_u = B_eval_1D
#         B_eval_1D_v = B_eval_1D
#
#         B_eval_2D = compute_2d_basis_functions(B_eval_1D_u, B_eval_1D_v, p)
#         B_evals_2D[p] = B_eval_2D
#
#     # Memoizer to store previously computed C_el
#     memo = {}
#     hits = 0
#
#     for el in T.elements:
#         p = el.p
#         len_bf = len(el.supported_b_splines)
#         supported_b_splines_list = list(el.supported_b_splines)
#         supported_b_splines_list.sort(key=lambda bspline: bspline.id)
#         el.supported_b_splines_list = supported_b_splines_list
#
#         # Generate the norm_knots_bfs key
#         norm_knots_bfs = tuple((tuple(b.norm_knots_u), tuple(b.norm_knots_v)) for b in supported_b_splines_list)
#
#         # Check if all b.trunc are False
#         if all(not b.trunc for b in supported_b_splines_list):
#             # Check if the result is already memoized
#             if norm_knots_bfs in memo:
#                 C_el = memo[norm_knots_bfs]
#                 hits += 1
#             else:
#                 C_el = np.zeros((len_bf, (p + 1) ** 2))
#                 for fnum, f in enumerate(supported_b_splines_list):
#                     u_list_N = np.linspace(el.u_min, el.u_max, p + 1)
#                     v_list_N = np.linspace(el.v_min, el.v_max, p + 1)
#                     N_eval = []
#                     for v in v_list_N:
#                         for u in u_list_N:
#                             N_eval.append(f(u, v))
#                     C_el[fnum, :] = np.linalg.solve(B_evals_2D[p], N_eval)
#                 # Store the computed C_el in the memoizer
#                 memo[norm_knots_bfs] = C_el
#         else:
#             C_el = np.zeros((len_bf, (p + 1) ** 2))
#             for fnum, f in enumerate(supported_b_splines_list):
#                 u_list_N = np.linspace(el.u_min, el.u_max, p + 1)
#                 v_list_N = np.linspace(el.v_min, el.v_max, p + 1)
#                 N_eval = []
#                 for v in v_list_N:
#                     for u in u_list_N:
#                         N_eval.append(f(u, v))
#                 C_el[fnum, :] = np.linalg.solve(B_evals_2D[p], N_eval)
#
#         el.BEO = C_el
#
#     print(hits)




