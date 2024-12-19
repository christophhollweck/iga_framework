import splinepy
import numpy as np
from iga_framework import element_6dof_RM_shell

def surface():

    ku = [0, 0, 0, 0.5, 1, 1, 1]
    kv = [0, 0, 0, 1, 1, 1]
    p = 2
    conti = p - 1

    cps = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [-1, 0.5, 1],
            [-1, 0.5, 0],
            [1, 0.5, 0],
            [1, 0.5, 1],
            [-1, 1, 1],
            [-1, 1, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
    )

    Bspl = splinepy.BSpline(
        degrees=[p, p],
        knot_vectors=[ku, kv],
        control_points=cps,
    )

    Bspl.insert_knots(0, [0.25, 0.75])
    Bspl.insert_knots(1, [0.25, 0.5, 0.75])

    cps = Bspl.cps
    ku = Bspl.knot_vectors[0]
    kv = Bspl.knot_vectors[1]

    return Bspl

def solve(RS):
    
    # clamped at all 4 edges - CCCC

    t = 0.1
    E = 10920
    nue = 0.3
    rho = 1
    load = 0

    P = np.array([0, 0, -load])
    shell = element_6dof_RM_shell.shell(E, t, nue, rho, P)

    fixed_DOFs = set()
    cps_on_edges = shell.find_control_point_ids_on_edges(RS, xmin=True, xmax=True, ymin=True, ymax=True)
    fixed_dirs = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    fixed_DOFs.update(shell.find_fixed_DOFs_on_edges(cps_on_edges, fixed_dirs))

    shell.P = P
    K, _ = shell.build_K_f(RS)

    coords = [[0.5, 0.5]]
    loads = [[0, 0, 10]]
    fpoint = shell.get_pointload_vector(RS, coords, loads)
    K_bc, f_bc = shell.set_dirichlet_bc(K, fpoint, fixed_DOFs)

    dis = np.linalg.solve(K_bc, f_bc)
    # from scipy.sparse.linalg import spsolve
    # # Lösen des Gleichungssystems für Sparse-Matrizen
    # dis = spsolve(K_bc, f_bc)

    shell.update_cps(RS, dis)
    shell.postprocessing(RS)
    x1 = RS.eval_displacement(0.5,0.5)[-1]
    # disp = np.linalg.norm(x1)
    print('z-disp:', x1)

    return(RS)
