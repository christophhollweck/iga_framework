import splinepy
import numpy as np
from iga_framework import element_6dof_RM_shell


def surface(n_elU, n_elV, ax, ay, ximax, etamax, conti, p):

    # knots to insert
    uins = np.linspace(0, ximax, n_elU + 1)
    uins = uins[1:-1]
    vins = np.linspace(0, etamax, n_elV + 1)
    vins = vins[1:-1]

    # knotvectors
    U = [0, 0, ximax, ximax]
    V = [0, 0, etamax, etamax]

    # coords(x, y)
    cps0 = np.array([
        [0., 0., 0.],
        [ax, 0., 0.],
        [0, ay, 0.],
        [ax, ay, 0.]
    ])

    # create Bspl Surface
    Bspl = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[U, V],
        control_points=cps0
    )

    # Order Elevation
    for i in range(p - 1):
        Bspl.elevate_degrees([0, 1])

    # after k-refinement, reduce continuity to desired value
    # multiplicity of knot (e.g. m=2 -> double knot values)
    m = p - conti
    for i in range(m):
        # Knot Insertion
        if len(uins) > 0:
            Bspl.insert_knots(0, uins)
        if len(vins) > 0:
            Bspl.insert_knots(1, vins)

    return Bspl


def solve(RS):
    
    # clamped at all 4 edges - CCCC
    # clamped plate, reference solution uz = -0.001503

    t = 0.1
    E = 10920
    nue = 0.3
    rho = 1
    load = 1

    P = np.array([0, 0, -load])
    shell = element_6dof_RM_shell.shell(E, t, nue, rho, P)

    fixed_DOFs = set()
    cps_on_edges = shell.find_control_point_ids_on_edges(RS, xmin=True, xmax=True, ymin=True, ymax=True)
    fixed_dirs = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    fixed_DOFs.update(shell.find_fixed_DOFs_on_edges(cps_on_edges, fixed_dirs))

    shell.P = P
    K, f = shell.build_K_f(RS)
    K_bc, f_bc = shell.set_dirichlet_bc(K, f, fixed_DOFs)

    # coords = [[1, 0]]
    # loads = [[0, 0, -1000]]
    # fpoint = shell.get_pointload_vector(T, coords, loads)
    #K_bc, f_bc = shell.set_dirichlet_bc(K, fpoint, fixed_DOFs)

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


