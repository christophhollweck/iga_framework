import splinepy
import numpy as np
from iga_framework import element_2dof_flat_nonlin_membrane


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
    # parameter for membran
    t = 1
    E = 1
    nu = 0.3
    load = 1
    P = np.array([load, 0])

    membrane = element_2dof_flat_nonlin_membrane.nonlinear_membran(E, nu, t, P, RS)

    fixed_DOFs = []
    cps_on_edges = membrane.find_control_point_ids_on_edges(xmin=True, xmax=False, ymin=False, ymax=False)
    fixed_dirs = ['x', 'y']
    fixed_DOFs.append(membrane.find_fixed_DOFs_on_edges(cps_on_edges, fixed_dirs))
    membrane.newton_raphson(fixed_DOFs, tolerance=1e-6, max_iterations=50)

    membrane.postprocessing(RS)

    return RS
