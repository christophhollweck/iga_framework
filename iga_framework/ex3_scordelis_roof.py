import numpy as np
import matplotlib.pyplot as plt
import splinepy
from iga_framework import element_6dof_RM_shell


def get_roof1d(phi0=50, phi1=90, R=25, n_elx=4, degree=2, n=100):
    """
    Creates 1D spline data by fitting a curve to points on a circular arc.

    Parameters:
    - phi0: Start angle (in degrees).
    - phi1: End angle (in degrees).
    - R: Radius of the circular arc.
    - n_elx: Number of elements along the x-direction.
    - degree: Degree of the spline curve.
    - n: Number of points used to approximate the arc.

    Returns:
    - data: Original data points on the circular arc (as an array).
    - curve: Fitted spline curve object.
    """
    # Convert angles to radians
    phi0 = np.radians(phi0)
    phi1 = np.radians(phi1)

    # Generate points along the circular arc
    phi = np.linspace(phi1, phi0, n)
    data = [R * np.array([np.cos(phi[i]), np.sin(phi[i])]) for i in range(len(phi))]

    # Fit a spline curve to the data
    n_cps0 = n_elx + degree  # Number of control points
    curve1d, residual_curve = splinepy.helpme.fit.curve(data, degree=degree, n_control_points=n_cps0)
    print(f"Residual of Bspline Curve Fitting: {residual_curve:.5f}")  # Print the fitting residual

    return np.array(data), curve1d


def plot_roof1d(data, curve, n=100):
    """
    Visualizes the 1D roof spline: the original data points, the fitted spline curve,
    control points, knot points, and the control polygon.

    Parameters:
    - data: Original data points on the circular arc.
    - curve: Fitted spline curve object.
    - n: Number of evaluation points along the curve.
    """
    # Evaluate the spline curve at n points
    u = np.linspace(0, 1, n)
    curve_eval = np.array([curve.evaluate([[ui]])[0] for ui in u])

    # Extract unique knot points and evaluate the curve at these positions
    knots = np.unique(curve.knot_vectors[0])
    knot_points = np.array([curve.evaluate([[k]])[0] for k in knots])

    # Control points of the spline
    cps = np.array(curve.cps)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label="Data Points", color="blue", s=15)  # Data points
    plt.plot(curve_eval[:, 0], curve_eval[:, 1], label="Spline Curve", color="red", linewidth=2)  # Spline curve
    plt.scatter(knot_points[:, 0], knot_points[:, 1], label="Knot Points", color="green", s=50, marker="x")  # Knots
    plt.scatter(cps[:, 0], cps[:, 1], label="Control Points", color="orange", s=50, marker="o")  # Control points
    plt.plot(cps[:, 0], cps[:, 1], color="orange", linestyle="--", linewidth=1, label="Control Polygon")  # Control polygon

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Spline Approximation of a Circular Arc with Knots and Control Points")
    plt.show()


def extrude_geometry(curve1d, n_ely, L):
    """
    Extrudes a 1D curve into 2D by adding a second parameter direction (e.g., height).

    Parameters:
    - curve1d: 1D spline curve to be extruded.
    - n_ely: Number of elements in the extruded direction.
    - L: Length of extrusion.

    Returns:
    - Bspl2d: 2D B-spline geometry created by extrusion.
    """
    # Convert 1D control points to 3D for extrusion
    cps1d = curve1d.cps
    cps1d = np.array([[cp[0], 0, cp[1]] for cp in cps1d])  # Add z-coordinate
    ku = curve1d.knot_vectors[0]
    pu = curve1d.degrees[0]

    # Create 1D B-spline
    Bspl1d = splinepy.BSpline(degrees=[pu], knot_vectors=[ku], control_points=cps1d)

    # Extrude by duplicating control points in the second parameter direction
    kv = [0, 0, 1, 1]  # Knot vector for the extruded direction
    pv = 1  # Degree in the extruded direction
    cps2d = []
    for i in range(2):
        for cp in cps1d:
            cps2d.append([cp[0], i * L, cp[2]])

    # Create 2D B-spline
    Bspl2d = splinepy.BSpline(degrees=[pu, pv], knot_vectors=[ku, kv], control_points=cps2d)

    # Perform degree elevation if necessary
    for i in range(pu - 1):
        Bspl2d.elevate_degrees([1])

    # Refine the mesh by inserting knots along the extruded direction
    vins = np.linspace(0, 1, n_ely + 1)
    vins = vins[1:-1]
    Bspl2d.insert_knots(1, vins)

    return Bspl2d


def global_refine_geometry(Bspl2d, reftimes):
    """
    Refines a 2D B-spline geometry globally by inserting knots.

    Parameters:
    - Bspl2d: 2D B-spline geometry to refine.
    - reftimes: Number of global refinement iterations.

    Returns:
    - Bspl2d: Refined 2D B-spline geometry.
    """
    for i in range(reftimes):
        ku = Bspl2d.knot_vectors[0]
        kv = Bspl2d.knot_vectors[1]
        ku_unique = np.unique(ku)
        kv_unique = np.unique(kv)

        # Insert midpoints of knot spans in both parameter directions
        uins = 0.5 * (ku_unique[:-1] + ku_unique[1:])
        vins = 0.5 * (kv_unique[:-1] + kv_unique[1:])

        Bspl2d.insert_knots(0, uins)
        Bspl2d.insert_knots(1, vins)

    return Bspl2d


def surface(n_elx, n_ely, degree):

    n = 100
    phi0 = 50
    phi1 = 90
    R = 25
    L = 25

    """
    Creates a quarter Scordelis-Lo roof geometry (extruded circular arc as a 2D spline).

    Parameters:
    - phi0, phi1, R: Arc parameters (angles and radius).
    - L: Extrusion length.
    - n_elx, n_ely: Number of elements in x and y directions.
    - degree: Spline degree in both directions.
    - n: Number of points for curve fitting.

    Returns:
    - Bspl2d: 2D B-spline representation of the roof.
    """
    # Create the base arc and extrude it into 2D
    data, curve = get_roof1d(phi0, phi1, R, n_elx, degree, n=n)
    Bspl2d = extrude_geometry(curve, n_ely, L)
    return Bspl2d

def solve(RS):

    # scordelis roof, reference solution uz = 0.3024
    t = 0.25
    E = 4.32e8
    nue = 0
    rho = 1
    load = 90

    P = np.array([0, 0, -load])
    shell = element_6dof_RM_shell.shell(E, t, nue, rho, P)

    fixed_DOFs = set()
    fixed_cps = set()

    # ymin
    cps_on_edges = shell.find_control_point_ids_on_edges(RS, xmin=False, xmax=False, ymin=True, ymax=False)
    fixed_cps.update(cps_on_edges)
    fixed_dirs = ['x', 'z', 'ry']
    fixed_DOFs.update(shell.find_fixed_DOFs_on_edges(cps_on_edges, fixed_dirs))

    # ymax
    cps_on_edges = shell.find_control_point_ids_on_edges(RS, xmin=False, xmax=False, ymin=False, ymax=True)
    fixed_cps.update(cps_on_edges)
    fixed_dirs = ['y', 'rx', 'rz']
    fixed_DOFs.update(shell.find_fixed_DOFs_on_edges(cps_on_edges, fixed_dirs))

    # xmin
    cps_on_edges = shell.find_control_point_ids_on_edges(RS, xmin=True, xmax=False, ymin=False, ymax=False)
    fixed_cps.update(cps_on_edges)
    fixed_dirs = ['x', 'ry', 'rz']
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
    x1 = RS.eval_displacement(1,1)[-1]
    # disp = np.linalg.norm(x1)
    print('z-disp:', x1)

    return(RS)


if __name__ == "__main__":
    # Parameters
    n = 100
    phi0 = 50
    phi1 = 90
    R = 25
    L = 25
    n_elx = 3
    n_ely = 3
    degree = 2

    # Generate and visualize the geometry
    Bspl2d = get_scordelis_roof(phi0, phi1, R, L, n_elx, n_ely, degree)
    Bspl2d.show()  # Show initial geometry
    Bspl2d = global_refine_geometry(Bspl2d, 1)  # Apply one global refinement
    Bspl2d.show()  # Show refined geometry
