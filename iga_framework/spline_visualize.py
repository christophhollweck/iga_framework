import matplotlib.pyplot as plt
import matplotlib.patches as plp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator


# THB
def plot_mesh_overloading(RS, overloading=True, text=True, relative=True, filename=None,
                          color=True, title=True, axes=False) -> None:
    """
    Plots the LR-mesh.
    """
    fig = plt.figure(figsize=(10, 6))
    axs = fig.add_subplot(1, 1, 1)
    overloaded_patch = plp.Patch(color='red', label='Overloaded Elements')
    underloaded_patch = plp.Patch(color='green', label='Underloaded Elements')
    normal_patch = plp.Patch(color='black', label='Normal Elements')

    for m in RS.elements:
        w = m.u_max - m.u_min
        h = m.v_max - m.v_min
        midpoint_x = (m.u_min + m.u_max) / 2
        midpoint_y = (m.v_min + m.v_max) / 2

        if len(m.supported_b_splines) == (m.p + 1) ** 2:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='black' if color else 'black',
                                        alpha=0.2))
        elif len(m.supported_b_splines) < (m.p + 1) ** 2:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='green' if color else 'black',
                                        alpha=0.2))
        else:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='red' if color else 'white',
                                        alpha=0.2))
        if text:
            axs.text(midpoint_x, midpoint_y, '{}'.format(len(m.supported_b_splines)), ha='center',
                     va='center')

    if title:
        plt.title('dim(S) = {}'.format(len(RS.basis)))

    if not axes:
        plt.axis('off')

    max_u = RS.ku[-1]
    max_v = RS.kv[-1]
    axs.set_xlim(0, max_u)
    axs.set_ylim(0, max_v)

    # Adjust layout to fit legend
    plt.subplots_adjust(right=0.75)  # Increase right margin further

    # Add legend outside the plot
    axs.legend(handles=[overloaded_patch, underloaded_patch, normal_patch],
               loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.show()

def plot_elements(RS):
    """
    Plots the buckets and elements of the given object RS.

    :param RS: Object containing 'buckets' and 'elements'.
    """
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)

    # Check if buckets exist and plot them
    if hasattr(RS, 'buckets') and RS.buckets:
        for bucket in RS.buckets:
            AABB = bucket.AABB  # Axis-aligned bounding box
            u_center = (AABB[0][0] + AABB[0][1]) / 2
            v_center = (AABB[1][0] + AABB[1][1]) / 2

            # Plot bucket boundary
            axs.plot(
                [AABB[0][0], AABB[0][1], AABB[0][1], AABB[0][0], AABB[0][0]],
                [AABB[1][0], AABB[1][0], AABB[1][1], AABB[1][1], AABB[1][0]],
                'b-', linewidth=5, alpha=0.5
            )

            # Add bucket ID
            axs.text(u_center, v_center, '{}'.format(str(bucket.id)),
                     bbox=dict(facecolor='blue', alpha=1), color='white',
                     ha='center', va='center', fontsize=25)

    # Check if elements exist and plot them
    if hasattr(RS, 'elements') and RS.elements:
        for m in RS.elements:
            # Calculate dimensions and midpoint of the element
            w = m.u_max - m.u_min
            h = m.v_max - m.v_min
            midpoint_x = (m.u_min + m.u_max) / 2
            midpoint_y = (m.v_min + m.v_max) / 2

            # Plot element as a rectangle
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True,
                                            color='black', alpha=0.2))

            # Add element ID
            axs.text(midpoint_x, midpoint_y, '{}'.format(m.id),
                     ha='center', va='center')

    # Finalize plot
    plt.xlabel('u')
    plt.ylabel('v')
    if hasattr(RS, 'buckets') and RS.buckets:
        plt.title('Buckets and Elements')
    else:
        plt.title('Elements')
    plt.show()


def plot_ips(RS):
    fig = plt.figure(figsize=(15, 15))
    for cell in RS.elements:
        AABB = [[cell.u_min, cell.u_max], [cell.v_min, cell.v_max]]
        cellAABB = np.array(AABB)
        x = cellAABB[0, [0, 1, 1, 0, 0]]
        y = cellAABB[1, [0, 0, 1, 1, 0]]
        plt.plot(x, y, color='black')
        for gp in cell.gausspoints:
            plt.scatter(gp.xi, gp.eta, color='red', s=8)
    plt.show()

def plot_basis(RS, sc=5, N=30):
    U = RS.ku
    V = RS.kv
    ximax = U[-1]
    etamax = V[-1]
    xi = np.linspace(0, ximax, N)
    eta = np.linspace(0, etamax, N)
    z = np.zeros((N, N, len(RS.basis)))

    k = -1
    for f in RS.basis:
        k += 1
        for i in range(N):
            for j in range(N):
                z[i, j, k] = f(xi[i], eta[j])

    unity_check = np.sum(z, axis=2)
    if np.all((0.999 < unity_check) & (unity_check < 1.001)):
        print('Space holds Partition of Unity')
    else:
        print('No Partition of Unity!')

    Mat = []
    k = -1

    for fun in RS.basis:
        k += 1
        xi_dir = fun.knots_u
        eta_dir = fun.knots_v
        pnts = N
        xi = np.linspace(xi_dir[0], xi_dir[-1], pnts)
        eta = np.linspace(eta_dir[0], eta_dir[-1], pnts)
        Xi, Eta = np.meshgrid(xi, eta)
        z = np.zeros((pnts, pnts))
        for i in range(pnts):
            for j in range(pnts):
                z[i, j] = fun(xi[i], eta[j])
        Mat.append([Xi.T, Eta.T, z])

    # PyVista Plotting
    plotter = pv.Plotter()

    # Add basis functions as surfaces
    for Xi, Eta, Z in Mat:
        X = sc * Xi
        Y = sc * Eta
        surface = pv.StructuredGrid(X, Y, Z)
        surface.point_data['z_values'] = Z.ravel(order='F')
        plotter.add_mesh(surface, cmap="coolwarm", opacity=1)

    # Add flat bottom surface
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.zeros((5, 5))
    X, Y = np.meshgrid(x, y)
    flat_surface = pv.StructuredGrid(sc * X.T, sc * Y.T, z)
    flat_surface.point_data['z_values'] = z.ravel(order='F')
    plotter.add_mesh(flat_surface, cmap="coolwarm", opacity=1)

    # Show the plot
    plotter.show()

def plot_surface(RS, n_points=10, s=0.05, evalpnt=None, stressinterp=False, loadvector=None, output_path=None, display_mode='displacement_z'):
    """
    Plot the surface with options to display stress or displacement in different components.

    Parameters:
    - RS: The refined space object.
    - n_points: Number of points for mesh refinement.
    - s: Scaling factor for arrows.
    - evalpnt: Evaluation point.
    - stressinterp: Flag for stress interpolation.
    - loadvector: Load vector to be displayed.
    - output_path: Path to save the plot (optional).
    - display_mode: Mode for color display ('displacement_x', 'displacement_y', 'displacement_z',
                   'displacement_magnitude', 'stress_von_mises').
    """

    def eval(u, v):
        x = np.zeros_like(u)
        y = np.zeros_like(u)
        z = np.zeros_like(u)
        ux = np.zeros_like(u)
        uy = np.zeros_like(u)
        uz = np.zeros_like(u)
        u_mag = np.zeros_like(u)
        for i in range(len(u)):
            eval_point = RS(u[i], v[i])
            eval_disp = RS.eval_displacement(u[i], v[i])
            geo = eval_point + eval_disp
            x[i] = geo[0]
            y[i] = geo[1]
            z[i] = geo[2]
            ux[i] = eval_disp[0]
            uy[i] = eval_disp[1]
            uz[i] = eval_disp[2]
            u_mag[i] = np.linalg.norm(eval_disp)
        return x, y, z, ux, uy, uz, u_mag

    points = []
    faces = []
    scalars = []
    horizontal_lines = set()
    vertical_lines = set()
    point_offset = 0

    for element in RS.elements:
        u_min, v_min, u_max, v_max = element.u_min, element.v_min, element.u_max, element.v_max
        u, v = np.meshgrid(np.linspace(u_min, u_max, n_points), np.linspace(v_min, v_max, n_points))
        x, y, z, ux, uy, uz, u_mag = eval(u.flatten(), v.flatten())

        horizontal_lines.add((u_min, u_max, v_min))
        horizontal_lines.add((u_min, u_max, v_max))
        vertical_lines.add((v_min, v_max, u_min))
        vertical_lines.add((v_min, v_max, u_max))

        element_points = np.column_stack((x, y, z))
        points.append(element_points)

        for i in range(n_points - 1):
            for j in range(n_points - 1):
                p1 = point_offset + i * n_points + j
                p2 = point_offset + i * n_points + (j + 1)
                p3 = point_offset + (i + 1) * n_points + j
                p4 = point_offset + (i + 1) * n_points + (j + 1)
                faces.append([3, p1, p2, p3])
                faces.append([3, p3, p2, p4])

        if display_mode == 'displacement_x':
            scalars.extend(ux)
        elif display_mode == 'displacement_y':
            scalars.extend(uy)
        elif display_mode == 'displacement_z':
            scalars.extend(uz)
        elif display_mode == 'displacement_magnitude':
            scalars.extend(u_mag)
        elif display_mode == 'stress_von_mises':
            if stressinterp and len(element.gausspoints) > 0:
                gp_locations = np.array([(gp.xi, gp.eta) for gp in element.gausspoints])
                von_mises_stresses = [gp.sigma_vM for gp in element.gausspoints]
                rbf_interpolator = RBFInterpolator(gp_locations, von_mises_stresses, kernel='linear')
                interpolated_stress = rbf_interpolator(np.column_stack((u.flatten(), v.flatten())))
                scalars.extend(interpolated_stress)
            else:
                avg_stress = np.mean([gp.sigma_vM for gp in element.gausspoints]) if len(element.gausspoints) > 0 else 0
                scalars.extend([avg_stress] * element_points.shape[0])

        point_offset += len(element_points)

    points = np.vstack(points)
    faces = np.hstack(faces)
    scalars = np.array(scalars)

    surface = pv.PolyData(points)
    surface.faces = faces
    surface[display_mode] = scalars

    plotter = pv.Plotter()
    plotter.add_mesh(surface, scalars=display_mode, cmap="coolwarm", show_scalar_bar=True, show_edges=False, point_size=0)

    # Erzeuge die Linien
    line_points = []  # Liste für die Punkte der Linien
    line_cells = []  # Liste für die Linienzellen (Verbindungen zwischen Punkten)

    line_offset = 0  # Offset für die Linienindizes
    # Horizontale Linien
    for line in horizontal_lines:
        u_start, u_end, v_const = line
        ul = np.linspace(u_start, u_end, n_points)
        vl = np.full_like(ul, v_const)  # Konstantes v
        xl, yl, zl, _, _, _, _ = eval(ul, vl)
        for i in range(len(xl)):
            line_points.append([xl[i], yl[i], zl[i]])
            if i > 0:  # Verbinde aufeinanderfolgende Punkte zu einer Linie
                line_cells.append([2, line_offset + i - 1, line_offset + i])
        line_offset += len(xl)

    # Vertikale Linien
    for line in vertical_lines:
        v_start, v_end, u_const = line
        vl = np.linspace(v_start, v_end, n_points)
        ul = np.full_like(vl, u_const)  # Konstantes u
        xl, yl, zl, _, _, _, _ = eval(ul, vl)
        for i in range(len(xl)):
            line_points.append([xl[i], yl[i], zl[i]])
            if i > 0:  # Verbinde aufeinanderfolgende Punkte zu einer Linie
                line_cells.append([2, line_offset + i - 1, line_offset + i])
        line_offset += len(xl)

    # Erstelle das PolyData für die Linien
    line_mesh = pv.PolyData(np.array(line_points))  # Mesh für die Linien
    line_mesh.lines = np.array(line_cells)  # Linienzellen (Verbindungen zwischen Punkten)

    plotter.add_mesh(line_mesh, color="black", line_width=0.1, point_size=0)

    axis_length = 1.0

    x_arrow = pv.Arrow(start=[0, 0, 0], direction=[axis_length, 0, 0], scale=s * axis_length)
    plotter.add_mesh(x_arrow, color="red", label="X-Axis", point_size=0)

    y_arrow = pv.Arrow(start=[0, 0, 0], direction=[0, axis_length, 0], scale=s * axis_length)
    plotter.add_mesh(y_arrow, color="green", label="Y-Axis", point_size=0)

    z_arrow = pv.Arrow(start=[0, 0, 0], direction=[0, 0, axis_length], scale=s * axis_length)
    plotter.add_mesh(z_arrow, color="blue", label="Z-Axis", point_size=0)

    if loadvector is not None:
        loadvector = loadvector / np.linalg.norm(loadvector)
        v_start = np.array([1, 1, 1])
        v_end = v_start + loadvector
        v_arrow = pv.Arrow(start=v_start, direction=v_end - v_start, scale=s * axis_length)
        plotter.add_mesh(v_arrow, color="black", label="Load Vector", point_size=0)

    plotter.show()
