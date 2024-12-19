from scipy.linalg import eigh
import numpy as np
import numpy.polynomial.legendre as gq
from scipy.sparse import lil_matrix


class shell():

    def __init__(self, E, t, nue, rho, P):
        """
        Initialize the membran with its material characteristics.
        :param E, t, nue, rho: Youngs Modulus, thickness, Poissions ratio, density
        """
        self.E = E
        self.t = t
        self.nue = nue
        self.rho = rho
        # load vector [x,y,z]
        self.P = P
        kappa = 5/6
        self.Cmat_loc = E / (1 - nue ** 2) * np.array([[1, nue, 0, 0, 0, 0], [nue, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0.5 * (1 - nue), 0, 0], [0, 0, 0, 0, kappa/2*(1 - nue), 0], [0, 0, 0, 0, 0, kappa/2*(1 -nue)]])



    def build_K_f(self, RS, trimming=False):
        """
        calculation of the stiffness matrix and load vector for surface load
        :param RS: an object of the spline space
        """
        if trimming:
            cells = [cell for cell in RS.elements if len(cell.gausspoints) > 0]
        else:
            cells = RS.elements
            [cell.get_gp_gw_2d() for cell in cells]

        # get all active functions ids
        afuncs = np.unique([b.id for cell in cells for b in cell.supported_b_splines])
        n_cps = len(RS.basis)
        dof_per_cp = 6
        n_dof_glob = dof_per_cp * n_cps
        nel = len(cells)

        K = np.zeros((n_dof_glob, n_dof_glob))
        # K = lil_matrix((n_dof_glob, n_dof_glob))
        f = np.zeros(n_dof_glob)

        # thickness
        t = self.t

        for cell in cells:
            n_el_cps = len(cell.supported_b_splines)
            # sort bf ids according to their ids
            bfnums = np.sort([b.id for b in cell.supported_b_splines])
            # sort bf according to their ids
            bfs_loc = list(cell.supported_b_splines)
            bfs_loc.sort(key=lambda bspline: bspline.id)

            K_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
            f_el = np.zeros(dof_per_cp * n_el_cps)
            # cps
            cpsx = [b.cp[0] for b in bfs_loc]
            cpsy = [b.cp[1] for b in bfs_loc]
            cpsz = [b.cp[2] for b in bfs_loc]

            # integration points and weights for thickness integration
            gauss_t = gq.leggauss(2)
            ngp_t = len(gauss_t[0])
            gp_t = gauss_t[0]
            gw_t = gauss_t[1]

            n_mean = np.array([0.0, 0.0, 0.0])

            # inplane loop
            for iwip in cell.gausspoints:
                gpx = iwip.xi
                gpy = iwip.eta
                gw = iwip.weight_inplane
                iwip.C_loc = self.Cmat_loc
                iwip.t = t

                # bfs arranged in a Vector
                N_vec = np.array([bf(gpx, gpy) for bf in bfs_loc])
                # bfs derivatives arranged in a Matrix
                dNdxi_mat = np.array([bf.grad(gpx, gpy) for bf in bfs_loc]).T  # Gradients of basis functions

                # bfs arranged in a Matrix
                N_mat = np.zeros((dof_per_cp, dof_per_cp * n_el_cps))
                I = np.identity(dof_per_cp)  # [dof_per_cp, dof_per_cp]
                for i, bf_val in enumerate(N_vec):
                    N_mat[:, i * dof_per_cp:(i + 1) * dof_per_cp] = I * bf_val

                # get the covariante basis vectors a1, a2, normal n and derivatives of the normal dn/dxi and dn/deta
                # dn/dxi and dn/deta is perpendicular to n -> dotproduct == 0
                a1, a2, n, a1_xi, a2_eta, a1_eta, a2_xi, dn_dxi, dn_deta = self.get_local_basis_and_variations(RS, gpx, gpy)
                n1 = n[0]
                n2 = n[1]
                n3 = n[2]

                n_mean += n

                # loop for thickness direction
                for k in range(ngp_t):
                    # calculate values for the jacobian
                    zeta = gp_t[k]
                    iwip.zeta.append(zeta)
                    iwip.weight_outerplane.append(gw_t[k])
                    dxdxi = np.dot(dNdxi_mat[0, :], cpsx) + t / 2 * zeta * dn_dxi[0]
                    dydxi = np.dot(dNdxi_mat[0, :], cpsy) + t / 2 * zeta * dn_dxi[1]
                    dzdxi = np.dot(dNdxi_mat[0, :], cpsz) + t / 2 * zeta * dn_dxi[2]
                    dxdeta = np.dot(dNdxi_mat[1, :], cpsx) + t / 2 * zeta * dn_deta[0]
                    dydeta = np.dot(dNdxi_mat[1, :], cpsy) + t / 2 * zeta * dn_deta[1]
                    dzdeta = np.dot(dNdxi_mat[1, :], cpsz) + t / 2 * zeta * dn_deta[2]
                    dxdzeta = t / 2 * n[0]
                    dydzeta = t / 2 * n[1]
                    dzdzeta = t / 2 * n[2]

                    # assemble J
                    J = np.array([[dxdxi, dydxi, dzdxi], [dxdeta, dydeta, dzdeta], [dxdzeta, dydzeta, dzdzeta]])

                    # detJ2 is needed for force integral as it is a surface integral/load
                    g1 = np.array([dxdxi, dydxi, dzdxi])
                    g2 = np.array([dxdeta, dydeta, dzdeta])
                    cross_product = np.cross(g1, g2)
                    det_J2 = np.linalg.norm(cross_product)

                    # detJ3 is needed for K as it is a volume integral
                    det_J3 = np.linalg.det(J)

                    # Inverse of J needed for the B-Matrix
                    # Jinv = [dxidx  detadx  dzetadx]
                    #        [dxidy  detady  dzetady]
                    #        [dxidz  detadz  dzetadz] (=> J*Jinv = 1)

                    Jinv = np.linalg.inv(J)
                    dzetadx, dzetady, dzetadz = Jinv[0, 2], Jinv[1, 2], Jinv[2, 2]

                    # arrange the derivatives of the basisfunction w.r.t. the parameter space in a matrix with 3 rows
                    # third row is zero as dNi/dzeta is always zeros (no function of zeta)
                    dNdxi_3mat = np.zeros((3, n_el_cps))
                    dNdxi_3mat[:2, :] = dNdxi_mat
                    # isoparametric concept for basisfunctions (dNi/dX = invJ*dNi/dXi)
                    dNdx_mat = Jinv @ dNdxi_3mat

                    # isoparametric concept for normals dnx/dX = invJ*dnx/dXi
                    # (variation of the normal n w.r.t. zeta is assumed to be zero. This should hold true
                    # assuming that the director does not deform and undergoes a rigid body rotation only
                    # x component of n == [dnxdx, dnxdy, dnxdz]
                    # y component of n == [dnydx, dnydy, dnydz]
                    # z component of n == [dnzdx, dnzdy, dnzdz]

                    dn_dXi = np.vstack([dn_dxi, dn_deta, np.zeros(3)])
                    dndX = Jinv @ dn_dXi

                    # arrange into vectors
                    dndx, dndy, dndz = dndX

                    # resave
                    dn1dx, dn2dx, dn3dx = dndx
                    dn1dy, dn2dy, dn3dy = dndy
                    dn1dz, dn2dz, dn3dz = dndz

                    dNdx = dNdx_mat[0, :]
                    dNdy = dNdx_mat[1, :]
                    dNdz = dNdx_mat[2, :]

                    # local orthonormal basis
                    t1 = np.array([dxdxi, dydxi, dzdxi])
                    t1 = t1 / np.linalg.norm(t1)
                    t3 = n
                    t2 = np.cross(t3, t1)

                    iwip.local_basis.append([t1, t2, t3])

                    # Rotationmatrix from local System (t1,t2,t3) -> (x,y,z)
                    T = np.zeros((6, 6))

                    T[0, 0] = t1[0] * t1[0]
                    T[0, 1] = t2[0] * t2[0]
                    T[0, 2] = t3[0] * t3[0]
                    T[0, 3] = 2 * t1[0] * t2[0]
                    T[0, 4] = 2 * t2[0] * t3[0]
                    T[0, 5] = 2 * t1[0] * t3[0]

                    T[1, 0] = t1[1] * t1[1]
                    T[1, 1] = t2[1] * t2[1]
                    T[1, 2] = t3[1] * t3[1]
                    T[1, 3] = 2 * t1[1] * t2[1]
                    T[1, 4] = 2 * t2[1] * t3[1]
                    T[1, 5] = 2 * t1[1] * t3[1]

                    T[2, 0] = t1[2] * t1[2]
                    T[2, 1] = t2[2] * t2[2]
                    T[2, 2] = t3[2] * t3[2]
                    T[2, 3] = 2 * t1[2] * t2[2]
                    T[2, 4] = 2 * t2[2] * t3[2]
                    T[2, 5] = 2 * t1[2] * t3[2]

                    T[3, 0] = t1[0] * t1[1]
                    T[3, 1] = t2[0] * t2[1]
                    T[3, 2] = t3[0] * t3[1]
                    T[3, 3] = t1[0] * t2[1] + t2[0] * t1[1]
                    T[3, 4] = t2[0] * t3[1] + t3[0] * t2[1]
                    T[3, 5] = t1[0] * t3[1] + t3[0] * t1[1]

                    T[4, 0] = t1[1] * t1[2]
                    T[4, 1] = t2[1] * t2[2]
                    T[4, 2] = t3[1] * t3[2]
                    T[4, 3] = t1[1] * t2[2] + t2[1] * t1[2]
                    T[4, 4] = t2[1] * t3[2] + t3[1] * t2[2]
                    T[4, 5] = t1[1] * t3[2] + t3[1] * t1[2]

                    T[5, 0] = t1[0] * t1[2]
                    T[5, 1] = t2[0] * t2[2]
                    T[5, 2] = t3[0] * t3[2]
                    T[5, 3] = t1[0] * t2[2] + t2[0] * t1[2]
                    T[5, 4] = t2[0] * t3[2] + t3[0] * t2[2]
                    T[5, 5] = t1[0] * t3[2] + t3[0] * t1[2]

                    iwip.rotationmatrices.append(T)

                    # rotate material law into global system
                    Cmat_glob = T @ self.Cmat_loc @ np.transpose(T)

                    # B-Matrix
                    B = np.zeros((6, 6 * n_el_cps))

                    # Bv
                    B[0, 0::dof_per_cp] = dNdx
                    B[1, 1::dof_per_cp] = dNdy
                    B[2, 2::dof_per_cp] = dNdz
                    B[3, 0::dof_per_cp] = dNdy
                    B[3, 1::dof_per_cp] = dNdx
                    B[4, 1::dof_per_cp] = dNdz
                    B[4, 2::dof_per_cp] = dNdy
                    B[5, 0::dof_per_cp] = dNdz
                    B[5, 2::dof_per_cp] = dNdx

                    # B_omega
                    B[0, 4::dof_per_cp] = (N_vec * (dn3dx * zeta + dzetadx * n3) + dNdx * n3 * zeta) * t / 2
                    B[0, 5::dof_per_cp] = (-N_vec * (dn2dx * zeta + dzetadx * n2) - dNdx * n2 * zeta) * t / 2

                    B[1, 3::dof_per_cp] = (-N_vec * (dn3dy * zeta + dzetady * n3) - dNdy * n3 * zeta) * t / 2
                    B[1, 5::dof_per_cp] = (N_vec * (dn1dy * zeta + dzetady * n1) + dNdy * n1 * zeta) * t / 2

                    B[2, 3::dof_per_cp] = (N_vec * (dn2dz * zeta + dzetadz * n2) + dNdz * n2 * zeta) * t / 2
                    B[2, 4::dof_per_cp] = (-N_vec * (dn1dz * zeta + dzetadz * n1) - dNdz * n1 * zeta) * t / 2

                    B[3, 3::dof_per_cp] = (-N_vec * (dn3dx * zeta + dzetadx * n3) - dNdx * n3 * zeta) * t / 2
                    B[3, 4::dof_per_cp] = (N_vec * (dn3dy * zeta + dzetady * n3) + dNdy * n3 * zeta) * t / 2
                    B[3, 5::dof_per_cp] = (N_vec * (
                                dn1dx * zeta - dn2dy * zeta + dzetadx * n1 - dzetady * n2) + dNdx * n1 * zeta - dNdy * n2 * zeta) * t / 2

                    B[4, 3::dof_per_cp] = (N_vec * (
                                dn2dy * zeta - dn3dz * zeta + dzetady * n2 - dzetadz * n3) + dNdy * n2 * zeta - dNdz * n3 * zeta) * t / 2
                    B[4, 4::dof_per_cp] = (-N_vec * (dn1dy * zeta + dzetady * n1) - dNdy * n1 * zeta) * t / 2
                    B[4, 5::dof_per_cp] = (N_vec * (dn1dz * zeta + dzetadz * n1) + dNdz * n1 * zeta) * t / 2

                    B[5, 3::dof_per_cp] = (N_vec * (dn2dx * zeta + dzetadx * n2) + dNdx * n2 * zeta) * t / 2
                    B[5, 4::dof_per_cp] = (-N_vec * (
                                dn1dx * zeta - dn3dz * zeta + dzetadx * n1 - dzetadz * n3) - dNdx * n1 * zeta + dNdz * n3 * zeta) * t / 2
                    B[5, 5::dof_per_cp] = (-N_vec * (dn2dz * zeta + dzetadz * n2) - dNdz * n2 * zeta) * t / 2

                    iwip.Bmatrices.append(B)
                    K_el += (np.transpose(B) @ Cmat_glob @ B) * gw * gw_t[k] * det_J3

                    Bdrill = np.zeros((1, dof_per_cp * n_el_cps))
                    Bdrill[0, 0::dof_per_cp] = -0.5 * dNdy
                    Bdrill[0, 1::dof_per_cp] = 0.5 * dNdx
                    Bdrill[0, 5::dof_per_cp] = -N_vec
                    K_el += (0.05 * self.E * (Bdrill.T @ Bdrill) * gw * gw_t[k] * det_J3)

                # load in x-direction (1. DOF for shells)
                Nelx = np.array(N_mat[0, :])
                # load in y-direction (2. DOF for shells)
                Nely = np.array(N_mat[1, :])
                # load in z-direction (3. DOF for shells)
                Nelz = np.array(N_mat[2, :])

                f_el += (Nelx.T * self.P[0]) * gw * gw_t[k] * det_J2
                f_el += (Nely.T * self.P[1]) * gw * gw_t[k] * det_J2
                f_el += (Nelz.T * self.P[2]) * gw * gw_t[k] * det_J2

            # K_el -> K

            # # stabilisation of the drilling DOF on an element level using a mean normal
            # # 0.005 gave good results for scordelis roof
            # n_mean /= len(cell.gausspoints)
            # for jj in range(n_el_cps):
            #     dof_rot = [jj * 6 + 3, jj * 6 + 4, jj * 6 + 5]
            #     Ksub = K_el[np.ix_(dof_rot, dof_rot)]
            #     K_el[np.ix_(dof_rot, dof_rot)] += 0.005 * np.max(np.diag(Ksub)) * np.outer(n_mean, n_mean)

            dofs = np.sort(np.concatenate([bfnums * dof_per_cp + i for i in range(dof_per_cp)]))
            K[np.ix_(dofs, dofs)] += K_el

            # for i, dof_i in enumerate(dofs):
            #     for j, dof_j in enumerate(dofs):
            #         K[dof_i, dof_j] += K_el[i, j]  # Eintrag zur Sparse-Matrix hinzufügen

            # f_el -> f
            f[np.ix_(dofs)] += f_el

        # take only active DOFs (all for untrimmed)
        adofs = np.sort(np.concatenate([afuncs * dof_per_cp + i for i in range(dof_per_cp)]))
        self.K = K[np.ix_(adofs, adofs)]

        return K, f

    def get_pointload_vector(self, RS, coords, loads, trimming=False):
        """
        calculation of the pointload vector
        coords: list of parametric coordinates xi and eta
        loads: list of load vectors containing x,y,z contribution
        :param hierarchical_space: an object of the hierarchical space
        """
        if trimming:
            cells = [cell for cell in RS.elements if len(cell.gauss_points) > 0]
        else:
            cells = RS.elements

        # get all active functions ids
        afuncs = np.unique([b.id for cell in cells for b in cell.supported_b_splines])
        n_cps = len(RS.basis)
        dof_per_cp = 6
        n_dof_glob = dof_per_cp * n_cps
        nel = len(cells)
        f = np.zeros(n_dof_glob)


        for coord, load in zip(coords, loads):
            for cell in cells:
                if cell.contains(coord[0], coord[1]):
                    n_el_cps = len(cell.supported_b_splines)
                    # sort bf ids according to their ids
                    bfnums = np.sort([b.id for b in cell.supported_b_splines])
                    # sort bf according to their ids
                    bfs_loc = list(cell.supported_b_splines)
                    bfs_loc.sort(key=lambda bspline: bspline.id)
                    f_el = np.zeros(dof_per_cp * n_el_cps)
                    # bfs arranged in a Matrix
                    N_mat = np.zeros((dof_per_cp, dof_per_cp * n_el_cps))
                    # loc_bfnum = local bf indices starting from 0
                    for bf_loc, loc_bfnum in zip(bfs_loc, range(n_el_cps)):
                        bf_val = bf_loc(coord[0], coord[1])
                        N_mat[:, dof_per_cp * loc_bfnum: dof_per_cp * loc_bfnum + dof_per_cp] = np.identity(dof_per_cp) * \
                                                              bf_val

                    # load in x-direction (1. DOF for shells)
                    Nelx = np.array(N_mat[0, :])
                    # load in y-direction (2. DOF for shells)
                    Nely = np.array(N_mat[1, :])
                    # load in z-direction (3. DOF for shells)
                    Nelz = np.array(N_mat[2, :])

                    f_el += (Nelx.T * load[0])
                    f_el += (Nely.T * load[1])
                    f_el += (Nelz.T * load[2])

                    # f_el -> f
                    dofs = np.sort(np.concatenate([bfnums * dof_per_cp + i for i in range(dof_per_cp)]))
                    f[np.ix_(dofs)] += f_el

        # take only active DOFs (all for untrimmed)
        adofs = np.sort(np.concatenate([afuncs * dof_per_cp + i for i in range(dof_per_cp)]))
        return f

    def find_control_point_ids_on_edges(self, RS, xmin=False, xmax=False, ymin=False, ymax=False):
        # Step 1: Find min and max values for x and y among all control points
        control_points = np.array([b.cp for b in RS.basis])
        x_min_val = control_points[:, 0].min()
        x_max_val = control_points[:, 0].max()
        y_min_val = control_points[:, 1].min()
        y_max_val = control_points[:, 1].max()

        control_point_ids = set()  # Set to store the IDs of control points on the edges

        # Step 2: Loop over all control points and check if they are on the specified edges
        for b in RS.basis:
            cp = b.cp  # Access the control point (x, y, z)
            cp_id = b.id  # Access the ID of the control point

            # Initialize a flag to check if the control point is on the edge
            on_edge = False

            # Check for each edge based on the boolean inputs and np.isclose
            if xmin and np.isclose(cp[0], x_min_val):
                on_edge = True
            if xmax and np.isclose(cp[0], x_max_val):
                on_edge = True
            if ymin and np.isclose(cp[1], y_min_val):
                on_edge = True
            if ymax and np.isclose(cp[1], y_max_val):
                on_edge = True

            # If the control point is on one of the edges, add its ID to the set
            if on_edge:
                control_point_ids.add(cp_id)

        return control_point_ids

    def find_fixed_DOFs_on_edges(self, cps_on_edges, fixed_dirs):
        # u, v, w, rx, ry, rz
        fixed_DOFs = {
            'x': [6 * cp for cp in cps_on_edges],
            'y': [6 * cp + 1 for cp in cps_on_edges],
            'z': [6 * cp + 2 for cp in cps_on_edges],
            'rx': [6 * cp + 3 for cp in cps_on_edges],
            'ry': [6 * cp + 4 for cp in cps_on_edges],
            'rz': [6 * cp + 5 for cp in cps_on_edges]
        }

        total_fixed_DOFs = set()
        for fixed_dir in fixed_dirs:
            total_fixed_DOFs.update(fixed_DOFs[fixed_dir])

        return total_fixed_DOFs

    def set_dirichlet_bc(self, K, f, fixed_DOFs):

        # essential bc
        for dof in fixed_DOFs:
            K[:, dof] = 0
            K[dof, :] = 0
            K[dof, dof] = 1
            f[dof] = 0

        return K, f

    # import numpy as np
    # from scipy.sparse import lil_matrix
    #
    # def set_dirichlet_bc(self, K, f, fixed_DOFs):
    #     """
    #     Setzt Dirichlet-Randbedingungen für Sparse-Matrizen.
    #     K: Sparse-Matrix (z. B. lil_matrix)
    #     f: Rechte-Seiten-Vektor (numpy-Array oder Sparse-Vektor)
    #     fixed_DOFs: Liste der fixierten Freiheitsgrade
    #     """
    #     # Konvertiere K zu lil_matrix, falls erforderlich
    #     if not isinstance(K, lil_matrix):
    #         K = K.tolil()
    #
    #     # Essential Boundary Conditions
    #     for dof in fixed_DOFs:
    #         # Nullsetzen der Zeile und Spalte
    #         K[dof, :] = 0
    #         K[:, dof] = 0
    #
    #         # Setze Diagonaleintrag auf 1
    #         K[dof, dof] = 1
    #
    #         # Setze rechte Seite auf 0
    #         f[dof] = 0
    #
    #     return K.tocsr(), f  # Optional: Rückgabe in csr-Format

    def update_cps(self, RS, dis):
        for b in RS.basis:
            i = b.id
            b.displacements = np.array(dis[6 * i:6 * i + 3])
            b.dof_values = np.array(dis[6 * i:6 * i + 6])


    def postprocessing(self, RS):
        # Postprocessing
        for el in RS.elements:
            bfs_el = list(el.supported_b_splines)
            bfs_el.sort(key=lambda bspline: bspline.id)
            u_el = np.array([bf.dof_values for bf in bfs_el]).flatten()
            for gauss_xy in el.gausspoints:
                t = gauss_xy.t
                N = np.zeros(3)  # Normal forces: Nx, Ny, Nz
                M = np.zeros(3)  # Moments: Mx, My, Mz
                Q = np.zeros(3)  # Shear force: Qx, Qy, Qz
                for i in range(len(gauss_xy.weight_outerplane)):
                    zeta = gauss_xy.zeta[i]
                    w = gauss_xy.weight_outerplane[i]
                    B = gauss_xy.Bmatrices[i]
                    C_loc = gauss_xy.C_loc
                    R = gauss_xy.rotationmatrices[i]

                    eps_glob = B @ u_el
                    gauss_xy.strain_global.append(eps_glob)

                    eps_loc = R.T @ eps_glob
                    gauss_xy.strain_local.append(eps_loc)

                    sigma_loc = C_loc @ eps_loc
                    gauss_xy.stress_local.append(sigma_loc)

                    # Calculation of shear forces and moments
                    # σ in Voigt notation: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
                    sigma_xx = sigma_loc[0]
                    sigma_yy = sigma_loc[1]
                    sigma_zz = sigma_loc[2]
                    tau_xy = sigma_loc[3]
                    tau_yz = sigma_loc[4]
                    tau_xz = sigma_loc[5]

                    gauss_xy.sigma_vM.append(np.sqrt(
                        sigma_xx ** 2 + sigma_yy ** 2 + sigma_zz ** 2
                        - sigma_xx * sigma_yy - sigma_yy * sigma_zz - sigma_zz * sigma_xx
                        + 3 * (tau_xy ** 2 + tau_yz ** 2 + tau_xz ** 2)
                    ))

                    # Calculation of normal and shear forces (N, Q) and moments (M)
                    A = w * t  # Area, e.g., from Gauss weight and thickness

                    N[0] += sigma_xx * A  # Nx
                    N[1] += sigma_yy * A  # Ny
                    N[2] += sigma_zz * A  # Nz

                    # Calculate moments
                    M[0] += (tau_xy * zeta) * A  # Mx
                    M[1] += (tau_yz * zeta) * A  # My
                    M[2] += (tau_xz * zeta) * A  # Mz

                    # Calculate shear force (assumed calculation)
                    Q[0] += tau_xy * A  # Qx
                    Q[1] += tau_yz * A  # Qy
                    Q[2] += tau_xz * A  # Qz

                gauss_xy.N = N
                gauss_xy.M = M
                gauss_xy.Q = Q


    def get_local_basis(self, RS, xi, eta):
        """
        Computes the covariant basis a1, a2 and a3 at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: [a1, a2, a3]
        """

        def where_uv(xi, eta):
            for el in RS.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        a1 = np.zeros(3)
        a2 = np.zeros(3)
        for f in el.supported_b_splines:
            cp = np.array(f.cp)
            grad = np.array(f.grad(xi, eta))
            a1 += cp * grad[0]  # First tangent vector
            a2 += cp * grad[1]  # Second tangent vector

        a3 = np.cross(a1, a2)
        a3 =a3/np.linalg.norm(a3)   # unit normal vector
        return a1, a2, a3

    def get_local_basis_mixed_variation(self, RS, xi, eta):
        """
        Computes the covariant basis derivative a1,2 = a2,1 at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: a1,2
        """

        def where_uv(xi, eta):
            for el in RS.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        a1_2 = np.zeros(3)
        for f in el.supported_b_splines:
            cp = np.array(f.cp)
            a1_2 += cp * f.grad_mix(xi, eta)

        return a1_2

    def get_local_basis_variation(self, RS, xi, eta):
        """
        Computes the covariant basis derivative a1,1 and a2,2 at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: [a1,1, a2,2]
        """

        def where_uv(xi, eta):
            for el in RS.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        a1_1 = np.zeros(3)
        a2_2 = np.zeros(3)
        for f in el.supported_b_splines:
            cp = np.array(f.cp)
            grad2 = np.array(f.grad2(xi, eta))
            a1_1 += cp * grad2[0]
            a2_2 += cp * grad2[1]

        return a1_1, a2_2

    def get_local_basis_and_variations(self, RS, xi, eta):

        a1, a2, a3 = self.get_local_basis(RS,xi,eta)
        a1_1, a2_2 = self.get_local_basis_variation(RS,xi,eta)
        a1_2 = self.get_local_basis_mixed_variation(RS,xi,eta)
        a2_1 = a1_2

        def get_a3_i(a1_alpha, a2_alpha):

            cross_product = np.cross(a1, a2)
            a3_alpha = ((np.cross(a1_alpha, a2) + np.cross(a1, a2_alpha))*np.linalg.norm(cross_product)-cross_product*((np.dot(cross_product, (np.cross(a1_alpha, a2) + np.cross(a1, a2_alpha)))/np.linalg.norm(cross_product))))/(np.linalg.norm(cross_product) ** 2)

            return a3_alpha

        """
        Computes the covariant derivative of the normal at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        """
        a3_1 = get_a3_i(a1_1, a2_1)
        a3_2 = get_a3_i(a1_2, a2_2)

        return a1, a2, a3, a1_1, a2_2, a1_2, a2_1, a3_1, a3_2





