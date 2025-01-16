import numpy as np
from scipy.special import roots_legendre

class HeatEquation:
    def __init__(self, HS, dirichlet, neumann, analytical_solution=None):
        self.HS = HS
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.analytical_solution = analytical_solution
        self.K = None
        self.f = None

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        self.min_u = None
        self.max_u = None
        self.min_v = None
        self.max_v = None

        self._get_min_max_cps_coords()
        self._get_min_max_uv()

    def _get_min_max_cps_coords(self):
        x_coords = [bf.cp[0] for bf in self.HS.basis]
        y_coords = [bf.cp[1] for bf in self.HS.basis]

        self.min_x = min(x_coords)
        self.max_x = max(x_coords)
        self.min_y = min(y_coords)
        self.max_y = max(y_coords)

    def _get_min_max_uv(self):
        self.min_u = self.HS.spaces[0].ku[0]
        self.max_u = self.HS.spaces[0].ku[-1]
        self.min_v = self.HS.spaces[0].kv[0]
        self.max_v = self.HS.spaces[0].kv[-1]

    def collect_basis_ids(self, position='left', threshold=0.01):
        """
        Returns the basis function IDs based on the specified position or corner.

        :param position: str, can be 'left', 'right', 'bottom', 'top',
                         'corner1', 'corner2', 'corner3', 'corner4'
        :param threshold: float, the threshold for checking if a control point is near the boundary or corner
        :return: list of basis function IDs
        """
        ids = []

        for basis_function in self.HS.basis:
            control_point = basis_function.cp

            if position == 'left':
                if abs(control_point[0] - self.min_x) < threshold:
                    ids.append(basis_function.id)
            elif position == 'right':
                if abs(control_point[0] - self.max_x) < threshold:
                    ids.append(basis_function.id)
            elif position == 'bottom':
                if abs(control_point[1] - self.min_y) < threshold:
                    ids.append(basis_function.id)
            elif position == 'top':
                if abs(control_point[1] - self.max_y) < threshold:
                    ids.append(basis_function.id)
            elif position == 'corner1':
                if (abs(control_point[0] - self.min_x) < threshold and
                        abs(control_point[1] - self.min_y) < threshold):
                    ids.append(basis_function.id)
            elif position == 'corner2':
                if (abs(control_point[0] - self.max_x) < threshold and
                        abs(control_point[1] - self.min_y) < threshold):
                    ids.append(basis_function.id)
            elif position == 'corner3':
                if (abs(control_point[0] - self.min_x) < threshold and
                        abs(control_point[1] - self.max_y) < threshold):
                    ids.append(basis_function.id)
            elif position == 'corner4':
                if (abs(control_point[0] - self.max_x) < threshold and
                        abs(control_point[1] - self.max_y) < threshold):
                    ids.append(basis_function.id)
            else:
                raise ValueError(
                    "Parameter 'position' muss entweder 'left', 'right', 'bottom', 'top', 'corner1', 'corner2', 'corner3' oder 'corner4' sein.")

        return ids

    def build_K(self):
        cells = self.HS.elements
        n_cps = len(self.HS.basis)
        n_dof_glob = n_cps
        K = np.zeros((n_dof_glob, n_dof_glob))

        for cell in cells:
            n_el_cps = len(cell.supported_b_splines_after_ref)
            bfnums = [b.id for b in cell.supported_b_splines_after_ref]
            bfs_loc = cell.supported_b_splines_after_ref

            cell.get_gp_gw_2d()
            iwip_el = cell.IWIP_untrimmed
            K_el = np.zeros((n_el_cps, n_el_cps))
            cpsx = [b.cp[0] for b in cell.supported_b_splines_after_ref]
            cpsy = [b.cp[1] for b in cell.supported_b_splines_after_ref]

            for iwip in iwip_el:
                gpx = iwip[0]
                gpy = iwip[1]
                gw = iwip[2]

                dNdxi_mat = np.zeros((2, n_el_cps))

                for bf_loc, loc_bfnum in zip(bfs_loc, range(n_el_cps)):
                    dNdxi_mat[:, loc_bfnum] = bf_loc.grad(gpx, gpy)

                dxdxi = np.dot(dNdxi_mat[0, :], cpsx)
                dydxi = np.dot(dNdxi_mat[0, :], cpsy)
                dxdeta = np.dot(dNdxi_mat[1, :], cpsx)
                dydeta = np.dot(dNdxi_mat[1, :], cpsy)

                J = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])
                det_J = np.linalg.det(J)
                print(det_J)

                dNdx_mat = np.zeros((2, n_el_cps))
                for loc_bfnum in range(len(bfs_loc)):
                    dNdx_mat[:, loc_bfnum] = np.linalg.solve(J, dNdxi_mat[:, loc_bfnum])

                # B = np.zeros((2, n_el_cps))
                # B[0, :] = dNdx_mat[0, :]
                # B[1, :] = dNdx_mat[1, :]

                # K_el += (np.transpose(B) @ B * gw * det_J)
                K_el += (np.transpose(dNdx_mat) @ dNdx_mat * gw * det_J)

            bfnums = np.array(bfnums)
            dofs = bfnums
            K[np.ix_(dofs, dofs)] += K_el

        return K

    def integrate_neumann_bc(self, q):
        cells = self.HS.elements
        rhs = np.zeros(len(self.HS.basis))

        for cell in cells:
            n_el_cps = len(cell.supported_b_splines_after_ref)
            bfnums = [b.id for b in cell.supported_b_splines_after_ref]
            bfs_loc = cell.supported_b_splines_after_ref

            cell.get_gp_gw_2d()
            iwip_el = cell.IWIP_untrimmed
            f_el = np.zeros(n_el_cps)
            cpsx = [b.cp[0] for b in cell.supported_b_splines_after_ref]
            cpsy = [b.cp[1] for b in cell.supported_b_splines_after_ref]

            for iwip in iwip_el:
                gpx = iwip[0]
                gpy = iwip[1]
                gw = iwip[2]

                N = np.array([bf(gpx, gpy) for bf in bfs_loc])

                J = np.zeros((2, 2))
                J[0, 0] = np.dot([bf.grad(gpx, gpy)[0] for bf in bfs_loc], cpsx)
                J[0, 1] = np.dot([bf.grad(gpx, gpy)[0] for bf in bfs_loc], cpsy)
                J[1, 0] = np.dot([bf.grad(gpx, gpy)[1] for bf in bfs_loc], cpsx)
                J[1, 1] = np.dot([bf.grad(gpx, gpy)[1] for bf in bfs_loc], cpsy)
                det_J = np.linalg.det(J)
                print(det_J)

                gp_ph = self.HS(gpx, gpy)
                f_el += N * q(gp_ph[0], gp_ph[1]) * gw * det_J

            bfnums = np.array(bfnums)
            dofs = bfnums
            rhs[dofs] += f_el
            print(rhs)

        return rhs

    def set_bc(self):
        dir_dof = []
        for edge in self.dirichlet:
            dir_dof.extend(self.collect_basis_ids(edge, 0.01))

        K = self.build_K()
        f = self.integrate_neumann_bc(self.neumann)

        K[dir_dof, :] = 0
        K[:, dir_dof] = 0
        K[dir_dof, dir_dof] = 1
        f[dir_dof] = 0

        return K, f

    def solve(self):
        K, f = self.set_bc()
        t = np.linalg.solve(K, f)
        for b, val in zip(self.HS.basis, t):
            b.cp[2] = val
