from scipy.linalg import eigh
import numpy as np
# from autograd import grad, jacobian, hessian
import numpy.polynomial.legendre as gq


class nonlinear_membran():

    def __init__(self, E, nu, t, P, RS):
        """
        Initialize the membran with its material characteristics.
        :param E, t, nu, P: Youngs Modulus, thickness, Poissions ratio, load
        """
        self.E = E
        self.t = t
        self.nu = nu
        # load vector [x,y]
        self.P = P
        self.RS = RS

        # # initial displacement and position for every cp
        # for bf in self.RS.basis:
        #     bf.cp = bf.cp[:2]
        #     bf.u_cp = np.array([0.0, 0.0])


    def dXdxi_eval(self, Ngrad, q):

        qx = q[::2]
        qy = q[1::2]
        qq = np.zeros((len(qx), 2))
        qq[:,0] = qx
        qq[:,1] = qy
        return np.dot(Ngrad, qq)  # Jacobian matrix

    def isoparametric_concept(self, N_grad, J):
        return np.linalg.solve(J, N_grad)


    def strain_eval(self, dNxdx, dNydx, dNxdy, dNydy, u_cps):

        ux_x = np.dot(dNxdx, u_cps)
        ux_y = np.dot(dNxdy, u_cps)
        uy_x = np.dot(dNydx, u_cps)
        uy_y = np.dot(dNydy, u_cps)

        grad_u = np.array([[ux_x, ux_y],[uy_x, uy_y]])

        E11 = ux_x + 0.5 * (ux_x**2 + uy_x**2)
        E22 = uy_y + 0.5 * (ux_y**2 + uy_y**2)
        E12 = ux_y + uy_x + ux_x * ux_y + uy_x * uy_y

        return np.array([E11, E22, E12]), grad_u

    def material_tangent_2d(self, E, nu):
        """
        Calculates the material tangent matrix in Voigt notation for the 2D plane stress case.

        Args:
            E: Young's modulus (scalar).
            nu: Poisson's ratio (scalar).

        Returns:
            C_voigt: A 3x3 numpy array representing the material tangent matrix in Voigt notation.
        """
        # Material tangent matrix in Voigt notation for plane stress
        factor = E / (1 - nu ** 2)
        C = factor * np.array([[1, nu, 0],
                                     [nu, 1, 0],
                                     [0, 0, (1 - nu) / 2]])

        return C

    def stress_eval(self, eps, E=1, nu=0.3):

        # Calculate the material tangent (stiffness matrix) for plane stress
        C = self.material_tangent_2d(E, nu)

        # Calculate the stress by multiplying the material tangent with the strain
        stress = np.dot(C, eps)

        return stress

    def strain_variation(self, dNxdx, dNydx, dNxdy, dNydy, u_cps):

        ux_x = np.dot(dNxdx, u_cps)
        ux_y = np.dot(dNxdy, u_cps)
        uy_x = np.dot(dNydx, u_cps)
        uy_y = np.dot(dNydy, u_cps)

        dE11 = dNxdx + ux_x * dNxdx + uy_x * dNydx
        dE22 = dNydy + ux_y * dNxdy + uy_y * dNydy
        dE12 = (1 + ux_x) * dNxdy + (1 + uy_y) * dNydx + ux_y * dNxdx + uy_x * dNydy

        return dE11, dE22, dE12

    def stress_variation(self, dE11, dE22, dE12, E=1, nu=0.3):

        C = self.material_tangent_2d(E, nu)

        deps = np.zeros((3, len(dE11)))
        deps[0,:] = dE11
        deps[1,:] = dE22
        deps[2,:] = dE12

        dsig = np.dot(C, deps)

        dsig11 = dsig[0, :]
        dsig22 = dsig[1, :]
        dsig12 = dsig[2, :]

        return dsig11, dsig22, dsig12


    def strain_variation2(self, dNxdx, dNydx, dNxdy, dNydy):

        ddE11 = np.outer(dNxdx, dNxdx) + np.outer(dNydx, dNydx)
        ddE22 = np.outer(dNxdy, dNxdy) + np.outer(dNydy, dNydy)
        ddE12 = np.outer(dNxdx, dNxdy) + np.outer(dNxdy, dNxdx) + np.outer(dNydx, dNydy) + np.outer(dNydy, dNydx)

        return ddE11, ddE22, ddE12


    def build_K_R(self, trimming=False):
        """
        calculation of the stiffness matrix
        """
        RS = self.RS
        if trimming:
            cells = [cell for cell in RS.elements if len(cell.gausspoints) > 0]
        else:
            cells = RS.elements
            [cell.get_gp_gw_2d() for cell in cells]
        nel = len(cells)
        # get all active functions ids
        afuncs = np.unique([b.id for cell in cells for b in cell.supported_b_splines])
        n_cps = len(RS.basis)
        dof_per_cp = 2
        n_dof_glob = dof_per_cp * n_cps
        K = np.zeros((n_dof_glob, n_dof_glob))
        R = np.zeros(n_dof_glob)

        t = self.t
        E = self.E
        nu = self.nu
        P = self.P

        for cell in cells:
            n_el_cps = len(cell.supported_b_splines)
            # sort bf ids according to their ids
            bfnums = np.sort([b.id for b in cell.supported_b_splines])
            # sort bf according to their ids
            bfs_loc = list(cell.supported_b_splines)
            bfs_loc.sort(key=lambda bspline: bspline.id)

            K_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
            R_el = np.zeros(dof_per_cp * n_el_cps)

            # q_cps = [x1,y1,...,xn,yn]
            q_cps = np.array([f.cp[:2] for f in bfs_loc]).flatten()
            # u_cps = [ux1,uy1,...,uxn,uyn]
            u_cps = np.array([f.displacements[:2] for f in bfs_loc]).flatten()

            # inplane loop
            for iwip in cell.gausspoints:
                gpx = iwip.xi
                gpy = iwip.eta
                gw = iwip.weight_inplane

                N = np.array([bf(gpx, gpy) for bf in bfs_loc])  # Basis function values
                Nx = np.array([[Ni, 0] for Ni in N]).flatten() # -> ux = Nx*u_cps
                Ny = np.array([[0, Ni] for Ni in N]).flatten() # -> uy = Ny*u_cps

                N_grad = np.array([bf.grad(gpx, gpy) for bf in bfs_loc]).T  # Gradients of basis functions
                J = self.dXdxi_eval(N_grad, q_cps)  # Jacobian
                dN = self.isoparametric_concept(N_grad, J) # derivatives of the bf w.r.t. x,y; dim = (2,n)
                dNdx = dN[0,:]
                dNdy = dN[1,:]
                # built some vectors to calculate displacement gradient one-by-one
                # insert zeros to "filter" x and y values, respectively
                dNxdx = np.array([[dNi, 0] for dNi in dNdx]).flatten() # -> duxdx = dNxdx * u_cps
                dNydx = np.array([[0, dNi] for dNi in dNdx]).flatten() # -> duydx = dNydx * u_cps
                dNxdy = np.array([[dNi, 0] for dNi in dNdy]).flatten() # -> duxdy = dNxdy * u_cps
                dNydy = np.array([[0, dNi] for dNi in dNdy]).flatten() # -> duydy = dNxdy * u_cps

                E_voigt, grad_u = self.strain_eval(dNxdx, dNydx, dNxdy, dNydy, u_cps)  # Dehnung
                dE11, dE22, dE12 = self.strain_variation(dNxdx, dNydx, dNxdy, dNydy, u_cps)
                ddE11, ddE22, ddE12 = self.strain_variation2(dNxdx, dNydx, dNxdy, dNydy)
                sig = self.stress_eval(E_voigt, E, nu)
                iwip.stress_local = sig
                iwip.grad_u = grad_u
                dsig11, dsig22, dsig12 = self.stress_variation(dE11, dE22, dE12, E, nu)
                dux = Nx
                duy = Ny
                det_J = np.linalg.det(J)

                sig11 = sig[0]
                sig22 = sig[1]
                sig12 = sig[2]

                R_el += ( - (sig11*dE11 + sig22*dE22 + sig12*dE12) + P[0]*dux + P[1]*duy) * det_J * gw
                K_el += - (np.outer(dsig11,dE11) + np.outer(dsig22,dE22) + np.outer(dsig12,dE12) + sig11*ddE11 + sig22*ddE22 + sig12*ddE12) * det_J * gw

            # K_el -> K
            dofs = np.sort(np.concatenate([bfnums * dof_per_cp + i for i in range(dof_per_cp)]))
            K[np.ix_(dofs, dofs)] += K_el
            # f_el -> f
            R[np.ix_(dofs)] += R_el
        return K,R

    def find_control_point_ids_on_edges(self, xmin=False, xmax=False, ymin=False, ymax=False):
        # Step 1: Find min and max values for x and y among all control points
        RS = self.RS
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
        # x, y
        fixed_DOFs = {
            'x': None,
            'y': None
        }
        for i, key in enumerate(fixed_DOFs.keys()):
            fixed_DOFs[key] = [2 * cp + i for cp in cps_on_edges]

        total_fixed_DOFs = []
        for fixed_dir in fixed_dirs:
            total_fixed_DOFs.extend(fixed_DOFs[fixed_dir])

        total_fixed_DOFs = np.sort(total_fixed_DOFs)

        return total_fixed_DOFs


    def set_dirichlet_bc(self, K, f, fixed_DOFs):

        # essential bc
        for dof in fixed_DOFs:
            K[:, dof] = 0
            K[dof, :] = 0
            K[dof, dof] = 1
            f[dof] = 0

        return K, f

    def newton_raphson(self, fixed_dofs, tolerance=1e-6, max_iterations=50):
        RS = self.RS
        for iteration in range(max_iterations):
            K, R = self.build_K_R()
            Kd, Rd = self.set_dirichlet_bc(K, R, fixed_dofs)
            delta_u = np.linalg.solve(Kd, -Rd)
            #print(delta_u)
            for b in RS.basis:
                i = b.id
                b.displacements[:2] += np.array(delta_u[2*i:2*i+2])
            print(f"Iteration {iteration + 1}, Residuum Norm: {np.linalg.norm(R)}")
            if np.linalg.norm(R) < tolerance:
                break


    def postprocessing(self, RS):
        # Postprocessing
        for el in RS.elements:
            for gauss_xy in el.gausspoints:
                # σ in Voigt notation: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
                S_xx = gauss_xy.stress_local[0]
                S_yy = gauss_xy.stress_local[1]
                S_zz = 0
                S_xy = gauss_xy.stress_local[2]
                S_yz = 0
                S_xz = 0

                #tranformation of 2PK stress S to cauchy stress sig
                S = np.array([[S_xx, S_xy],[S_xy, S_yy]])
                # Deformationgradient F = I + grad_u
                F = np.eye(2) + gauss_xy.grad_u
                J = np.linalg.det(F)  # Determinante von F
                if J < 0:
                    raise ValueError("invalid Deformationgradient: J must be strictly positive!")
                sigma = (1 / J) * np.dot(F, np.dot(S, F.T))  # Push-Forward-Operation

                sigma_xx = sigma[0, 0]
                sigma_yy = sigma[1, 1]
                sigma_zz = 0
                tau_xy = sigma[0, 1]
                tau_yz = 0
                tau_xz = 0

                gauss_xy.sigma_vM.append(np.sqrt(
                    sigma_xx ** 2 + sigma_yy ** 2 + sigma_zz ** 2
                    - sigma_xx * sigma_yy - sigma_yy * sigma_zz - sigma_zz * sigma_xx
                    + 3 * (tau_xy ** 2 + tau_yz ** 2 + tau_xz ** 2)
                ))

