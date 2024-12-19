from scipy.linalg import eigh
import numpy as np


class FE_bf():
    def __init__(self):
        self.funs = [self.N1, self.N2, self.N3, self.N4]
        self.derivs = [self.dN1, self.dN2, self.dN3, self.dN4]


    def N1(self, xi, eta):
        return 1/4 * (1 - xi) * (1 - eta)
    def N2(self, xi, eta):
        return 1 / 4 * (1 + xi) * (1 - eta)
    def N3(self, xi, eta):
        return 1 / 4 * (1 - xi) * (1 + eta)
    def N4(self, xi, eta):
        return 1 / 4 * (1 + xi) * (1 + eta)


    def dN1(self, xi, eta):
        return [-1/4*(1 - eta), -1/4*(1 - xi)]
    def dN2(self, xi, eta):
        return [1/4*(1 - eta), -1/4*(1 + xi)]
    def dN3(self, xi, eta):
        return [-1/4*(1 + eta), 1/4*(1 - xi)]
    def dN4(self, xi, eta):
        return [1/4*(1 + eta), 1/4*(1 + xi)]

def build_K_M_fem(HS, t, E, rho, nue):
    # find element area in physical space
    area_minel = 10e-6
    num_minel = -1
    P_minel = []
    for i, el in enumerate(HS.elements):
        AABB = [[el.u_min, el.u_max], [el.v_min, el.v_max]]
        P = []
        for v in AABB[1]:
            for u in AABB[0]:
                P.append(HS(u, v)[0:2])
        dx = P[1][0] - P[0][0]
        dy = P[2][1] - P[0][1]
        area = abs(dx * dy)
        if area > area_minel:
            area_minel = area
            num_minel = i
            P_minel = P

    FE_bfs = FE_bf()
    Cmat = E / (1 - nue ** 2) * np.array([[1, nue, 0], [nue, 1, 0], [0, 0, 0.5 * (1 - nue)]])
    n_el_cps = 4
    bfnums = [0,1,2,3]
    funs = FE_bfs.funs
    derivs = FE_bfs.derivs
    M_el = np.zeros((2 * n_el_cps, 2 * n_el_cps))
    K_el = np.zeros((2 * n_el_cps, 2 * n_el_cps))
    cpsx = [p[0] for p in P_minel]
    cpsy = [p[1] for p in P_minel]
    iwip_el = [[-1/np.sqrt(3), -1/np.sqrt(3), 1],[1/np.sqrt(3), -1/np.sqrt(3), 1],[-1/np.sqrt(3), 1/np.sqrt(3), 1],[1/np.sqrt(3), 1/np.sqrt(3), 1]]
    for iwip in iwip_el:
        gpx = iwip[0]
        gpy = iwip[1]
        gw = iwip[2]
        B = np.zeros((3, 2 * n_el_cps))
        N_mat = np.zeros((2, 2 * n_el_cps))
        dNdxi_mat = np.zeros((2, n_el_cps))
        # bfnum = global bf indices, loc_bfnum = local bf indices starting from 0
        for bfnum, loc_bfnum in zip(bfnums, range(n_el_cps)):
            dNdxi_mat[:, loc_bfnum] = derivs[bfnum](gpx, gpy)
            N_mat[:, 2 * loc_bfnum: 2 * loc_bfnum + 2] = np.identity(2) * \
                                                         funs[bfnum](gpx, gpy)

        dxdxi = np.dot(dNdxi_mat[0, :], cpsx)
        dydxi = np.dot(dNdxi_mat[0, :], cpsy)
        dxdeta = np.dot(dNdxi_mat[1, :], cpsx)
        dydeta = np.dot(dNdxi_mat[1, :], cpsy)

        J = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])
        # print(J)

        dNdx_mat = np.zeros((2, n_el_cps))
        for loc_bfnum in range(len(bfnums)):
            dNdx_mat[:, loc_bfnum] = np.linalg.solve(J, dNdxi_mat[:, loc_bfnum])

        B[0, ::2] = dNdx_mat[0, :]
        B[1, 1::2] = dNdx_mat[1, :]
        B[2, ::2] = dNdx_mat[1, :]
        B[2, 1::2] = dNdx_mat[0, :]

        # print(dNdx_mat, B)

        det_J = np.linalg.det(J)
        K_el = K_el + t * (np.transpose(B) @ Cmat @ B * gw * det_J)
        M_el = M_el + rho * t * (np.transpose(N_mat) @ N_mat * gw * det_J)


    M_el = np.diag(np.sum(M_el, axis=1))
    eigvalues, _ = eigh(K_el, M_el)
    omega = np.sqrt(np.abs(eigvalues))
    # print([om / (2 * np.pi) for om in omega])
    # print(len(omega))

    return np.max(omega)


class membrane():

    def __init__(self, E, t, nue, rho):
        """
        Initialize the membran with its material characteristics.
        :param E, t, nue, rho: Youngs Modulus, thickness, Poissions ratio, density
        """
        self.E = E
        self.t = t
        self.nue = nue
        self.rho = rho
        self.Cmat = E / (1 - nue ** 2) * np.array([[1, nue, 0], [nue, 1, 0], [0, 0, 0.5 * (1 - nue)]])
        self.K_el_dic = {}
        self.M_el_dic = {}


    def build_K_M(self, RS, trimming=False):
        """
        calculation of the stiffness and mass matrices
        :param hierarchical_space: an object of the hierarchical space
        """
        if trimming:
            cells = [cell for cell in RS.elements if len(cell.gausspoints) > 0]
        else:
            cells = RS.elements
            [cell.get_gp_gw_2d(full_quad=True) for cell in cells]

            # get all active functions ids
        afuncs = np.unique([b.id for cell in cells for b in cell.supported_b_splines])
        n_cps = len(RS.basis)
        dof_per_cp = 2
        n_dof_glob = dof_per_cp * n_cps
        nel = len(cells)

        K = np.zeros((n_dof_glob, n_dof_glob))
        M = np.zeros((n_dof_glob, n_dof_glob))

        M_el_dic = {}
        K_el_dic = {}


        for cell in cells:
            n_el_cps = len(cell.supported_b_splines)
            # sort bf ids according to their ids
            bfnums = np.sort([b.id for b in cell.supported_b_splines])
            # sort bf according to their ids
            bfs_loc = list(cell.supported_b_splines)
            bfs_loc.sort(key=lambda bspline: bspline.id)

            K_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
            M_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))

            # cps
            cpsx = [b.cp[0] for b in bfs_loc]
            cpsy = [b.cp[1] for b in bfs_loc]

            for iwip in cell.gausspoints:
                gpx = iwip.xi
                gpy = iwip.eta
                gw = iwip.weight_inplane
                iwip.C_loc = self.Cmat
                iwip.t = self.t

                B = np.zeros((3, 2 * n_el_cps))
                N_mat = np.zeros((2, 2 * n_el_cps))
                dNdxi_mat = np.zeros((2, n_el_cps))

                # bfnum = global bf indices, loc_bfnum = local bf indices starting from 0
                for bf_loc, loc_bfnum in zip(bfs_loc, range(n_el_cps)):
                    dNdxi_mat[:, loc_bfnum] = bf_loc.grad(gpx, gpy)
                    N_mat[:, 2 * loc_bfnum: 2 * loc_bfnum + 2] = np.identity(2) * \
                                                          bf_loc(gpx, gpy)

                dxdxi = np.dot(dNdxi_mat[0, :], cpsx)
                dydxi = np.dot(dNdxi_mat[0, :], cpsy)
                dxdeta = np.dot(dNdxi_mat[1, :], cpsx)
                dydeta = np.dot(dNdxi_mat[1, :], cpsy)

                J = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])
                # print(J)

                dNdx_mat = np.linalg.solve(J, dNdxi_mat)

                B[0, ::2] = dNdx_mat[0, :]
                B[1, 1::2] = dNdx_mat[1, :]
                B[2, ::2] = dNdx_mat[1, :]
                B[2, 1::2] = dNdx_mat[0, :]

                det_J = np.linalg.det(J)

                K_el = K_el + self.t * (np.transpose(B) @ self.Cmat @ B * gw * det_J)
                M_el = M_el + self.rho * self.t * (np.transpose(N_mat) @ N_mat * gw * det_J)
            K_el_dic[cell.id] = K_el
            M_el_dic[cell.id] = M_el

            dofs = np.sort(np.concatenate([bfnums * 2 + i for i in range(2)]))
            K[np.ix_(dofs, dofs)] += K_el
            M[np.ix_(dofs, dofs)] += M_el

        adofs = np.sort(np.concatenate([afuncs * 2 + i for i in range(2)]))
        self.K = K[np.ix_(adofs, adofs)]
        self.M = M[np.ix_(adofs, adofs)]
        self.K_el_dic = K_el_dic
        self.M_el_dic = M_el_dic


    def evp(self, lumping='active'):
        K = self.K
        M = self.M
        # if lumping == 'active':
        #     M = np.diag(np.sum(M, axis=1))
        self.eigvalues, self.eigvectors = eigh(K, M)
        self.omega = np.sqrt(np.abs(self.eigvalues))
        #print(self.omega)
        #print([om/(2*np.pi) for om in self.omega])
        #print(len(self.omega))

        self.omega_el_dic = {}
        for el, K_el, M_el in zip(self.K_el_dic.keys(), self.K_el_dic.values(), self.M_el_dic.values()):
            if lumping == 'active':
                M_el = np.diag(np.sum(M_el, axis=1))
            eigvalues, _ = eigh(K_el, M_el)
            self.omega_el_dic[el] = np.max(np.sqrt(np.abs(eigvalues)))




    # def plot_max_omega_el(self, HS, trimming=False):
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import Normalize
    #     from matplotlib.cm import ScalarMappable
    #
    #     fe_omega = build_K_M_fem(HS, self.t, self.E, self.rho, self.nue)
    #     max_omega = max(self.omega_el_dic.values())
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     plt.axis('equal')
    #     plt.axis('off')
    #     max_afuns = (HS.p + 1) * (HS.p + 1)
    #     if trimming:
    #         cells = [cell for cell in HS.M if len(cell.gausspoints) > 0]
    #         for brep in HS.trimming_curves:
    #             plt.plot(brep.x, brep.y, color='black')
    #     else:
    #         cells = HS.elements
    #
    #     # Normalize the omega values to the range [0, 1]
    #     norm = Normalize(vmin=0, vmax=fe_omega)
    #     cmap = plt.get_cmap('Reds')
    #
    #     for cell, omega in zip(cells, self.omega_el_dic.values()):
    #         omega_norm = omega / fe_omega
    #         cellAABB = np.array(cell.AABB)
    #         x = cellAABB[0, [0, 1, 1, 0, 0]]
    #         y = cellAABB[1, [0, 0, 1, 1, 0]]
    #
    #         # alpha = round(omega_norm, 2) / 6
    #         # plt.fill(x, y, color='red', alpha=alpha)
    #
    #         # Map the normalized omega to a color
    #         color = cmap(norm(omega))
    #         plt.fill(x, y, color=color)
    #
    #         if len(cell.supported_b_splines) > max_afuns:
    #             plt.plot(x, y, color='black', linewidth=2)
    #         else:
    #             plt.plot(x, y, color='black', linewidth=0.3)
    #         # if np.abs(max_omega - omega) < 0.00001:
    #         #     bbox = dict(facecolor='orange', alpha=0.5)
    #
    #             #box = dict(facecolor='red', alpha=0.1)
    #
    #         # Create the colorbar
    #         sm = ScalarMappable(cmap=cmap, norm=norm)
    #         sm.set_array([])
    #         cbar = plt.colorbar(sm, ax=ax)
    #         cbar.set_label('Maximale Eigenfrequenz')
    #
    #         if trimming:
    #             # plt.text(np.mean([ip[0] for ip in cell.IWIP_trimmed]), np.mean([ip[1] for ip in cell.IWIP_trimmed]), round(omega_norm, 2), color='black', ha='center', va='center', bbox=bbox)
    #             #plt.text(np.mean([ip[0] for ip in cell.IWIP_trimmed]), np.mean([ip[1] for ip in cell.IWIP_trimmed]), round(omega_norm, 2), color='black', ha='center', va='center')
    #             pass
    #         else:
    #             #plt.text(np.mean(cellAABB[0]), np.mean(cellAABB[1]), round(omega_norm, 2), color='black', ha='center', va='center', bbox=bbox)
    #             #plt.text(np.mean(cellAABB[0]), np.mean(cellAABB[1]), round(omega_norm, 2), color='black', ha='center', va='center')
    #             pass
    #
    #     #import tikzplotlib
    #     #tikzplotlib.save("/media/christoph/Daten1/Python_scripts/paper_figs/timestep/THB_overload_2D_p2.tex")
    #     #plt.savefig("/media/christoph/Daten1/Python_scripts/paper_figs/timestep/LR_relax_int_overload_2D_p3.svg", bbox_inches='tight')
    #     plt.show()


    def plot_max_omega_el(self, HS, trimming=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import numpy as np

        fe_omega = build_K_M_fem(HS, self.t, self.E, self.rho, self.nue)
        print(fe_omega)
        max_omega = max(self.omega_el_dic.values())
        fig, ax = plt.subplots(figsize=(8, 8))  # Hier plt.subplots verwenden
        ax.set_aspect('equal')
        ax.axis('off')
        max_afuns = (HS.p + 1)**2

        if trimming:
            cells = [cell for cell in HS.elements if len(cell.gausspoints) > 0]
            for brep in HS.trimming_curves:
                ax.plot(brep.x, brep.y, color='black')
        else:
            cells = HS.elements

        # Normalize the omega values to the range [0, 1]
        norm = Normalize(vmin=0.5, vmax=4)
        cmap = plt.get_cmap('Purples')

        for cell, omega in zip(cells, self.omega_el_dic.values()):
            omegas_ratio = omega / fe_omega
            omega_norm = norm(omega / fe_omega)
            #print(omega_norm)
            AABB = [[cell.u_min, cell.u_max], [cell.v_min, cell.v_max]]
            cellAABB = np.array(AABB)
            x = cellAABB[0, [0, 1, 1, 0, 0]]
            y = cellAABB[1, [0, 0, 1, 1, 0]]

            # Map the normalized omega to a color
            color = cmap(omega_norm)
            ax.fill(x, y, color=color)

            if len(cell.supported_b_splines) > max_afuns:
                # ax.plot(x, y, color='black', linewidth=2)
                ax.plot(x, y, color='black', linewidth=0.5)
            else:
                ax.plot(x, y, color='black', linewidth=0.5)

            plt.text(np.mean(cellAABB[0]), np.mean(cellAABB[1]), round(omegas_ratio, 2), color='black', ha='center',
                     va='center')

        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Maximale Eigenfrequenz')

        #import tikzplotlib
        # tikzplotlib.save("/media/christoph/Daten1/Python_scripts/paper_figs/timestep/THB_overload_2D_p2.tex")
        #plt.savefig("/media/christoph/Daten1/Python_scripts/paper_figs/inkscape/LR/Gengar_LR_2D_p2.svg", bbox_inches='tight')
        plt.show()



    def update_cps(self, RS, dis):
        for b in RS.basis:
            i = b.id
            b.displacements[:2] = np.array(dis[2*i:2*i+2])
            b.dof_values = np.array(dis[2*i:2*i+2])