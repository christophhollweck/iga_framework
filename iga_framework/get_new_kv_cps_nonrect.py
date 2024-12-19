'''
This file takes a k-file as input an refines the underlying 2D patch
'''

import splinepy
import numpy as np



def refine_patch(file_path, level, p_new):

    def get_Bspl(kv_u, kv_v, ax, ay, ximax, etamax, conti, degrees):
        '''
        Defines a standard Bspline
        '''
        # knots to insert
        # uins = np.linspace(0, ximax, n_elU+1)
        # uins = uins[1:-1]
        # vins = np.linspace(0, etamax, n_elV+1)
        # vins = vins[1:-1]
        uins = np.unique(kv_u)
        uins = uins[1:-1]
        vins = np.unique(kv_v)
        vins = vins[1:-1]

        if len(degrees) != 2 or degrees[0] != degrees[1]:
            raise ValueError('Only 2D Geometries and homogeneous degrees in u and v-direction supported')
        # degree
        p = degrees[0]

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
        for i in range(p-1):
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

    def refine(kv, level):

        new_kv = np.unique(kv)
        for i in range(level):
            new_kv = np.sort(np.concatenate((new_kv, (np.unique(new_kv)[1:] + np.unique(new_kv)[:-1]) / 2)))

        return np.setdiff1d(new_kv, kv).tolist()

    def format_text(input):

        contains_lists = any(isinstance(element, list) for element in input)
        if contains_lists:
            formatted_text = ''
            for sublist in input:
                for i, num in enumerate(sublist, 1):
                    if np.abs(num) < 1e-9:
                        num = 0.0
                    formatted_text += '{:20.12f}'.format(num)
                    if i % 4 == 0:
                        formatted_text += '\n'

        else:
            formatted_text = ''
            isend = len(input)
            for i, num in enumerate(input, 1):
                formatted_text += '{:20.12f}'.format(num)
                if i % 4 == 0 or i == isend:
                    formatted_text += '\n'

        return formatted_text


    # Keywords, nach denen gesucht wird
    start_keyword = '*IGA_2D_NURBS_XYZ'
    end_keyword = '*IGA_EDGE_XYZ'
    comment = '$'
    line_start = 0
    line_end = 0
    commentlines = []

    # get start and end line of *IGA_2D_NURBS_XYZ
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):  # Starte die Zeilennummerierung bei 1
            if line.startswith(start_keyword):
                line_start = line_num

            # Überprüfe, ob die aktuelle Zeile mit end_keyword beginnt
            elif line.startswith(end_keyword):
                line_end = line_num
                break  # Stoppe die Schleife, da wir alle benötigten Zeilen gefunden haben

    #get lines in *IGA_2D_NURBS_XYZ starting with $ to know where new block begins
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):  # Starte die Zeilennummerierung bei 1
            if line.startswith(comment) and line_start < line_num < line_end:
                commentlines.append(line_num)

    # extract lines
    head_line = commentlines[0]
    uni_line = commentlines[1]
    kv_r_start = commentlines[2]
    kv_s_start = commentlines[3]
    cp_start = commentlines[4]
    # print(head_line, uni_line, kv_r_start, kv_s_start, cp_start)
    kv_s = []
    kv_r = []
    cps = []

    # extract the lines and save in variables
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if line_num == head_line + 1:
                head = line.split()
                head = [int(val) for val in head]
            elif line_num == uni_line + 1:
                uni = line.split()
                uni = [int(val) for val in uni]
            elif kv_r_start < line_num < kv_s_start:
                kv_r.append(line.split())
            elif kv_s_start < line_num < cp_start:
                kv_s.append(line.split())
            elif cp_start < line_num < line_end:
                positionen = [0, 20, 40, 60]
                split = [line[i:j].strip() for i, j in zip(positionen[:-1], positionen[1:])]
                #print(split)
                cps.append(split)

    # write data in lists
    kv_r = [[float(val) for val in sublist] for sublist in kv_r]
    kv_r = [val for sublist in kv_r for val in sublist]

    kv_s = [[float(val) for val in sublist] for sublist in kv_s]
    kv_s = [val for sublist in kv_s for val in sublist]

    cps = [[float(val) for val in sublist] for sublist in cps]
    # print(head, uni, kv_r, kv_s, cps)


    # find xmin, xmax, ymin and ymax
    min_x = max_x = cps[0][0]
    min_y = max_y = cps[0][1]

    for sublist in cps:
        x, y, _ = sublist  # Annahme: Die Reihenfolge der Werte ist x, y, z, w
        # Aktualisiere die minimalen und maximalen Werte für x und y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)


    pid = head[0]
    nr = head[1]
    ns = head[2]
    pr = head[3]
    ps = head[4]


    nelr = len(np.unique(kv_r)) - 1
    nels = len(np.unique(kv_s)) - 1
    ax = max_x - min_x
    ay = max_y - min_y

    # define new Bspline
    # new_pr = pr
    # new_ps = ps
    new_pr = p_new
    new_ps = p_new

    p = new_pr
    # print(pr, ps)
    # print(kv_r, kv_s)
    # print(cps)

    Bspl = splinepy.BSpline(
        degrees=[pr, ps],
        knot_vectors=[kv_r, kv_s],
        control_points=np.array(cps)
    )

    kv_r_ins = refine(kv_r, level)
    kv_s_ins = refine(kv_s, level)

    # ggf. hier ax und ay tauschen
    # Bspl = get_Bspl(kv_r, kv_s, ax, ay, 1, 1, p-1, [new_pr, new_ps])
    if level > 0:
        Bspl.insert_knots(0, kv_r_ins)
        Bspl.insert_knots(1, kv_s_ins)




    #update lines

    new_kv_r = list(Bspl.knot_vectors[0])
    new_kv_s = list(Bspl.knot_vectors[1])

    new_nr = len(np.unique(new_kv_r)) + new_pr - 1
    new_ns = len(np.unique(new_kv_s)) + new_ps - 1

    new_kv_r = format_text(new_kv_r)
    new_kv_s = format_text(new_kv_s)

    new_cps = Bspl.control_points
    new_cps = [cp.tolist() + [1.0] for cp in new_cps]
    # Tausche x und y für richtige Darstellung wie in LS-Dyna
    # führt zu 90 grad rotation
    new_cps = [[sublist[0], sublist[1], *sublist[2:]] for sublist in new_cps]
    new_cps = format_text(new_cps)
    # print(new_cps)


    new_head = [pid, new_nr, new_ns, new_pr, new_ps]
    new_head = ''.join([f'{int(num):10d}' for num in new_head])
    new_uni = ''.join([f'{int(num):10d}' for num in uni])


    new_IGA_2D_NURBS_XYZ = '*IGA_2D_NURBS_XYZ\n' + \
        '$  ID2_XYZ|       NR|       NS|       PR|       PS|\n' + \
        new_head + '\n' + \
        '$     UNIR|     UNIS|\n' + \
        new_uni + '\n' + \
        '$                 R1|                 R2|                 R3|                 R4|\n' + \
        new_kv_r + \
        '$                 S1|                 S2|                 S3|                 S4|\n' + \
        new_kv_s + \
        '$                  X|                  Y|                  Z|                WGT|\n' + \
        new_cps

    # print(new_IGA_2D_NURBS_XYZ)


    # Lese den gesamten Inhalt der Datei ein
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Lösche die Zeilen zwischen cp_start und line_end
    file_content[line_start - 1:line_end - 1] = [new_IGA_2D_NURBS_XYZ]

    return file_content, Bspl



if __name__ == '__main__':

    file_path = r'/media/christoph/Daten1/LS_Dyna/CAD_Models/ellipse.key'
    level = 0

    file_content, Bspl = refine_patch(file_path, level, p_new=2)

    Bspl.show()

    # save_path = r'/media/christoph/Daten1/LS_Dyna/S_Rail_THB_LR/EVP/5_0_test.key'
    # # Schreibe die aktualisierten Zeilen zurück in die Datei
    # with open(save_path, 'w') as file:
    #     file.writelines(file_content)
