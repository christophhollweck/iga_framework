import numpy as np
"""
This script creates the BEXT inputdeck for THB-SPlines for LS-Dyna
"""

def write_coordinates_to_txt(coordinates, path, filename):
    filepath = path + '/' + filename
    with open(filepath, 'w') as file:
        for sublist in coordinates:
            line = ''.join([f'{coord:24.16f}' for coord in sublist]) + f'{1:24.16f}' + '\n'
            file.write(line)

def write_matrices_to_txt(matrices_dict, path, filename):
    filepath = path + '/' + filename
    with open(filepath, 'w') as file:
        for key, matrix in matrices_dict.items():
            matrix = matrix.tolist()
            for row in matrix:
                line = ''
                for i, num in enumerate(row):
                    if i > 0 and i % 5 == 0:
                        line += '\n'
                    line += f'{num:24.16f}'
                line += '\n'
                file.write(line)
            file.write('\n')

def write_connectivity(bfnums, path, filename):
    filepath = path + '/' + filename
    k = 1
    with open(filepath, 'w') as file:
        for bfnums_loc in bfnums:
            # Schreibe die Liste bfnums_loc in eine Zeile
            line = ''
            for num in bfnums_loc:
                line += f'{int(num):8d}'  # Formatieren als Integer mit 8 Zeichen
            line += '\n'
            file.write(line)

            # Erzeuge CVID und schreibe sie in die nächste Zeile
            CVID = []
            for i in range(9):
                CVID.append(k)
                k += 1

            line = ''
            for num in CVID:
                line += f'{int(num):8d}'  # Formatieren als Integer mit 8 Zeichen
            line += '\n'
            file.write(line)



def write_kfile(T, path, filename):

    print('write BEXT format...')

    NumCPs = len(T.basis)
    NumEl = len(T.elements)
    NumCoefVec = np.sum([len(e.supported_b_splines_list) for e in T.elements])
    Pol = 0

    bspline_list = list(T.basis)
    # Sort the list by the id attribute
    bspline_list.sort(key=lambda bspline: bspline.id)

    cps = np.array([b.cp for b in bspline_list])
    PR = T.p
    PS = T.p
    Pol_degrees = np.unique([e.p for e in T.elements])
    # NNs = np.unique([[len(e.supported_b_splines_list), e.p] for e in T.elements])
    NNs = np.unique([[len(e.supported_b_splines_list), e.p] for e in T.elements], axis=0)
    #print(NNs)
    NEjs = []
    for NN in NNs:
        NEjs.append(np.sum([1 for e in T.elements if len(e.supported_b_splines_list) == NN[0] and e.p == NN[1]]))
    Num_El_Blocks = len(NNs)
    ETYPE = 1
    NDCVB = Num_El_Blocks
    # Calculate NCVCds and NCVds
    NCVCds = [(deg + 1) ** 2 for deg in NNs[:, 1]]
    NCVds = [x * y for x, y in zip(NEjs, NNs[:, 0])]
    #print(NCVCds, NCVds)
    Coeff_Vec_dic = {}
    for i, NN in enumerate(NNs):
        coeff_list = []
        for e in T.elements:
            if len(e.supported_b_splines_list) == NN[0] and e.p ==NN[1]:
                coeff_list.append(e.BEO)
        Coeff_Vec_dic[i] = coeff_list

    lineA = 'B E X T 2.1' + '\n'
    lineB = 'BASIS TRANSFORM 2.1 THB' + '\n'
    #lineB = 'BASIS TRANSFORM 2.1' + '\n'
    line = f'{int(NumCPs):8d}' + f'{int(NumEl):8d}' + f'{int(NumCoefVec):8d}' + f'{int(Pol):8d}' + '\n'
    for row in cps.tolist():
        line += ''.join([f'{coord:24.16f}' for coord in row]) + f'{1:24.16f}' + '\n'
    line += f'{int(Num_El_Blocks):8d}' + '\n'
    for i in range(Num_El_Blocks):
        line += f'{int(ETYPE):8d}' + f'{int(NEjs[i]):8d}' + f'{int(NNs[i][0]):8d}' + f'{int(NNs[i][0]):8d}' + f'{int(NNs[i][1]):8d}' + f'{int(NNs[i][1]):8d}' + '\n'
    k = 1
    for NN in NNs:
        for e in T.elements:
            if len(e.supported_b_splines_list) == NN[0] and e.p == NN[1] and len(e.supported_b_splines_list) <= 10:  # == NCVCd:
                line += ''.join([
                    f'$ el_id:{int(e.id):5d} ',
                    f'el_p:{int(e.p):2d} ',
                    f'num_bf:{len(e.supported_b_splines_list):2d} ',
                    f'u_min:{float(e.u_min):.16f} ',
                    f'u_max:{float(e.u_max):.16f} ',
                    f'v_min:{float(e.v_min):.16f} ',
                    f'v_max:{float(e.v_max):.16f}'
                ]) + '\n'
                #here
                line += ''.join([f'{coord:24.16f}' for coord in [e.u_min, e.u_max, e.v_min, e.v_max]]) + '\n'
                line += ''.join([f'{int(f.id + 1):8d}' for f in e.supported_b_splines_list]) + '\n'
                line += ''.join([f'{int(i):8d}' for i in range(k, k + NN[0])]) + '\n'
                k += NN[0]

            if len(e.supported_b_splines_list) == NN[0] and e.p == NN[1] and len(e.supported_b_splines_list) > 10:
                # split the line, when len > NCVCd
                supported_b_splines_list = list(e.supported_b_splines)
                # Now you can safely use slicing
                line += ''.join([
                    f'$ el_id:{int(e.id):5d} ',
                    f'el_p:{int(e.p):2d} ',
                    f'num_bf:{len(e.supported_b_splines_list):2d} ',
                    f'u_min:{float(e.u_min):.16f} ',
                    f'u_max:{float(e.u_max):.16f} ',
                    f'v_min:{float(e.v_min):.16f} ',
                    f'v_max:{float(e.v_max):.16f}'
                ]) + '\n'
                #here
                line += ''.join([f'{coord:24.16f}' for coord in [e.u_min, e.u_max, e.v_min, e.v_max]]) + '\n'
                line += ''.join([f'{int(f.id + 1):8d}' for f in e.supported_b_splines_list[:10]]) + '\n'
                line += ''.join([f'{int(f.id + 1):8d}' for f in e.supported_b_splines_list[10:]]) + '\n'
                # line += ''.join([f'{int(f.id + 1):8d}' for f in e.supported_b_splines[:10]]) + '\n'
                # line += ''.join([f'{int(f.id + 1):8d}' for f in e.supported_b_splines[10:]]) + '\n'
                line += ''.join([f'{int(i):8d}' for i in range(k, k + 10)]) + '\n'
                line += ''.join([f'{int(i):8d}' for i in range(k + 10, k + NN[0])]) + '\n'
                k += NN[0]

    line += f'{int(NDCVB):8d}' + '\n'
    for i in range(NDCVB):
        line += f'{int(NCVds[i]):8d}' + f'{int(NCVCds[i]):8d}' + '\n'
    for i in range(NDCVB):
        coeff_list = Coeff_Vec_dic[i]
        for mat in coeff_list:
            for row in mat:
                for i, num in enumerate(row):
                    if i > 0 and i % 5 == 0:
                        line += '\n'
                    line += f'{num:24.16f}'
                line += '\n'



    filepath0 = path + '/' + 'open_' + filename
    filepath1 = path + '/' + filename
    filepath2 = path + '/' + 'open_THB_LSPP.k'

    # Zeilen formatieren
    line1 = "*KEYWORD\n"
    line2 = "*IGA_2D_BASIS_TRANSFORM_XYZ\n"
    line3 = f'{"$# patchid":>10}{"filename":>70}\n'
    line4 = f'{1:>10}{"open_" + filename:>70}\n'
    line5 = "*END\n"

    # Inhalt zusammenfügen
    content = line1 + line2 + line3 + line4 + line5

    # with open(filepath0, 'w') as file:
    #     file.write(lineA + line)
    with open(filepath1, 'w') as file:
        file.write(lineB + line)
    # with open(filepath2, 'w') as file:
    #     file.write(content)

    print('done writing BEXT format')
















