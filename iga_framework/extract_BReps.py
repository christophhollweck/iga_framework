import numpy as np
import splinepy
import os
import matplotlib.pyplot as plt



class BSpline:
    def __init__(self, NR, kv, cps, weights, ID, PR):
        self.NR = NR
        self.kv = kv
        self.cps = cps
        self.weights = weights
        self.ID = ID
        self.PR = PR
        self.obj = splinepy.NURBS(
                degrees=[self.PR],
                knot_vectors=[self.kv],
                control_points=np.array(self.cps),
                weights=np.array(self.weights)
            )

class BRep:
    def __init__(self, Brep_id):
        self.ID = Brep_id
        self.bsplines = []

    def eval_BRep(self):
        xlist = []
        ylist = []
        for Bspl in self.bsplines:
            uvec = np.linspace(0, 1, 100)
            for u in uvec:
                xy = Bspl.obj.evaluate([[u]])[0]
                xlist.append(xy[0])
                ylist.append(xy[1])

        return xlist, ylist

def get_BRep(file_path):

    # Keywords, nach denen gesucht wird
    keyword = '*'
    keyword_IGA_1D = '*IGA_1D_NURBS_UVW'
    comment = '$'
    Brep1d = '*IGA_1D_BREP'
    keywordlines = []
    keyword_IGA_1D_lines = []
    commentlines = []
    Brep1d_lines = []

    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if line.startswith(keyword):
                keywordlines.append(line_num)
            if line.startswith(comment):
                commentlines.append(line_num)
            if line.startswith(keyword_IGA_1D):
                keyword_IGA_1D_lines.append(line_num)
            if line.startswith(Brep1d):
                Brep1d_lines.append(line_num)

    dic_IGA_line = {}

    for IGA_line in keyword_IGA_1D_lines:
        dic_IGA_line[IGA_line] = []
        for commentline in commentlines:
            if len(dic_IGA_line[IGA_line]) < 4 and commentline > IGA_line:
                dic_IGA_line[IGA_line].append(commentline)

    dic_kv = {}
    for i, sec in enumerate(dic_IGA_line.values(), 1):
        dic_kv[i] = []
        start_line, end_line = sec[2], sec[3]
        # print(start_line, end_line)
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if start_line < line_num < end_line:
                    positionen = [0, 20, 40, 60, 80]
                    split = [line[i:j].strip() for i, j in zip(positionen[:-1], positionen[1:])]
                    split = [s for s in split if s]
                    # print(split)
                    dic_kv[i].extend(map(float, split))


    for kv in dic_kv.values():
        while abs(kv[-1] - 1) > 1e-10:
            kv.pop()

    dic_head = {}
    for i, sec in enumerate(dic_IGA_line.values(), 1):
        dic_head[i] = []
        start_line, end_line = sec[0], sec[1]
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if start_line < line_num < end_line:
                    # positionen = [0, 20, 40, 60]
                    # split = [line[i:j].strip() for i, j in zip(positionen[:-1], positionen[1:])]
                    # split = [s for s in split if s]
                    dic_head[i].extend(map(int, line.split()))

    dic_cps = {}
    dic_weights = {}
    for i, sec in enumerate(dic_IGA_line.values(), 1):
        dic_cps[i] = []
        dic_weights[i] = []
        start_line, end_line = sec[3], sec[3] + dic_head[i][1]
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if start_line < line_num <= end_line:
                    dic_cps[i].append([float(num) for num in line.split()[:2]])
                    dic_weights[i].append([float(num) for num in line.split()[2:]])


    dic_brep = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() == '*IGA_1D_BREP':
            # Nächste Zeile enthält ID1DBR
            id1dbr = int(lines[i + 2].strip())

            # Initialisiere die UVW-Werte Liste
            uvw_values = []
            i += 4  # Gehe zur ersten Zeile mit UVW-Werten

            while i < len(lines) and not lines[i].strip().startswith('*IGA_1D_BREP'):
                line_values = lines[i].strip().split()
                # Filtern Sie nur numerische Werte und fügen Sie sie der Liste hinzu
                try:
                    uvw_values += [int(x) for x in line_values]
                except ValueError:
                    break  # Beendet die Schleife, wenn keine numerischen Werte mehr vorhanden sind
                i += 1

            dic_brep[id1dbr] = uvw_values
        else:
            i += 1


    Bspl = []
    for head, kv, cps, weights in zip(dic_head.values(), dic_kv.values(), dic_cps.values(), dic_weights.values()):
        NR = head[1]
        kv = np.sort(kv).tolist()
        cps = cps
        ID = head[0]
        PR = head[2]
        Bspl.append(BSpline(NR, kv, cps, weights, ID, PR))

    Breps = [BRep(Brep_id) for Brep_id in dic_brep.keys()]
    for Brep in Breps:
        for b in Bspl:
            if b.ID in dic_brep[Brep.ID]:
                Brep.bsplines.append(b)

    return Breps

def plot_BRep(BReps):
    for brep in BReps:
        x, y = brep.eval_BRep()
        plt.plot(x, y)
    plt.show()


if __name__ == '__main__':


    poldeg = 2
    # # Dateipfad
    # name = "tensile"
    # main_path = f'/media/christoph/Daten1/LS_Dyna/EVP/{name}/{name}_p{poldeg}/{name}_p{poldeg}_level0'
    # file_path = os.path.join(main_path, f'{name}_p{poldeg}_level0.key')
    # Bspl2D_path = os.path.join(main_path, 'Bspl2D.pkl')

    file_path = '/media/christoph/Daten/LS_Dyna/EVP/gengar/gengar_p2/gengar_p2_level0/gengar_p2_level0.key'

    BReps = get_BRep(file_path)
    plot_BRep(BReps)



    pass
    # with open(path_obj, 'wb') as file:
    #     pickle.dump(Brep, file)


