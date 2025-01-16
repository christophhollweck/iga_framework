import splinepy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

class Brep:

    def __init__(self, bsplines: list, Brep_id: int, elements: list, outerloop = False) -> None:

        self.Brep_id = Brep_id
        self.outerloop = outerloop
        self.bsplines = bsplines
        self.bspline_ids = []
        self.split_bsplines = set()

        self.polygon = None
        self.get_polygon()

        self.bspline_lines = []
        self.get_bspline_lines()
        self.cut_elements = set()
        self.inside_elements = set()
        self.check_elements(elements)
        self.intersection_points = set()
        self.split_Brep(self.cut_elements)



    def get_polygon(self):
        all_points = []
        for bspl in self.bsplines:
            evals = self.eval_bspl(bspl, n=1000)
            # Liste der Punkte aus evals erstellen
            points = [Point(x, y) for x, y in evals]
            all_points.extend(points)
        self.polygon = Polygon(all_points)

    def get_bspline_lines(self):
        for bspl in self.bsplines:
            evals = self.eval_bspl(bspl, n=1000)
            # Liste der Punkte aus evals erstellen
            points = [Point(x, y) for x, y in evals]
            self.bspline_lines.append(LineString(points))

    def check_elements(self, elements):
        for el in elements:
            el_box = box(el.u_min, el.v_min, el.u_max, el.v_max)
            if self.polygon.contains(el_box):
                self.inside_elements.add(el)
            if self.polygon.boundary.intersects(el_box.boundary):
                self.cut_elements.add(el)

    def update_elements(self):
        new_inside_elements = set()
        new_cut_elements = set()
        inside_elements_to_remove = set()
        cut_elements_to_remove = set()
        new_split_bsplines = set()
        split_bsplines_to_remove = set()

        for el in self.inside_elements:
            if not el.active:
                inside_elements_to_remove.add(el)
                children = self.get_children(el)
                new_inside_elements.update(children)
        self.inside_elements.difference_update(inside_elements_to_remove)
        self.inside_elements.update(new_inside_elements)

        for el in self.cut_elements:
            if not el.active:
                split_bsplines_to_remove.update(el.split_bsplines)
                cut_elements_to_remove.add(el)
                children = self.get_children(el)
                new_cut_elements.update(children)
        self.split_bsplines.difference_update(split_bsplines_to_remove)

        new_cut_elements_purged = set()
        for el in new_cut_elements:
            el_box = box(el.u_min, el.v_min, el.u_max, el.v_max)
            if self.polygon.boundary.intersects(el_box.boundary):
                new_cut_elements_purged.add(el)
                self.cut_elements.add(el)
            if self.polygon.contains(el_box):
                self.inside_elements.add(el)

        self.split_Brep(new_cut_elements_purged)


    def get_children(self, element):
        active_children = []
        # Funktion, die rekursiv alle aktiven Kinder findet
        def recursive_collect(element):
            if element.active:
                active_children.append(element)
            else:
                for child in element.children:
                    if child.active:
                        active_children.append(child)
                    else:
                        recursive_collect(child)

        # Startet die rekursive Sammlung
        recursive_collect(element)
        return active_children


    def split_Brep(self, elements_set):
        for el in elements_set:
            el_box = box(el.u_min, el.v_min, el.u_max, el.v_max)
            for bspl_line, bspl in zip(self.bspline_lines, self.bsplines):
                intersection = bspl_line.intersection(el_box.boundary)
                intersection_points_temp = []
                if intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        intersection_points_temp.append((point.x, point.y))
                        self.intersection_points.add((point.x, point.y))
                    self.split_bspl(bspl, el, intersection_points_temp)
                elif intersection.geom_type == 'Point':
                    point = intersection
                    intersection_points_temp.append((point.x, point.y))
                    self.intersection_points.add((point.x, point.y))
                    self.split_bspl(bspl, el, intersection_points_temp)


    def split_bspl(self, bspl, el, intersection_points_temp, ncps=3):
        inverse = np.unique(np.sort([a[0] for a in bspl.proximities(np.array(intersection_points_temp))]))
        Brep_split_list = []
        start = bspl.evaluate([[0]])[0]
        end = bspl.evaluate([[1]])[0]
        if len(inverse) > 1:
            if el.contains(start[0], start[1]) and el.contains(end[0], end[1]):
                evals_slice0 = self.eval_bspl(bspl, 0, inverse[0], n=ncps)
                evals_slice1 = self.eval_bspl(bspl, inverse[1], 1, n=ncps)
                curve_new0, _ = splinepy.helpme.fit.curve(evals_slice0, degree=2)
                curve_new1, _ = splinepy.helpme.fit.curve(evals_slice1, degree=2)
                el.split_bsplines.append(curve_new0)
                self.split_bsplines.add(curve_new0)
                el.split_bsplines.append(curve_new1)
                self.split_bsplines.add(curve_new1)

            else:
                for i in range(len(inverse) - 1):
                    evals_slice = self.eval_bspl(bspl, inverse[i], inverse[i + 1], n=ncps)
                    curve_new, _ = splinepy.helpme.fit.curve(evals_slice, degree=2)
                    eval = curve_new.evaluate([[0.5]])
                    if el.contains(eval[0][0], eval[0][1]):
                        el.split_bsplines.append(curve_new)
                        self.split_bsplines.add(curve_new)
        else:
            if el.contains(start[0], start[1]):
                evals_slice = self.eval_bspl(bspl, 0, inverse[0], n=ncps)
                curve_new, _ = splinepy.helpme.fit.curve(evals_slice, degree=2)
                el.split_bsplines.append(curve_new)
                self.split_bsplines.add(curve_new)

            elif el.contains(end[0], end[1]):
                evals_slice = self.eval_bspl(bspl, inverse[0], 1, n=ncps)
                curve_new, _ = splinepy.helpme.fit.curve(evals_slice, degree=2)
                el.split_bsplines.append(curve_new)
                self.split_bsplines.add(curve_new)

    def eval_bspl(self, curve, eta_min=None, eta_max=None, n=5):
        # evaluate a Bspl. Object
        if eta_min is None:
            eta_min = curve.kvs[0][0]
        if eta_max is None:
            eta_max = curve.kvs[0][-1]

        etas = np.linspace(eta_min, eta_max, n)
        etas = [[eta] for eta in etas]
        evals = curve.evaluate(etas)

        return evals

    def plot_Brep(self, elements):
        # Plot erstellen
        fig, ax = plt.subplots(figsize=(8, 8))
        for el in elements:
            u_min, v_min = el.u_min, el.v_min
            u_max, v_max = el.u_max, el.v_max

            # Kanten als Linien plotten
            edges = [
                [(u_min, v_min), (u_max, v_min)],
                [(u_max, v_min), (u_max, v_max)],
                [(u_max, v_max), (u_min, v_max)],
                [(u_min, v_max), (u_min, v_min)]
            ]
            for edge in edges:
                x_coords = [edge[0][0], edge[1][0]]
                y_coords = [edge[0][1], edge[1][1]]
                ax.plot(x_coords, y_coords, color='black')

            # Element-ID in der Mitte des Elements anzeigen
            x_center = (u_min + u_max) / 2
            y_center = (v_min + v_max) / 2
            ax.text(x_center, y_center, str(el.id), ha='center', va='center', fontsize=12)

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
        num_colors = len(colors)
        color_index = 0
        for bspl in self.split_bsplines:
            evals = self.eval_bspl(bspl, n=1000)
            x_coords = evals[:, 0]
            y_coords = evals[:, 1]
            ax.plot(x_coords, y_coords, linestyle='-', color=colors[color_index])

            x_coords = bspl.cps[:, 0]
            y_coords = bspl.cps[:, 1]
            ax.plot(x_coords, y_coords, marker='o', linestyle='--', color=colors[color_index])

            color_index = (color_index + 1) % num_colors

        # Schnittpunkte plotten, falls vorhanden
        if self.intersection_points:
            for point in self.intersection_points:
                ax.scatter(point[0], point[1], color='red', marker='x', label='Schnittpunkte', s=100)

        plt.xlabel('X-Achse')
        plt.ylabel('Y-Achse')
        plt.title('Plot von evals')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
