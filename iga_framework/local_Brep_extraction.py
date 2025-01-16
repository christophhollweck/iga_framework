import splinepy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon


def get_active_elements(Breps):

    el_in_inner_loop = set()
    el_in_outer_loop = set()
    for Brep in Breps:
        if Brep.outerloop:
            el_in_outer_loop.update(Brep.cut_elements, Brep.inside_elements)

        if not Brep.outerloop:
            el_in_inner_loop.update(Brep.inside_elements)

    active_elements = el_in_outer_loop - el_in_inner_loop

    for el in active_elements:
        el.active_after_trimming = True




def plot_Breps(Breps, elements, brep_vis = 'complete'):
    # Plot erstellen
    get_active_elements(Breps)
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

        if el.active_after_trimming:
            ax.fill([u_min, u_max, u_max, u_min], [v_min, v_min, v_max, v_max], color='gray', alpha=0.5)

        # Element-ID in der Mitte des Elements anzeigen
        x_center = (u_min + u_max) / 2
        y_center = (v_min + v_max) / 2
        ax.text(x_center, y_center, str(el.id), ha='center', va='center', fontsize=12)


    if brep_vis == 'split':
        for Brep in Breps:
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
            num_colors = len(colors)
            color_index = 0
            for bspl in Brep.split_bsplines:
                evals = Brep.eval_bspl(bspl, n=1000)
                x_coords = evals[:, 0]
                y_coords = evals[:, 1]
                ax.plot(x_coords, y_coords, linestyle='-', color=colors[color_index])

                x_coords = bspl.cps[:, 0]
                y_coords = bspl.cps[:, 1]
                ax.plot(x_coords, y_coords, marker='o', linestyle='--', color=colors[color_index])

                color_index = (color_index + 1) % num_colors

            # Schnittpunkte plotten, falls vorhanden
            if Brep.intersection_points:
                for point in Brep.intersection_points:
                    ax.scatter(point[0], point[1], color='red', marker='x', label='Schnittpunkte', s=100)

    if brep_vis == 'complete':
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
        num_colors = len(colors)
        color_index = 0
        for Brep in Breps:
            for bspl in Brep.bsplines:
                evals = Brep.eval_bspl(bspl, n=1000)
                x_coords = evals[:, 0]
                y_coords = evals[:, 1]
                ax.plot(x_coords, y_coords, linestyle='-', color=colors[color_index])

                x_coords = bspl.cps[:, 0]
                y_coords = bspl.cps[:, 1]
                ax.plot(x_coords, y_coords, marker='o', linestyle='--', color=colors[color_index])

            color_index = (color_index + 1) % num_colors

    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title('Plot von evals')
    plt.axis('equal')
    plt.show()
