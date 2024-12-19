import numpy as np
import splinepy
from iga_framework.Brep import Brep

def circle(T, center_x, center_y, radius, Brep_id: int, outerloop: bool, num_points = 50):

    if outerloop:
        # Winkelwerte von 0 bis 2π
        angles = np.linspace(0.1, 2 * np.pi + 0.1, num_points)
    else:
        angles = np.flip(np.linspace(0.1, 2 * np.pi + 0.1, num_points))

    # Polarkoordinaten in kartesische Koordinaten umwandeln
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)

    # Liste der x,y-Paare erstellen
    circle_points = np.column_stack((x_coords, y_coords))
    Brep0_list = [circle_points]

    curves = []
    for data_i in Brep0_list:
        # fit curve by interpolating data sample points
        curve, _ = splinepy.helpme.fit.curve(data_i, degree=2)
        curves.append(curve)
    Brep0 = Brep(curves, Brep_id, T.elements, outerloop)

    return Brep0

