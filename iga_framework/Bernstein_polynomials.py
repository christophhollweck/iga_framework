import numpy as np
import matplotlib.pyplot as plt
"""
This script evaluates the Bernsteinpolinomials between -1 and 1
"""

def eval_BS(xi, p):
    B = []
    if p == 1:
        B.append(0.5 * (1 - xi))
        B.append(0.5 * (1 + xi))

    if p == 2:
        B.append(0.25 * (1 - xi) ** 2)
        B.append(0.5 * (1 - xi ** 2))
        B.append(0.25 * (1 + xi) ** 2)

    if p == 3:
        B.append(0.125 * (1 - xi) ** 3)
        B.append(0.125 * 3 * (1 - xi) ** 2 * (1 + xi))
        B.append(0.125 * 3 * (1 - xi) * (1 + xi) ** 2)
        B.append(0.125 * (1 + xi) ** 3)

    if p == 4:
        B.append(0.0625 * (1 - xi) ** 4)
        B.append(0.0625 * ((8 - 4 * xi) * xi ** 2 - 8) * xi + 0.25)
        B.append(0.0625 * 6 * (xi ** 2 - 1) ** 2)
        B.append(0.0625 * ((-8 - 4 * xi) * xi ** 2 + 8) * xi + 0.25)
        B.append(0.0625 * (1 + xi) ** 4)

    return B

if __name__ == "__main__":


    xi_values = np.linspace(-1, 1, 1000)
    import tikzplotlib
    #plt.style.use("ggplot")
    #plt.grid(True)
    plt.xlim(-1, 1)  # Set x limits
    plt.ylim(0, 1)

    for p in range(4, 5):
        B_values = np.array([eval_BS(xi, p) for xi in xi_values]).T

        for i in range(B_values.shape[0]):
            plt.plot(xi_values, B_values[i], label=f'B_{i+1}')

        #plt.plot(xi_values, np.sum(B_values, axis=0), label='POU')



        # plt.title(f'Plot for p={p}')
        # plt.xlabel('xi')
        # plt.ylabel('B values')
        # plt.legend()
        # plt.grid(True)
        # plt.show()



    tikzplotlib.save("/media/christoph/Daten1/Python_scripts/paper_figs/bernp4.tex")
    plt.show()

