#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import misc
from ASD import *
from timeit import default_timer as timer


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "scaled":
        algorithm = "scaled"
    else:
        algorithm = "asd"

    print("algorithm:", algorithm)

    image = cv2.imread("../Data/boat.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cambiar tamaño
    # image = cv2.resize(image, (128, 128))
    m, n = image.shape

    # Aproximación de rango bajo
    rank = 50
    low_rank_image = low_rank_approximation(image, rank)

    # Máscara
    mask = np.random.choice([0, 1], (m, n), p=[0.65, 0.35])
    masked_image = mask*low_rank_image

    # Valores iniciales
    x0 = np.random.random_integers(0, 255, m*rank)
    x0 = np.reshape(x0, (m, rank))

    y0 = np.random.random_integers(0, 255, n*rank)
    y0 = np.reshape(x0, (rank, n))

    # Optimizar
    iter_max = 10000001

    if algorithm == "asd":
        minimize = alternating_steepest_descent
    elif algorithm == "scaled":
        minimize = scaled_alternating_steepest_descent

    start = timer()
    asd_image, residuals = minimize(x0, y0, masked_image,
            mask, iter_max, norm_tol=1e-3)
    end = timer()
    print("Tiempo:", end - start, "segundos.")

    # Guardar residuos en txt
    np.savetxt("../Results/Residuals.txt", residuals)

    # Gráfica de residuos
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
    ax.semilogy(residuals, linewidth=2.0, linestyle="-", marker="o")

    fig.tight_layout()
    plt.savefig("../Results/Plot.png", bbox_inches="tight", pad_inches=0)
    # plt.show()

    # # Mostrar resultados
    # plt.figure(dpi=150); plt.imshow(image, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(mask, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(low_rank_image, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(masked_image, cmap="gray")
    # plt.figure(dpi=150); plt.imshow(asd_image, cmap="gray")
    # plt.show()

    cv2.imwrite("../Results/image.png", image)
    cv2.imwrite("../Results/mask.png", 255*mask)
    cv2.imwrite("../Results/low_rank.png", low_rank_image)
    cv2.imwrite("../Results/masked.png", masked_image)
    cv2.imwrite("../Results/asd.png", asd_image)


if __name__ == "__main__":
    main()
