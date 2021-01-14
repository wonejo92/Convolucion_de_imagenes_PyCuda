# secuencial

import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import time

inicio = time.time()

img = skimage.io.imread('1024.jpg')
#img = skimage.io.imread('face.jpg')
img = skimage.color.rgb2gray(img)



def correlacion(img, mascara):
    filas, columnas = img.shape
    m, n = mascara.shape
    Nueva = np.zeros((filas + m - 1, columnas + n - 1))
    n = n // 2
    m = m // 2
    imagenFiltrada = np.zeros(img.shape)
    Nueva[m:Nueva.shape[0] - m, n:Nueva.shape[1] - n] = img
    for i in range(m, Nueva.shape[0] - m):
        for j in range(n, Nueva.shape[1] - n):
            matriz_vecinos = Nueva[i - m:i + m + 1, j - m:j + m + 1]
            result = matriz_vecinos * mascara
            imagenFiltrada[i - m, j - n] = result.sum()
    return imagenFiltrada


def gaussianBLUR(m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m // 2
    n = n // 2
    for x in range(-m, m + 1):
        # print(x)
        for y in range(-n, n + 1):
            # print(y)
            x1 = sigma * (2 * np.pi) ** 2
            x2 = np.exp(-(x * 2 + y) / (2 * sigma * 2))
            gaussian[x + m, y + n] = (1 / x1) * x2
    return gaussian


g = gaussianBLUR(5, 5, 1)
n = correlacion(img, g)
#Impresion dle tiempo de ejecucion
final = time.time()
total = final - inicio
print(" El tiempo de ejecucion es de: ",total," Segundos. ")
plt.imshow(img, cmap='gray')
plt.figure()
plt.imshow(n, cmap='gray')
plt.show