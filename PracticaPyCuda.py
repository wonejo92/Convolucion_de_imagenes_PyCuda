import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np
import math
import sys
import timeit
import cv2
from PIL import Image

try:
    input_image = str(sys.argv[1])
    output_image = str(sys.argv[2])
    numSigma = str(sys.argv[3])
except IndexError:
    sys.exit("Faltan parametros!!")

try:    
    img = cv2.imread(input_image)    
    #convirtiendo en scala de gris
    imagenGris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_array = np.array(imagenGris) 
    #copiamos todas las columnas y filas de la imagen    
    oneCanal = input_array[:, :].copy()
except FileNotFoundError:
    sys.exit("No se encuentra la imagen")



#      generamos en kernel gaussiano       #
sigma = int(numSigma)  # Desviacion estandar/numero sima
kernel_width=5#Ventana de convolucion
kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
#operacion con respuesta a mediato inferior
kernel_half_width = kernel_width // 2
for i in range(-kernel_half_width, kernel_half_width + 1):
    for j in range(-kernel_half_width, kernel_half_width + 1):
        kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                / (2 * np.pi * sigma ** 2)
        )
gaussian_kernel = kernel_matrix / kernel_matrix.sum()



#calculo hilos bloques y grids segun la imagen ingresada, parametros enviados a CUDA


#conocer la dimension de la imagen
height, width = input_array.shape[:2]
dim_block = 1#numero hilos lanzados
print(height)
print(width)
#subimos al mediato superior
dim_grid_x = math.ceil(width)#numero bloques lanzados
dim_grid_y = math.ceil(height)

#ejecutar funcion en cuda
mod = compiler.SourceModule("""
__global__ void applyFilter(const unsigned char *input,
                             unsigned char *output,
                             const unsigned int width,
                             const unsigned int height,
                             const float *kernel,
                             const unsigned int kernelWidth) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < height && col < width) {
        const int half = kernelWidth / 2;
        float blur = 0.0;
        for(int i = -half; i <= half; i++) {
            for(int j = -half; j <= half; j++) {

                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));

                const float w = kernel[(j + half) + (i + half) * kernelWidth];
                blur += w * input[x + y * width];
            }
        }
        output[col + row * width] = static_cast<unsigned char>(blur);
    }
}
""")
apply_filter = mod.get_function('applyFilter')
#tiempos de ejecución
time_started = timeit.default_timer()
apply_filter(
        drv.In(oneCanal),
        drv.Out(oneCanal),
        np.uint32(width),
        np.uint32(height),
        drv.In(gaussian_kernel),
        np.uint32(kernel_width),
        block=(dim_block, dim_block, 1),#UN SOLO HILO DENTRO DE UN BLOQUE 
        grid=(dim_grid_x, dim_grid_y))#N BLOQUES POR CADA DIMENSION X e Y EN ESTE CASO 1024
time_ended = timeit.default_timer()

output_array = np.empty_like(input_array)
output_array[:,:] = oneCanal
Image.fromarray(output_array).save(output_image[:-4]+numSigma+output_image[len(output_image)-4:])
print('Tiempo Ejecución: ', time_ended - time_started, 's')

print('Proceso Terminado...')
