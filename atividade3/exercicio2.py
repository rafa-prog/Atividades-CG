import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

VIZINHANCA = 3

img = cv2.imread('lena.png') #abrir a imagem
#print(img) #Mostra os pixels

img_pb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img.shape[0] #altura
width = img.shape[1] #largura 

mat = np.zeros(img.shape).astype(np.uint8)

mat_med = mat_knn = mat_mediana = mat_roberts = mat_laplacian = mat_sobel = mat_prewitt= mat

tam_viz = int(VIZINHANCA/ 2)


def media():
    soma = 0
    for i in range(y - tam_viz, y + tam_viz + 1):
        for j in range(x - tam_viz, x + tam_viz + 1):
            soma += img_pb[i, j]
        
    return (soma / (VIZINHANCA * VIZINHANCA))


def knn():
    soma = 0
    k = 9
    vet_viz = np.zeros(VIZINHANCA * VIZINHANCA)

    aux = 0

    meio = int(VIZINHANCA * VIZINHANCA / 2)

    if(VIZINHANCA % 2 == 0):
        meio += 1

    for i in range(y - tam_viz, y + tam_viz + 1):
        for j in range(x - tam_viz, x + tam_viz + 1):
            vet_viz[aux] = img_pb[i, j]
            aux += 1

    
    vet_viz.sort()

    pos_meio = 0

    for i in range(len(vet_viz)):
        if(vet_viz[i] == meio):
            pos_meio = i
            break

    aux1 = pos_meio - 1
    aux2 = pos_meio + 1

    vet_soma = np.zeros(k)

    if(len(vet_viz) > k):
        for i in range(k):
            if(abs(vet_viz[pos_meio] - vet_viz[aux1]) < abs(vet_viz[pos_meio] - vet_viz[aux2])):
                vet_soma[i] = vet_viz[aux1]
                aux1 -= 1
            else:
                vet_soma[i] = vet_viz[aux2]
                aux2 += 1
    else:
        vet_soma = vet_viz

    soma = sum(vet_soma) / k

    return soma
                

def mediana():
    aux = 0
    vet_mediana = np.zeros(VIZINHANCA * VIZINHANCA)

    for i in range(y - tam_viz, y + tam_viz + 1):
        for j in range(x - tam_viz, x + tam_viz + 1):
            vet_mediana[aux] = img_pb[i, j]
            aux += 1
    
    meio = int(VIZINHANCA * VIZINHANCA / 2)

    if(VIZINHANCA % 2 == 0):
        meio += 1

    vet_mediana.sort()

    mediana = vet_mediana[meio]

    return mediana


def laplacian_operator():
    laplacian_image = np.zeros((height, width), dtype=np.uint8)
    
    laplacian_mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighbors = img_pb[i-1:i+2, j-1:j+2]
            laplacian_value = np.sum(neighbors * laplacian_mask)
            laplacian_image[i, j] = np.clip(laplacian_value, 0, 255)
    
    return laplacian_image


def roberts_edge_detector():
    roberts_edge_image = np.zeros((height, width), dtype=np.uint8)
    
    roberts_x_mask = np.array([[1, 0], [0, -1]])
    roberts_y_mask = np.array([[0, 1], [-1, 0]])
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighbors = img_pb[i-1:i+1, j-1:j+1]
            roberts_x_value = np.sum(neighbors * roberts_x_mask)
            roberts_y_value = np.sum(neighbors * roberts_y_mask)
            roberts_edge_value = np.sqrt(np.square(roberts_x_value) + np.square(roberts_y_value))
            roberts_edge_image[i, j] = np.clip(roberts_edge_value, 0, 255)
    
    return roberts_edge_image


def prewitt_edge_detector():
    prewitt_edge_image = np.zeros((height, width), dtype=np.uint8)
    
    prewitt_x_mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y_mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighbors = img_pb[i-1:i+2, j-1:j+2]
            prewitt_x_value = np.sum(neighbors * prewitt_x_mask)
            prewitt_y_value = np.sum(neighbors * prewitt_y_mask)
            prewitt_edge_value = np.sqrt(np.square(prewitt_x_value) + np.square(prewitt_y_value))
            prewitt_edge_image[i, j] = np.clip(prewitt_edge_value, 0, 255)
    
    return prewitt_edge_image


def sobel_edge_detector():
    sobel_edge_image = np.zeros((height, width), dtype=np.uint8)
    
    sobel_x_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighbors = img_pb[i-1:i+2, j-1:j+2]
            sobel_x_value = np.sum(neighbors * sobel_x_mask)
            sobel_y_value = np.sum(neighbors * sobel_y_mask)
            sobel_edge_value = np.sqrt(np.square(sobel_x_value) + np.square(sobel_y_value))
            sobel_edge_image[i, j] = np.clip(sobel_edge_value, 0, 255)
    
    return sobel_edge_image


for y in range(tam_viz, mat.shape[0] - tam_viz):
    for x in range(tam_viz, mat.shape[1] - tam_viz):
        mat_med[y, x] = media()
        mat_knn[y, x] = knn()
        mat_mediana[y, x] = mediana()
        mat_laplacian = laplacian_operator()
        mat_roberts = roberts_edge_detector()
        mat_prewitt = prewitt_edge_detector()
        mat_sobel = sobel_edge_detector()

        

plt.figure(figsize=(16,16))
plt.subplot(131)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(mat_mediana, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(mat_mediana, cv2.COLOR_BGR2RGB))
plt.axis('off')



plt.figure(figsize=(16,16))
plt.subplot(221)
plt.imshow(cv2.cvtColor(mat_laplacian, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(222)
plt.imshow(cv2.cvtColor(mat_roberts, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(223)
plt.imshow(cv2.cvtColor(mat_prewitt, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(224)
plt.imshow(cv2.cvtColor(mat_sobel, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()