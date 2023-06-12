import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

VIZINHANCA = 5

img = cv2.imread('lena.png') #abrir a imagem
#print(img) #Mostra os pixels

img_pb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img.shape[0] #altura
width = img.shape[1] #largura 

mat = np.zeros(img.shape).astype(np.uint8)

mat_med = mat_knn = mat_mediana = mat

tam_viz = int((VIZINHANCA - 1)/ 2)


def media():
    soma = 0
    for i in range(y - tam_viz, y + tam_viz):
        for j in range(x - tam_viz, x + tam_viz):
            soma += img_pb[i, j]
        
    return (soma / (VIZINHANCA * VIZINHANCA))


def knn():
    soma = 0
    k = 9
    vet_viz = np.zeros(VIZINHANCA * VIZINHANCA)

    aux = 0

    meio = 0

    for i in range(y - tam_viz, y + tam_viz):
        for j in range(x - tam_viz, x + tam_viz):
            if(VIZINHANCA % 2 == 0):
                if(i == (VIZINHANCA / 2) and i == j):
                    meio = img_pb[i, j]
            else:
                if(i == ((VIZINHANCA / 2) + 1) and i == j):
                    meio = img_pb[i, j]

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
    soma = 0
    for i in range(y - tam_viz, y + tam_viz):
        for j in range(x - tam_viz, x + tam_viz):
            soma += img_pb[i, j]
        
    return (soma / (VIZINHANCA * VIZINHANCA))

for y in range(tam_viz, mat.shape[0] - tam_viz):
    for x in range(tam_viz, mat.shape[1] - tam_viz):
        #mat_med[y, x] = media()
        mat_knn[y, x] = knn()
        #mat_mediana = mediana()
        
print(mat_med)

print(mat_knn)

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(mat_knn, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()