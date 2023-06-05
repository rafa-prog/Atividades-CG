import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

VIZINHANCA = 15

img = cv2.imread('lena.png') #abrir a imagem
#print(img) #Mostra os pixels

img_pb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height = img.shape[0] #altura
width = img.shape[1] #largura 

mat = np.zeros(img.shape).astype(np.uint8)

tam_viz = int(VIZINHANCA / 2)

def arranjo_med():
    for y in range(tam_viz, mat.shape[0] - tam_viz):
        for x in range(tam_viz, mat.shape[1] - tam_viz):
            soma = 0

            for i in range(y - tam_viz, y + tam_viz):
                for j in range(x - tam_viz, x + tam_viz):
                    soma += img_pb[i, j]
        
            mat[y, x] = soma / (VIZINHANCA * VIZINHANCA)

return mat 

def arranjo_kn():
    for y in range(tam_viz, mat.shape[0] - tam_viz):
        for x in range(tam_viz, mat.shape[1] - tam_viz):
            soma = 0

            for i in range(y - tam_viz, y + tam_viz):
                for j in range(x - tam_viz, x + tam_viz):
                    soma += img_pb[i, j]
        
            mat[y, x] = soma / (VIZINHANCA * VIZINHANCA)

return mat 



#for y in range(height):
 #   for x in range(width):

print(mat)

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()