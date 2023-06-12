import numpy as np
import cv2
import matplotlib.pyplot as plt

mat = [11,12,12,12,12,20,22,22,22,23,23,24,24,30,34,35,56,60,65,66,67,88,99,99,99]

print(mat.index(35))

pos = mat.index(35)

aux1 = pos - 1
aux2 = pos + 1

k = 9

vet = np.zeros(k).astype(int)

print(vet)

soma = 0

for i in range(k):
    if(abs(mat[pos] - mat[aux1]) < abs(mat[pos] - mat[aux2])):
        vet[i] = mat[aux1]
        aux1 -= 1
    else:
        vet[i] = mat[aux2]
        aux2 += 1
    soma += vet[i]

print(vet)

print(soma/k)

