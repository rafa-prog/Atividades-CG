import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

A = np.random.randint(0,8,[5,5])

vet = np.zeros(6).astype(int)

for x in range(len(vet)):
    print(x)

h = [0,0,0,0,0,0,0,0]

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        h[A[i,j]] += 1

for i in range(8):
    h[i] /= 25

teste = [[1,2,3], [4,5,6], [7,8,9]]

plt.figure(figsize=(12,12))
plt.subplot(121)
bars = plt.bar(range(8), h, color="green")

for i, bar in enumerate(bars):
    if(i == 2):
        bar.set_color("red")

plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")
plt.show()