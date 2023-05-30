import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

img1 = cv2.imread("lena.png")
img2 = cv2.imread('atividade2/unequalized.jpg')

height = img1.shape[0] #altura
width = img1.shape[1] #largura 

z = np.arange(256)
s_ident = z
s_inver = 255-z

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')

img_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))
plt.axis('off')

img1_invert = 255-img1
plt.subplot(223)
plt.imshow(cv2.cvtColor(img1_invert, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis('off')

img1_pb_invert = 255-img_pb
plt.subplot(224)
plt.imshow(cv2.cvtColor(img1_pb_invert, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis('off')
plt.show()


plt.figure(figsize=(16,16))
normalizedImg = img_pb
img_norm = cv2.normalize(img1, normalizedImg, 0, 100, cv2.NORM_MINMAX)
img_norm_pb = cv2.normalize(img_pb, normalizedImg, 0, 100, cv2.NORM_MINMAX)

plt.subplot(231)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis('off')

plt.subplot(232)
plt.imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis('off')

plt.subplot(233)
plt.imshow(cv2.cvtColor(img_norm_pb, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis('off')

c_scale = 255 / (np.log2(1+255))

# z_log2 = c_scale * np.log2(z+1) formula log

img_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_log = (c_scale * np.log2(1 + img1.astype(np.int32))).astype(np.uint8)
img1_log_pb = (c_scale * np.log2(1 + img_pb.astype(np.int32))).astype(np.uint8)

plt.subplot(234)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")

plt.subplot(235)
plt.imshow(cv2.cvtColor(img1_log, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")

plt.subplot(236)
plt.imshow(cv2.cvtColor(img1_log_pb, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")

plt.show()


k = 0.025
# s_sigmoid = (255 / (1 + np.exp(-k * (z - 127)))).astype(np.uint8) formula sigmoid

plt.figure(figsize=(6,6))
img1_invertlog = (255 / (1 + np.exp(-k * (img1.astype(np.int32) - 127)))).astype(np.uint8)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")

plt.subplot(222)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_invertlog, cv2.COLOR_BGR2RGB), cmap="gray")


T = 100
indL = np.where(img_pb > T)
img1_thresh = np.zeros(img_pb.shape)
img1_thresh[indL] = 1


plt.subplot(223)
plt.imshow(img_pb, cmap="gray")
plt.axis("off")
plt.subplot(224)
plt.imshow(img1_thresh, cmap="gray")
plt.axis("off")
plt.show()

# thresh = cv2.threshold(img_pb, 100, 255, cv2.THRESH_BINARY)  ou _INV / TRUNC / TOZERO / OTSU

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))

hist_img1_pb = np.zeros(256).astype(float)

for y in range(height):
    for x in range(width):
        hist_img1_pb[img_pb[y,x]] += 1

for y in range(256):
    hist_img1_pb[y] = hist_img1_pb[y] / (255 * 255) 

plt.subplot(122)
plt.bar(range(256), hist_img1_pb, color="blue")
plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")

hist_blue_img1 = np.zeros(256).astype(float)
hist_green_img1 = np.zeros(256).astype(float)
hist_red_img1 = np.zeros(256).astype(float)

for y in range(height):
    for x in range(width):
        hist_blue_img1[img1[y,x][0]] += 1
        hist_green_img1[img1[y,x][1]] += 1
        hist_red_img1[img1[y,x][2]] += 1

for y in range(256):
    hist_blue_img1[y] = hist_blue_img1[y] / (255 * 255)
    hist_green_img1[y] = hist_green_img1[y] / (255 * 255)
    hist_red_img1[y] = hist_red_img1[y] / (255 * 255)

plt.subplot(221)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.bar(range(256), hist_blue_img1, color="blue")
plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")

plt.subplot(223)
plt.bar(range(256), hist_green_img1, color="green")
plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")

plt.subplot(224)
plt.bar(range(256), hist_red_img1, color="red")
plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")
plt.show()


img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(img2_pb)

resultado1 = np.hstack((img2_pb, equalized_image)) # colocando imagens juntas
cv2.imwrite('atividade2/resultado1.png',resultado1)


equalized_image = cv2.equalizeHist(img_pb)

resultado2 = np.hstack((img_pb, equalized_image)) # colocando imagens juntas
cv2.imwrite('atividade2/resultado2.png',resultado2)
