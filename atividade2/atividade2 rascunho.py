import cv2 #importa a biblioteca
from matplotlib import pyplot as plt
import numpy as np
import math

img1 = cv2.imread('img_aluno.png')
img2 = cv2.imread('unequalized.jpg')

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

img_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.subplot(223)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))

scale_percent = 50
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
plt.subplot(224)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

plt.show()

z = np.arange(256)
s_ident = z
s_inver = 255-z

plt.figure(figsize=(6,6))
plt.plot(z, s_inver)
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(222)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))

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
plt.show()

c_scale = 255 / (np.log2(1+255))

z_log2 = c_scale * np.log2(z+1)

img_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_log = (c_scale * np.log2(1 + img1.astype(np.int32))).astype(np.uint8)

plt.figsize=(15,15)
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_log, cv2.COLOR_BGR2RGB), cmap="gray")
plt.show()

k = 0.025
s_sigmoid = (255 / (1 + np.exp(-k * (z - 127)))).astype(np.uint8)

plt.figure(figsize=(6,6))
img1_invertlog = (255 / (1 + np.exp(-k * (img1.astype(np.int32) - 127)))).astype(np.uint8)
plt.subplot(121)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.axis("off")
plt.imshow(cv2.cvtColor(img1_invertlog, cv2.COLOR_BGR2RGB), cmap="gray")
plt.show()

T = 100
indL = np.where(img_pb > T)
img1_thresh = np.zeros(img_pb.shape)
img1_thresh[indL] = 1
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(img_pb, cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.imshow(img1_thresh, cmap="gray")
plt.axis("off")
plt.show()

# thresh = cv2.threshold(img_pb, 100, 255, cv2.THRESH_BINARY)    ou _INV / TRUNC / TOZERO / OTSU

plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(cv2.cvtColor(img_pb, cv2.COLOR_BGR2RGB))

hist_img1 = np.zeros(255 * 3).astype(int)

for y in range(255):
    for x in range(255):
        for c in range(3):
            hist_img1[img1[y,1][c] + (c * 256)] += 1

print(hist_img1)

hist_img1_pb = np.zeros(255).astype(float)

for y in range(255):
    for x in range(255):
        hist_img1_pb[img_pb[y,x]] += 1

for y in range(255):
    hist_img1_pb[y] = hist_img1_pb[y] / (255 * 255) 


print(hist_img1_pb)

plt.subplot(222)
plt.bar(range(255), hist_img1_pb)
plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")

plt.subplot(223)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(224)
bars = plt.bar(range(255 * 3), hist_img1)

for i, bar in enumerate(bars):
    if(i >= 256 and i <= (256 * 2)):
        bar.set_color("green")
    if(i >= (256 * 2) and i <= (256 * 3)):
        bar.set_color("red")

plt.xlabel("valor de intensidade")
plt.ylabel("frequencia")

plt.show()