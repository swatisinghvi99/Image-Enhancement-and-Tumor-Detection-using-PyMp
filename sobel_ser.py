import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from time import perf_counter

def binarize_array(numpy_array, threshold=200):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 0
            else:
                numpy_array[i][j] = 255
    return numpy_array


def sobel_util_horizontal(img, x, y):
    pixelval = (img[x - 1][y - 1] - 1) + (img[x - 1][y] - 2) + (img[x - 1][y + 1] * -1)
    pixelval += (img[x + 1][y - 1] * 1) + (img[x + 1][y] * 1) + (img[x + 1][y + 1] * 1)
    return pixelval


def sobel_util_vertical(img, x, y):
    pixelval = (img[x - 1][y - 1] - 1) + (img[x][y - 1] - 2) + (img[x + 1][y - 1] * -1)
    pixelval += (img[x - 1][y + 1] * 1) + (img[x][y + 1] * 2) + (img[x + 1][y + 1] * 1)
    return pixelval

def power_law(img,gamma):
    x = img.shape[0]
    y = img.shape[1]

    for i in range(0,x):
        for j in range(0,y):
            img[i][j] = 255*(img[i][j]/255)**gamma

    return img

def sobel(img):
    padimg = np.pad(img, 1, mode='constant')

    x = padimg.shape[0]
    y = padimg.shape[1]

    for i in range(1, x - 2):
        for j in range(1, y - 2):
            img[i - 1][j - 1] = math.sqrt(
                sobel_util_horizontal(padimg, i, j) ** 2 + sobel_util_vertical(padimg, i, j) ** 2)

    return img


if __name__ == '__main__':
    a = perf_counter()
    gray = cv2.imread('brain.jpg', 0)
    npimg = np.copy(gray)
    x_size1 = npimg.shape[0]
    y_size1 = npimg.shape[1]
    ew= (x_size1 +2,y_size1 +2)
    simg = np.zeros(ew,np.uint8)
    for i in range(0, npimg.shape[0]):
        for j in range(0, npimg.shape[1]):
            simg[i+1][j+1] = npimg[i][j]
    plt.subplot(1, 2, 1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.title('(a) Original Grey Scale Image')
    plt.xticks([])
    plt.yticks([])

    i = None
    j = None

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    laplaceimg = np.copy(gray)
    im = cv2.filter2D(laplaceimg, -2, kernel)
    plt.subplot(1, 2, 2)
    plt.imshow(im, cmap=plt.cm.binary)
    plt.title('(b) Laplacian of (a)')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot1.png')
    plt.show()

    final = cv2.add(im, laplaceimg)
    k=np.copy(final)
    plt.subplot(1, 2, 1)
    plt.imshow(final, cmap=plt.cm.binary)
    plt.title('(c) Sharpened Image')
    plt.xticks([])
    plt.yticks([])

    sobelimg = sobel(simg)
    binaryimg = binarize_array(sobelimg)
    plt.subplot(1, 2, 2)
    plt.imshow(sobelimg, cmap=plt.cm.binary)
    plt.title('(d) Sobel of (a)')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot2.png')
    plt.show()


    newimg = cv2.blur(binaryimg, (5, 5))

    x_size2 = final.shape[0]
    y_size2 = final.shape[1]
    ee = (x_size2 + 2, y_size2 + 2)
    si = np.zeros(ee, np.uint8)
    for i in range(0, final.shape[0]):
        for j in range(0, final.shape[1]):
            si[i + 1][j + 1] = final[i][j]

    plt.subplot(1, 2, 1)
    plt.imshow(newimg, cmap=plt.cm.binary)
    plt.title('(e) Sobel smoothened image\nwith averaging filter')
    plt.xticks([])
    plt.yticks([])

    final1 = npimg * k
    plt.subplot(1, 2, 2)
    plt.imshow(final1, cmap=plt.cm.binary)
    plt.title('(f) Product of (c) and (e) ')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot3.png')
    plt.show()
    e=np.copy(final1)
    q = cv2.add(npimg, e)
    plt.subplot(1, 2, 1)
    plt.imshow(q, cmap=plt.cm.binary)
    plt.title('(g) Sharpened Image obtained \nby sum of (a) and (f)')
    plt.xticks([])
    plt.yticks([])
    qq=np.copy(q)
    finalimg = power_law(qq, 2.5)
    b = perf_counter()
    plt.subplot(1, 2, 2)
    plt.imshow(finalimg, cmap=plt.cm.binary)
    plt.title('(h) Final Image: Power-law \ntransformation of (g)')
    plt.xticks([])
    plt.yticks([])
    print("Time for serial execution (in seconds) = " + str(b - a))
    plt.savefig('plot4.png')
    plt.show()



