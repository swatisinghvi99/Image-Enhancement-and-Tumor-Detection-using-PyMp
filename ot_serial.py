import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from time import perf_counter

MAX_IMAGESIZE = 4000
MAX_BRIGHTNESS = 255
GRAYLEVEL = 256
MAX_FILENAME = 256
MAX_BUFFERSIZE = 256
from time import perf_counter
def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 0
            else:
                numpy_array[i][j] = 255
    return numpy_array


def power_law(img, gamma):
    x = img.shape[0]
    y = img.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            img[i][j] = 255 * (img[i][j] / 255) ** gamma
    return img

if __name__ == '__main__':
    a = time.time()
    gray = cv2.imread('woman.jpg', 0)
    y = (gray.shape[0] + 2, gray.shape[1] + 2);
    npimg = np.zeros(y, np.uint8)
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            npimg[i + 1][j + 1] = gray[i][j]

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
    plt.savefig('plot9.png')
    plt.show()
    final = cv2.add(im, laplaceimg)
    plt.subplot(1, 2, 1)
    plt.imshow(final, cmap=plt.cm.binary)
    plt.title('(c) Sharpened Image')
    plt.xticks([])
    plt.yticks([])

    x = (final.shape[0] + 2, final.shape[1] + 2);
    npim = np.zeros(x, np.uint8)

    for i in range(0, final.shape[0]):
        for j in range(0, final.shape[1]):
            npim[i + 1][j + 1] = final[i][j]


    image1 = npimg
    x_size1 = npimg.shape[0]
    y_size1 = npimg.shape[1]
    w=(x_size1,y_size1)
    image2 = np.zeros(w,np.uint8)
    hist = np.zeros(256, np.int32)
    prob = [0.0] * 256
    myu = [0.0] * 256
    omega = [0.0] * 256
    sigma = [0.0] * 256
    for x in range(0, x_size1):
        for y in range(0, y_size1):
            hist[image1[x][y]] += 1
    # /* calculation of probability density */
    for i in range(0, GRAYLEVEL):
        prob[i] = float(hist[i]) / (x_size1 * y_size1)
    # /* omega & myu generation */
    omega[0] = prob[0]
    myu[0] = 0.0  # /* 0.0 times prob[0] equals zero */
    for i in range(1, GRAYLEVEL):
        omega[i] = omega[i - 1] + prob[i]
        myu[i] = myu[i - 1] + i * prob[i]
    threshold = 0
    max_sigma = 0.0
    for i in range(0, GRAYLEVEL - 1):
        if (omega[i] != 0.0 and omega[i] != 1.0):
            sigma[i] = ((myu[GRAYLEVEL - 1] * omega[i] - myu[i]) ** 2) / (omega[i] * (1.0 - omega[i]))
        else:
            sigma[i] = 0.0
        if (sigma[i] > max_sigma):
            max_sigma = sigma[i]
            threshold = i

    # /* binarization output into image2 */
    x_size2 = x_size1
    y_size2 = y_size1
    for i in range(0, x_size2):
        for j in range(0, y_size2):
            if (image1[i][j] > threshold).all():
                image2[i][j] = MAX_BRIGHTNESS
            else:
                image2[i][j] = 0
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap=plt.cm.binary)
    plt.title('(d) Otsu Binarization of (a)')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot10.png')
    plt.show()
    newimg = cv2.blur(image2, (5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(newimg, cmap=plt.cm.binary)
    plt.title('(e) Otsu Image smoothened \nwith averaging filter')
    plt.xticks([])
    plt.yticks([])
    final1 = npim * newimg
    plt.subplot(1, 2, 2)
    plt.imshow(final1, cmap=plt.cm.binary)
    plt.title('(f) Product of (c) and (e) ')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plot11.png')
    plt.show()

    q = cv2.add(npimg, final1)

    plt.subplot(1, 2, 1)
    plt.imshow(q, cmap=plt.cm.binary)
    plt.title('(g) Sharpened Image obtained \nby sum of (a) and (f)')
    plt.xticks([])
    plt.yticks([])

    s = (q.shape[0] + 2, q.shape[1] + 2);
    npi = np.zeros(s, np.uint8)

    for i in range(0, q.shape[0]):
        for j in range(0, q.shape[1]):
            npi[i + 1][j + 1] = q[i][j]



    finalimg = power_law(npi, 2.5)
    b = time.time()
    plt.subplot(1, 2, 2)
    plt.imshow(finalimg, cmap=plt.cm.binary)
    plt.title('(h) Final Image: Power-law \ntransformation of (g)')
    plt.xticks([])
    plt.yticks([])
    print("Time for serial execution (in seconds) = " + str(b - a))
    plt.savefig('plot12.png')
    plt.show()
