import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import math as m

#Guassian Smoothing
def GaussianSmoothing(img_in, k_size, sigma):
    guassKernel = g_kernel(k_size, sigma) # Calculate the gaussain kernel
    result = cv2.filter2D(img_in, cv2.CV_64F, guassKernel) # Uses convolution from cv2; Use CV_64F
    return result

def g_kernel(size, sigma):
    i = j = size
    gauss = np.zeros((i,j))
    i = size//2
    j = size//2

    for x in range (-i, i+1):
        for y in range (-j, j+1):
            # Equation of the 2D Gaussian
            equation = (1/(2 * np.pi * (sigma**2))) * (np.exp(-(x**2+y**2) / (2*sigma**2)))
            gauss[x+i, y+j] = equation 
    return gauss

# Using Sobel Operator
def ImageGradient(s):
    sobelGradX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8
    # sobelGradY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, 1]])/8
    sobelGradY = sobelGradX.T # Transpose of X-direction Array 1

    GradX = cv2.filter2D(s, cv2.CV_64F, sobelGradX)
    GradY = cv2.filter2D(s, cv2.CV_64F, sobelGradY)

    mag = np.hypot(GradX, GradY) # Hypotenuse = sqrt(a^2+b^2)
    mag *= 255.0 / mag.max()
    mag = mag.astype(np.uint8) # Auto convert to uint8 to show img

    theta = np.arctan2(GradY, GradX) # arctan2 gives theta between -pi to pi {Radian!!!}
    theta = np.rad2deg(theta) # Convert to Degrees between 0 to 360 {Degrees}

    # print(theta)
    return (mag, theta)

# Non-maxima Suppression
def NonmaximaSuppress(mag_image, theta):
    height = mag_image.shape[0]
    width = mag_image.shape[1]
    result = np.zeros(mag_image.shape)

    # theta = theta * (180 / np.pi) # Convert Radian to Degree
    theta[theta < 0] +=180
    # Iterate through the pixels 
    # Compare neighbor between 16 positions [between each of the 8 sections]
    for i in range (1, width-1):
        for j in range (1, height-1):
            if np.any(theta[i, j] <= 22.5) or np.any(theta[i, j] > 157.5):
                if np.any(mag_image[i, j] <= mag_image[i-1, j]) and np.any(mag_image[i, j] <= mag_image[i+1, j]):
                    result[i, j] = 0
            if np.any(theta[i, j] > 22.5) and np.any(theta[i, j] <= 67.5):
                if np.any(mag_image[i, j] <= mag_image[i-1, j-1]) and np.any(mag_image[i, j] <= mag_image[i+1, j+1]):
                    result[i, j] = 0
            if np.any(theta[i, j] > 67.5) and np.any(theta[i, j] <= 112.5):
                if np.any(mag_image[i, j] <= mag_image[i+1, j+1]) and np.any(mag_image[i, j] <= mag_image[i-1, j-1]):
                    result[i, j] = 0
            if np.any(theta[i, j] > 112.5) and np.any(theta[i, j] <= 157.5):
                if np.any(mag_image[i, j] <= mag_image[i+1, j-1]) and np.any(mag_image[i, j] <= mag_image[i-1, j+1]):
                    result[i, j] = 0
    print(result)
    return result
            

if __name__ == "__main__":
    k_size = 9
    sigma = 2

    img_in = cv2.imread('lena_gray.png')
    gaussFilter = GaussianSmoothing(img_in, k_size, sigma)
    Mag, Theta = ImageGradient(gaussFilter)
    Mag = NonmaximaSuppress(Mag, Theta)

    cv2.imshow("Altered Image NSM", Mag)
    # cv2.imshow("Altered Image", Theta)
    cv2.waitKey(0)