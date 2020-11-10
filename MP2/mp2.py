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
    # result = result.astype(np.uint8) # Convert to uint8 to show img        
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
    sobelGradY = sobelGradX.T # Transpose of X-direction Array

    GradX = cv2.filter2D(s, cv2.CV_64F, sobelGradX)
    GradY = cv2.filter2D(s, cv2.CV_64F, sobelGradY)

    mag = np.hypot(GradX, GradY) # Hypotenuse = sqrt(a^2+b^2)
    mag *= 255.0 / mag.max()
    # mag = mag.astype(np.uint8) # Convert to uint8 to show img

    theta = np.arctan2(GradY, GradX) # arctan2 gives theta between -pi to pi {Radian!!!}
    theta = np.rad2deg(theta) + 180 # Convert to Degrees between 0 to 360 {Degrees}
    # theta = theta.astype(np.uint8) # Convert to uint8 to show img
    
    # print(theta)
    return (mag, theta)

# Non-maxima Suppression
def NonmaximaSuppress(mag_image, theta):
    height = mag_image.shape[0]
    width = mag_image.shape[1]
    result = np.zeros(mag_image.shape)

    # Iterate through the pixels 
    # Compare neighbor between 16 positions [between each of the 8 sections]
    for x in range (1, width-1):
        for y in range (1, height-1):
            angle = theta[x, y]            
            if np.any(angle >= 0) and np.any(theta < 22.5) or np.any(angle >= 157.5) and np.any(angle < 180): # 0 - 22.5
                # Compare left-right
                a = mag_image[x, y-1]
                b = mag_image[x, y+1]
            elif np.any(angle >= 22.5) and np.any(angle < 67.5): # 22.5 - 67.5
                # Compare diagonally
                a = mag_image[x-1, y+1]
                b = mag_image[x+1, y-1]
            elif np.any(angle >= 67.5) and np.any(angle < 112.5): # 67.5 - 112.5
                # Compare top-bottom
                a = mag_image[x-1, y]
                b = mag_image[x+1, y]
            elif np.any(angle >= 112.5) and np.any(angle < 157.5): # 112.5 - 157.5
                # Compare diagonally
                a = mag_image[x+1, y+1]
                b = mag_image[x-1, y-1]

            if np.any(mag_image[x, y] >= a) and np.any(mag_image[x, y] >= b):
                result[x, y] = mag_image[x, y]
    result = result.astype(np.uint8) # Convert to uint8 to show img        
    print(result)
    return result            

if __name__ == "__main__":
    k_size = 9
    sigma = 6

    img_in = cv2.imread('lena_gray.png')
    gaussFilter = GaussianSmoothing(img_in, k_size, sigma)
    Mag, Theta = ImageGradient(gaussFilter)
    Mag = NonmaximaSuppress(Mag, Theta)

    # cv2.imshow("Altered Image Gradient [Guassian Filter]", gaussFilter)
    # cv2.imshow("Altered Image Gradient [Theta]", Theta)
    # cv2.imshow("Altered Image Gradient [Mag]", Mag)
    cv2.imshow("Altered Image NMS", Mag)
    cv2.waitKey(0)