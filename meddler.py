#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


#Class made to reduce the quality of given images, to create more images for a dataset

#%%
def Rotator(image):
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated 

#%%
def Blur(image, filter):
    blur = cv2.blur(image, (filter, filter))
    return blur

#%%
def Noise(noise_typ,image):

    if noise_typ == "0": #GAUSSIAN

        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss

        return noisy

    elif noise_typ == "1": #SALT AND PEPPER

        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0

        return out

    elif noise_typ == "2": #POISSON

        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)

        return noisy

    elif noise_typ == "3": #SPECKLE

        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss

        return noisy

#%%
def Preprocess(image, counter):

    filter = random.randint(0, 5)
    noisefactor = random.randint(0,3)

    for i in range(10):
            temp1 = Rotator(image)
            temp1 = Blur(image, filter)
            temp1 = Noise(noisefactor, image)

            print("Finished Image", counter)

            cv2.imwrite("Meddled/dataset" + str(i) + str(counter) + ".png", temp1, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return temp1

#%%
image = cv2.imread("Images/0.png", cv2.COLOR_RGB2GRAY)
Preprocess(image, 1)

print("Finished...")

