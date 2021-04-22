import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#Class made to reduce the quality of given images, to create more images for a dataset

#Storing the Images in the Subfolder as a List
images = {}
for x in os.listdir("Images"):
    images[x] = cv2.imread("Images/" + x)
