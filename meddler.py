import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

#Class made to reduce the quality of given images, to create more images for a dataset

#Storing the Images in the Subfolder as a List
images = {}
for image in tqdm(os.listdir("Images"), desc = 'Loading Raw Data'):
    
    #Reading and Writing into List
    images[image] = cv2.imread("Images/" + image, 0)  
    
    
