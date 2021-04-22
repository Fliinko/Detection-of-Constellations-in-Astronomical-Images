import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.notebook import tqdm

#Class made to reduce the quality of given images, to create more images for a dataset

#Storing the Images in the Subfolder as a List
images = {}
for image in tqdm(os.listdir("Images"), desc = 'Loading Data'):
    images[image] = cv2.imread("Images/" + x)
    
    #Saving as Greyscale to facilitate thresholding and star classification
    images[image] = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)
    
#Threshodling the Image - Stars would remain to be white pixels 
def Thresh(img, alpha = 0.5):
    
    height, width = img.shape
    thresh = np.array([255*alpha])
    dark = np.count_nonzero(img < thresh)
    
    return dark/(height*width) > alpha

for location in images:
    
    images[location]['dark'] = Thresh(images[location]['img'])
