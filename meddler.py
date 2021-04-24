import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm.notebook import tqdm

class Meddler:
        
    #Class made to reduce the quality of given images, to create more images for a dataset

    #Storing the Images in the Subfolder as a List
    images = {}
    for image in tqdm(os.listdir("Images"), desc = 'Loading Data'):
        images[image] = cv2.imread("Images/" + x)

        #Saving as Greyscale to facilitate thresholding and star classification
        images[image] = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)
        
class Preprocessor:
        
    #Threshodling the Image - Stars would remain to be white pixels 
    def Otsu(images):
        
        counter = 0
        
        blur = cv2.GaussianBlur(images,(5,5),0)
        ret1,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        counter += 1
        
        cv2.imwrite("Images/Preprocessed/otsu" + counter + ".png", th1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        imgs = [blur, 0, th1]
        titles = ['Gaussian filtered Image','Histogram','Otsu Thresholding']
         
        for i in xrange(1):
            
            plt.subplot(3,3,i*3+1),plt.imshow(imgs[i*3],'gray')
            plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+2),plt.hist(imgs[i*3].ravel(),256)
            plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+3),plt.imshow(imgs[i*3+2],'gray')
            plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
            
        plt.show()
