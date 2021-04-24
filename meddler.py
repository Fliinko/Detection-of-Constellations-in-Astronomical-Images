import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm.notebook import tqdm

#Class made to reduce the quality of given images, to create more images for a dataset

#Storing the Images in the Subfolder as a List
images = {}
for image in tqdm(os.listdir("Images"), desc = 'Loading Data'):
    
    images[image] = cv2.imread("Images/" + image, 0)    
        

#Creating a variable which stores an image
img = images["0.png"].copy()

#Applying Otsu and Edge Detection
Preprocess(img)

#Thresholding the Image - Stars would remain to be white pixels 
def Otsu(images):
           
    blur = cv2.GaussianBlur(images,(5,5),0).astype('uint8')
    ret1,th1 = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Process is Done by here
    
    #Saving File    
    cv2.imwrite("Output/Preprocessed/Otsutest.png" , th1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    #Lists to make Histogram output neater
    imgs = [blur, 0, th1]
    titles = ['Gaussian filtered Image','Histogram','Otsu Thresholding']
   
    #Histogram and Image Plot, Helps Analyse new Image      
    plt.subplot(3,3,i*3+1),plt.imshow(imgs[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(imgs[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(imgs[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
            
    plt.show()
        
def EdgeRecognition(images):
  
    #This Function does all the work
    edges = cv2.Canny(images,100,200)
    
    
    #Histogram and Image Plot, Helps Analyse new Image
    cv2.imwrite("Output/Preprocessed/Edges/testedges.png", edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
def Preprocess(images):
    
    #Calling both preprocessing functions on same image, to observe difference when implemented against eachother
    Otsu(images)
    EdgeRecognition(images)

    #Turns out not much difference between them
