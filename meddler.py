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
    
templates = {}
for template in tqdm(os.listdir("Templates"), desc = 'Loading Templates'):
    
    #Reading and Writing into List
    templates[template] = cv2.imread("Templates/" + template, 0)
    
preprocessed = {}
for processed in preprocessed:
    
    preprocessed[processed] = cv2.imread("Output/" + processed, 0)

#Thresholding the Image - Stars would remain to be white pixels 
def Otsu(images):
        
    blur = cv2.GaussianBlur(images,(5,5),0)
    ret1,th1 = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite("Output/CancerOtsu2.png", th1, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def EdgeRecognition(images):
      
    edges = cv2.Canny(images,100,200)

    cv2.imwrite("Output/CancerEdges2.png", edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def Templates(images):
    
    #Edge Recognition
    edges = cv2.Canny(images,50,150,apertureSize = 3)
    
    #Line Recognition
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(images,(x1,y1),(x2,y2),(0,255,0),2)
        
    Otsu(images)

    cv2.imwrite('Templates/Preprocessed/Cancer.png',images)

def Preprocess(images):
    
    Otsu(images)
    EdgeRecognition(images)