import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm.notebook import tqdm

#Class made to reduce the quality of given images, to create more images for a dataset

#Storing the Images in the Subfolder as a List
images = {}
for image in tqdm(os.listdir("Images"), desc = 'Loading and Preprocessing Raw Data'):
    
    #Reading and Writing into List
    images[image] = cv2.imread("Images/" + image, 0)  
    
    #Applying Preprocessing Algorithms
    Preprocess(images[image])
    
templates = {}
for template in tqdm(os.listdir("Templates"), desc = 'Loading and Preprocessing Templates'):
    
    #Reading and Writing into List
    templates[template] = cv2.imread("Templates/Raw/", 0)
    
    #Applying Preprocessing Algorithms
    Templates(templates[template])


#Thresholding the Image - Stars would remain to be white pixels 
def Otsu(images):
    
    counter = 0
    
    for image in images:
        
        blur = cv2.GaussianBlur(images[image],(5,5),0)
        ret1,th1 = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        counter += 1

    cv2.imwrite("Output/Preprocessed/" + str(counter) + ".png" , th1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
 
        
def EdgeRecognition(images):
  
    counter = 0
    
    for image in images:
        edges = cv2.Canny(images[image],100,200)
        counter += 1

    cv2.imwrite("Output/Preprocessed/Edges/" + str(counter) + ".png", edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
def CornerRecognition(images):
    
    corners = cv2.goodFeaturesToTrack(images,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(images,(x,y),3,255,-1)

    plt.imshow(images),plt.show()
    cv2.imwrite("Output/Preprocessed/Edges/testcorners.png", images, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def Templates(images):
    
    #Thresholding the Image
    Otsu(images)
    
    #Edge Recognition
    edges = cv2.Canny(images,50,150,apertureSize = 3)
    
    #Line Recognition
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(images,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('Templates/Preprocessed/Cancer.png',images)
    
def Preprocess(images):
    
    #Calling both preprocessing functions on same image, to observe difference when implemented against eachother
    Otsu(images)
    EdgeRecognition(images)
    CornerRecognition(images)
    
    #Turns out Corner Recognition is not very useful, but tested none the less
    #Turns out not much difference between them
