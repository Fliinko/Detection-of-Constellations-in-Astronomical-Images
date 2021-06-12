#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10, floor
#%%

for x in os.listdir("Templates"):
    temp1 = cv2.imread("Templates/" + x)
    _, thresh = cv2.threshold(temp1, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("Templates/threshed" + x, thresh, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#%%
def CompressTemplate(template):

    temp1 = template
    try:
        final = cv2.resize(temp1, (800,800), interpolation = cv2.INTER_AREA)
        cv2.imwrite("PreprocessedTemplates/" + template+ ".png", final, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    except Exception as e:
        print(str(e))
        

for x in os.listdir("Templates"):
    print(x)
    image = cv2.imread(x, cv2.COLOR_BGR2GRAY)
    CompressTemplate("Templates/" + x)

#%%
def Rounder(x):
    return round(x, 15-int(floor(log10(abs(x))))-1)
#%%
#def TemplateMatching(images, templates):
for x in os.listdir("Matchable"):
    for y in os.listdir("Templates"):
        img = cv2.imread("Matchable/" + x)
        img1 = img
        #print(img1, img)

        template = cv2.imread("Templates/" + y)
        #print(template)
        w, h, _ = template.shape[::-1]

        method = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED']

        for i in method:
            img = img1
            method = eval(i)

            try:
                res = cv2.matchTemplate(img1, template, method)
        
                _minVal, _maxVal, minLoc, maxLoc  = cv2.minMaxLoc(res)
                Rounder(_maxVal)
                print("Matching", x, "to", y, "with: ", method,":\nAccuracy: ", _maxVal)

                loc = np.where(res >= 0.8)

                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
            
            except Exception as e:
                print(str(e))
                #cv2.imshow('Detected', img)


#FOR CNN 
#%%
from keras.models import Sequential
import tensorflow as tf
import tensorflow_datasets as tfds


from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import os
import random
from numpy import *
from PIL import Image

#Preparing Dataset For Training
#%%
path = "Matchable/"
CATEGORIES = [] #INSERT STAR NAMES HERE - YET TO DECIDE WHICH 
IMGSIZE = 1800

#Creating Training Dataset
#%%
train = []
def CreateTraining():
    for category in CATEGORIES:
        temp = os.path.join(path, category)
        class_number = CATEGORIES.index(category)
        
        for img in os.listdir(temp):
            img_array = cv2.imread(os.path.join(temp, img))
            train.append([img_array, class_number])

CreateTraining()

#%%
#Shuffling the Dataset
random.shuffle(train)

#%%
#Assign Labels and Features
x = []
y = []
for i in os.listdir("Templates"):
    for j in os.listdir("Matchable"):
        x.append("Templates/" + i)
        y.append("Matchable/" + j)

for features, label in train:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMGSIZE, IMGSIZE, 3)

#%%
#Normalizing X and Converting Labels 
x = x.astype('float32')
x /= 255

y = np_utils.to_categorical(y, 4)

#%%
#Splitting X and Y for CNN 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# %%
#Define, Compute and Train CNN 
batch_size = 16
nb_classes =4
nb_epochs = 5
img_rows, img_columns = 200, 200
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

#%%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(4,  activation=tf.nn.softmax)
])

#%%
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (x_test, y_test))

#%%
#Accuracy and Score of the Model 
score = model.evaluate(x_test, y_test, verbose = 0)

print("Test Score: ", score[0])
print("Test Accuracy: ", score[1])