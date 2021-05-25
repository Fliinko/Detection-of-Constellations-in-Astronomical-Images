#%%
import cv2
import numpy as np

#%%
def TemplateMatching(images, templates):
    
    w, h = templates.shape[::-1]

    res = cv2.matchTemplate(images, templates, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(images, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imwrite('Templates/CancerMatch.png',images)


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
import theano

#Preparing Dataset For Training
#%%
path = "/Images/"
CATEGORIES = [" "] #INSERT STAR NAMES HERE - YET TO DECIDE WHICH 
IMGSIZE = 200

#Creating Training Dataset
#%%
train = []
def CreateTraining():
    for category in CATEGORIES:
        temp = os.path.join(path, category)
        class_number = CATEGORIES.index(category)
        
        for img in os.listdir(temp):
            img_array = cv2.imread(os.path.join(temp, img))
            new_array = cv2.resizes(img_array, (IMGSIZE, IMGSIZE))

            train.append([new_array, class_number])

CreateTraining()

#%%
#Shuffling the Dataset
random.shuffle(train)

#%%
#Assign Labels and Features
x = []
y = []
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