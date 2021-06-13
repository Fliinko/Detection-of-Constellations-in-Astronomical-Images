#%%
import cv2
import numpy as np
import os
from math import log10, floor
import sys
from datetime import datetime
#%%


#%%
for x in os.listdir("Templates"):
    temp1 = cv2.imread("Templates/" + x)
    _, thresh = cv2.threshold(temp1, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("Templates/threshed" + x, thresh, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#%%
for x in os.listdir("Templates"):
    print(x)
    image = cv2.imread("Templates/" + x, cv2.COLOR_BGR2GRAY)
    final = cv2.resize(image, (1250, 1250), interpolation = cv2.INTER_AREA)
    cv2.imwrite("PreprocessedTemplates/" + x, final, [cv2.IMWRITE_PNG_COMPRESSION, 0])

#%%
def Rounder(x):
    return round(x, 15-int(floor(log10(abs(x))))-1)
#%%
#def TemplateMatching(images, templates):
global mask 
accuracies = []

start = datetime.now()
for x in os.listdir("Matchable"):
    for y in os.listdir("PreprocessedTemplates"):
        img = cv2.imread("Matchable/" + x)
        #print(img1, img)
        mask = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
        template = cv2.imread("PreprocessedTemplates/" + y)
        #print(template)
        w, h, _ = template.shape[::-1]

        method = ['cv2.TM_CCOEFF_NORMED']

        for i in method:
            method = eval(i)
            method_accepts_mask = (cv2.TM_SQDIFF == method or method == cv2.TM_CCORR_NORMED)
            if (method_accepts_mask):
                res = cv2.matchTemplate(img, template, method, None, mask)
            else:
                res = cv2.matchTemplate(img, template, method)

            cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, 1)
        
            _minVal, _maxVal, minLoc, maxLoc  = cv2.minMaxLoc(res, None)
            rounded = Rounder(_maxVal)
            accuracies.append(1-rounded)
            print("Matching", x, "to", y, "with: ", method,":\nAccuracy: ", (1-rounded))

            loc = np.where(res >= 0.8)

end = datetime.now()
print("\nFinished Template Matching in:\t", (end-start), "\n", "Accuracy: \t", (sum(accuracies)/len(accuracies)))

            #for pt in zip(*loc[::-1]):
                #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
         
                #cv2.imshow('Detected', img)


#FOR CNN 
#%%
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import random

#Preparing Dataset For Training
#%%
path = "PreprocessedTemplates/"
CATEGORIES = ["Cancer", "Cassiopeia", "Cepheus", "Corona Borealis", "Gemini", "Hercules", "Leo", "Libra", "Lyra" ]
IMGSIZE = 1250

#Creating Training Dataset
#%%
train_dir = 'Matchable/'
validation_dir = 'Templates/'
#%%
#Shuffling the Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

traingenerator = train_datagen.flow_from_directory(train_dir, batch_size=5, class_mode='binary', target_size=(1250,1250))
validationgenerator = test_datagen.flow_from_dataframe(validation_dir, batch_size=2, class_mode='binary', target_size=(1250,1250))

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
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics = ['acc'])

#%%
history = model.fit_generator(traingenerator, validation_data=validationgenerator, steps_per_epoch = 100, epochs = 15, validation_steps = 50 ,verbose = 1)

