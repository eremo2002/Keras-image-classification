import cv2
import keras
import keras.backend as K
import os
import glob

from keras.models import Model, load_model
from keras.layers import Lambda
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import depth_to_space, space_to_depth
from group_norm import *
from group_convolution import *

K.clear_session()
        
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = os.path.join('testset dir')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(100, 100),            
        batch_size=52,
        shuffle=False,
        class_mode='categorical',
        color_mode='grayscale')

filenames = test_generator.filenames

# if use Lambda layer
# model = load_model('model file dir', custom_objects={'tf': tf})

model = load_model('model file dir')
weights = glob.glob('weight file dir/*.h5')
print('The number of weight files : ', len(weights))


acc_temp = []
acc = np.array([])
weight_name = []

cnt = 0

for i in enumerate(weights):
        file_name = os.path.basename(weights[cnt])        
        model.load_weights('weight file dir'+file_name)


        print((cnt+1), ' / ', len(weights))
        print(file_name)      

        
        scores = model.evaluate_generator(test_generator, steps=59)
        print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
        
        acc = np.append(acc, scores[1]*100)        
        weight_name.append(file_name)
        print('============================================================')
        
        
        cnt+=1       



print('======= max acc & weight file =======')
print(acc[np.argmax(acc)])
print(weight_name[np.argmax(acc)])

print(acc)

