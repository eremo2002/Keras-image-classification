'''
model evaluation & prediction
'''

import keras
import keras.backend as K
import os
import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics

import tensorflow as tf
import pandas as pd
from keras.models import Model, load_model

K.clear_session()
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


    
np.random.seed(5)

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv("train.csv")

train_df['label'] = train_df['label'].astype(str)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True, 
    vertical_flip = False,
    zoom_range=0.10)

test_datagen = ImageDataGenerator(
    rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/faces_images',
        x_col="filename",
        y_col="label",
        target_size=(128, 128),
        batch_size=10,
        color_mode='rgb',
        class_mode='categorical')


test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='data/faces_images',
        x_col='filename',             
        y_col=None,
        shuffle=False,
        target_size=(128, 128),
        batch_size=10,
        color_mode='rgb',
        class_mode=None)


model_path = 'weight_focalloss_ver3'
model = load_model(model_path+'/model_focalloss.h5')
model.load_weights(model_path+'/epoch-1412_acc-0.9998_valloss-1.7219_valacc-0.9211.h5')
model.summary()


# model evaluation for training set
# it must be Acc 100%
model.compile(loss=focal_loss(alpha=1), optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
scores = model.evaluate_generator(train_generator, steps=200)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


# prediction for test set
prediction for testset
test_generator.reset()
prediction = model.predict_generator(
    generator = test_generator,
    steps = 200,
    verbose=1
)

predicted_class_indices = np.argmax(prediction, axis=1)

# use same class information with training process
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# make result.csv
submission = pd.read_csv(os.path.join('test.csv'))
submission['prediction'] = predictions
submission = submission.drop(columns='filename')
submission.to_csv('result.csv', index=False)





