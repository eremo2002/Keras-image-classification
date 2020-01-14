import numpy as np
import tensorflow as tf
import pandas as pd

import keras
from keras.layers import *
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras import regularizers
from keras.models import Model

from sgdr import *
from LRTensorBoard import LRTensorBoard


K.clear_session()
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.keras.backend.set_session(tf.Session(config=config))




# 시드 값 고정 및 하이퍼 파라미터 설정
np.random.seed(5)
epochs = 1000
train_batch_size = 32
valid_batch_size = 16


save_path = 'weight_MnasNet'

# pandas의 read_csv를 이용하여 train.csv, val.csv를 불러옵니다.
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("val.csv")

train_df['label'] = train_df['label'].astype(str)
valid_df['label'] = valid_df['label'].astype(str)



# 케라스의 IamageDataGenerator를 이용하여 training data에 대해서 아래와 같은 augmentation을 적용합니다.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True, 
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    vertical_flip = False,
    zoom_range=0.2)


# validation set의 경우 픽셀 값의 범위만 스케일링하며 그 외 다른 augmentation은 적용하지 않습니다.
test_datagen = ImageDataGenerator(
    rescale=1./255)


# generator를 사용하기 위해, dataframe이 있는 경로에서 데이터를 불러올 때 아래와 같은 포맷으로 불러옵니다.
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/faces_images',
        x_col="filename",
        y_col="label",
        color_mode='rgb',        
        target_size=(128, 128),
        batch_size=train_batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/faces_images',
        x_col="filename",
        y_col="label",
        color_mode='rgb',
        target_size=(128, 128),
        batch_size=valid_batch_size,
        class_mode='categorical')




'''
다음은 모델 구성을 위한 block 설계 부분입니다.

백본 네트워크로 MnasNet A1을 사용하였습니다.
MnasNet은 크게 MBConv_SE, MBConv, SepConv 3가지 block으로 구성되어 있으며
이러한 block 구조를 여러번 반복하여 네트워크를 구축하였습니다.

저는 3가지 block을 아래와 같은 SepConv(), MBConv_SE(), MBConv() 함수로 구현한 뒤 functional API 형식으로 모델을 구현하였습니다.

'''
 
def SepConv(inputs, in_channels, out_channels):
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=out_channels, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    return x


def MBConv_SE(inputs, expansion, in_channels, out_channels, kernel_size, strides):
    x = Conv2D(in_channels*expansion, 
                kernel_size=(1, 1), 
                padding='same', 
                use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = DepthwiseConv2D(kernel_size=kernel_size, 
                        strides=strides, 
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, in_channels*expansion))(se)
    se = Dense((in_channels*expansion)//4, activation='relu')(se)
    se = Dense((in_channels*expansion), activation='sigmoid')(se)
    se = Multiply()([x, se])

    x = Conv2D(out_channels, kernel_size=(1, 1), padding='same')(se)
    x = BatchNormalization()(x)

    if in_channels == out_channels:
        x = Add()([x, inputs])
    
    return x


def MBConv(inputs, expansion, in_channels, out_channels, kernel_size, strides):
    x = Conv2D(in_channels*expansion, 
                kernel_size=(1, 1), 
                padding='same', 
                use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    
    x = Conv2D(out_channels, 
                kernel_size=(1, 1), 
                padding='same')(x)
    x = BatchNormalization()(x)


    if in_channels == out_channels:
        x = Add()([x, inputs])

    return x





'''
loss 함수로 일반적으로 사용되는 cross_entropy 함수 대신 focal_loss 함수를 사용하였습니다.
focal_loss는 데이터셋이 불균형성(클래스 마다 샘플 수가 차이가 심한 경우)을 띄고 있는 경우 적용해볼만한 loss 함수 입니다.

예를 들어, A라는 클래스의 샘플의 수가 지나치게 많고 B라는 클래스의 샘플이 현저히 적은 경우 모델이 A클래스는 잘 예측할 수 있지만
B 클래스에 대한 정보가 부족하므로 B클래스에 대해선 예측 정확도가 떨어질 수 있습니다.
따라서 맞추기 쉬운 샘플에 대해선 전체 loss에 영향을 적게 주도록 조정하고, 맞추기 어려운 문제에 대해선 loss에 더 큰 영향을 끼칠 수 있게 합니다.

focal_loss() 함수는 아래 링크의 코드를 참조하였습니다.
https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
'''
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





# 위에서 정의한 SepConv, MBConv_SE, MBConv block 구조를 사용하여 모델을 구축합니다.

input_tensor = Input(shape=(128, 128, 3), dtype='float32', name='input')
x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)

# SepConv x1
x = SepConv(inputs=x, in_channels=32, out_channels=16)

# MBConv6 (k3x3) x2
x = MBConv(inputs=x, expansion=6, in_channels=16, out_channels=24, kernel_size=(3, 3), strides=(2, 2))
x = MBConv(inputs=x, expansion=6, in_channels=16, out_channels=24, kernel_size=(3, 3), strides=(1, 1))

# MBConv3_SE (k5x5) x3
x = MBConv_SE(inputs=x, expansion=3, in_channels=24, out_channels=40, kernel_size=(5, 5), strides=(2, 2))
x = MBConv_SE(inputs=x, expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=3, in_channels=40, out_channels=40, kernel_size=(5, 5), strides=(1, 1))

# MBConv6 (k3x3) x4
# x = MBConv(inputs=x, expansion=6, in_channels=40, out_channels=80, kernel_size=(3, 3), strides=(2, 2))
x = MBConv(inputs=x, expansion=6, in_channels=40, out_channels=80, kernel_size=(3, 3), strides=(1, 1))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))
x = MBConv(inputs=x, expansion=6, in_channels=80, out_channels=80, kernel_size=(3, 3), strides=(1, 1))

# MBConv6_SE (k3x3) x2
x = MBConv_SE(inputs=x, expansion=6, in_channels=80, out_channels=112, kernel_size=(3, 3), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=6, in_channels=112, out_channels=112, kernel_size=(3, 3), strides=(1, 1))

# MBConv6_SE (k5x5) x3
x = MBConv_SE(inputs=x, expansion=6, in_channels=112, out_channels=160, kernel_size=(5, 5), strides=(2, 2))
x = MBConv_SE(inputs=x, expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1))
x = MBConv_SE(inputs=x, expansion=6, in_channels=160, out_channels=160, kernel_size=(5, 5), strides=(1, 1))

# MBConv6 (k3x3) x1
x = MBConv(inputs=x, expansion=6, in_channels=160, out_channels=320, kernel_size=(3, 3), strides=(1, 1))

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(6, activation='softmax')(x)



model = Model(input_tensor, output_tensor)
model.summary()







# 모델 저장하기
from keras.models import load_model
model.save(save_path+'/model_MnasNet.h5')


model.compile(loss=focal_loss(alpha=1), optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])


checkpoint = ModelCheckpoint(filepath=save_path+'/epoch-{epoch:04d}_acc-{acc:.4f}_valloss-{val_loss:.4f}_valacc-{val_acc:.4f}.h5',
                            monitor='val_loss', 
                            verbose=1, 
                            save_weights_only=True)



history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=int(train_generator.n / train_generator.batch_size),
                                epochs=epochs, 
                                validation_data=validation_generator,
                                validation_steps=int(validation_generator.n / validation_generator.batch_size),
                                verbose=1, shuffle=True, callbacks=[checkpoint])

