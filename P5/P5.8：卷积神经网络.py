#   卷积神经网络：借助卷积核提取特征后,送人全连接网络
#   卷积神经网络的主要模块：
#   1）卷积：Convolutional
#   2）批标准化：Batch Normalization
#   3）激活：Activation
#   4）池化：Pooling
#   5）舍弃：Dropout
#   6）全连接：Full connection

#   卷积就是特征提取器,就是CBAPD

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPool2D, Dropout

model = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2)
])
