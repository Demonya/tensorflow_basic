#   在神经网络训练时,将一部分神经元按照一定概率从神经网络中暂时舍弃。
#   神经网络使用时,被舍弃的神经元恢复链接。
#   TF描述池化：tf.keras.layers.Dropout(舍弃的概率)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Activation,Dropout

model = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same'),  # 卷积层
    BatchNormalization(),  # 标准化
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2)  # Dropout层
])
