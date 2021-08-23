#   TF描述卷积层
#   tf.keras.layers.Conv2D(
#   filters=卷积核个数,
#   kernel_size=卷积核尺寸,  #   正方形写核长整数,或（核高h,核宽w）
#   strides=滑动步长,   #   横纵向相同写步长整数,或（纵向步长h,横向步长w）, 默认为1
#   padding='SAME' or 'VALID', # same：使用全零填充   valid：不使用全零填充
#   activation = 'relu' or 'sigmoid' or 'tanh' or 'softmax' 等,  #   （如果有BN层此次不写）
#   input_shape=(高,宽,通道数)   #   输入特征图维度,可省略
#   )

import tensorflow as tf 
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense

model = tf.keras.models.Sequential([
    Conv2D(6, 5, padding='valid', activation='sigmoid'),
    MaxPool2D(2, 2),
    Conv2D(6, (5, 5), padding='valid', activation='sigmoid'),
    MaxPool2D(2, 2),
    Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='sigmoid')
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(10, activation='softmax')])
    
])
