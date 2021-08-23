#   池化：用户减少特征数据量
#   最大值池化：可提取图片纹理,   均值池化：可保留背景特征
#   输入特征：           2*2,stride=2 最大池化：              2*2,stride均值池化：
#   [[1 1 2 4]              [[6 8]                          [[3.25 5.25]
#    [5 6 7 8]               [3 4]]                          [2       2]]
#    [3 2 1 0]
#    [1 2 3 4]]

#   TF描述池化：
#   最大池化:
#   tf.keras.layers.MaxPool2D(
#   pool_size=池化核尺寸,  # 正方形写核长整数,或 （核高h,核宽w）
#   stride=池化步长, # 步长整数,或（纵向步长h,横向步长w）,默认pool_size
#   padding='valid' or 'same' # same:使用全零填充   valid（默认）：不使用全零填充
#   )

#   均值池化:
#   tf.keras.layers.AveragePool2D(
#   pool_size=池化核尺寸,  # 正方形写核长整数,或 （核高h,核宽w）
#   stride=池化步长, # 步长整数,或（纵向步长h,横向步长w）,默认pool_size
#   padding='valid' or 'same' # same:使用全零填充   valid（默认）：不使用全零填充
#   )

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout

model = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5),padding='same'),  # 卷积层
    BatchNormalization(),  # 标准化
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),  # 池化层
    Dropout(0.2)  # Dropout层
])
