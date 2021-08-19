#   数据增强（增大数据量）
#   image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#   rescale = 所有数据将乘以该数值
#   rotation_range = 随机旋转角度数范围
#   width_shift_range = 随机宽度偏移量
#   height_shift_range = 随机高度偏移量
#   horizontal_flip = 是否随机水平翻转
#   zoom_range = 随机缩放的范围[1-n, 1+n]
#   )
#   image_gen_train.fit(x_train)

#   example：
#       image_gen_train = ImageDataGenerator(
#       rescale = 1. / 1.,   #  如为图像,分母为255时,可归至0-1
#       rotation_range = 45  #  随机45度旋转
#       width_shift_range = 0.15 #  宽度偏移
#       height_shift_range = 0.15 # 高度偏移
#       horizontal_flip = Flase #   水平翻转
#       zoom_range = 0.5 #  将图像随机缩放阈量50%
#       )
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据集增加一个维度，是数据和网络结构匹配。

image_gen_train = ImageDataGenerator(
    rescale=1. /1.,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5
)

image_gen_train.fit(x_train)

model = tf.keras.Sequential(
    [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=['sparse_categorical_accuracy']
)

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test), validation_freq=2)
model.summary()

