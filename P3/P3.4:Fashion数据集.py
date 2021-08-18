#   Fashion数据集：
#       提供6W张28*28像素点的衣裤等图片和标签,用于训练
#       提供1W张28*28像素点的衣裤等图片和标签,用于测试

#   导入Fashion的数据
import tensorflow as tf
# import matplotlib.pyplot as plt
# fashion = tf.keras.datasets.fashion_mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model


# (x_train, y_train), (x_test, y_test) = fashion.load_data()
# plt.imshow(x_train[1])
# plt.show()


#   Squential初始化model
# fashion = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_test, y_test) = fashion.load_data()
# x_train, x_test = x_train/255, x_test/255
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#           metrics=['sparse_categorical_accuracy'])
#
# model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
#
# model.summary()


#   定义CLASS函数初试model
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train/255, x_test/255

class FashionModel(Model):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=2)

model.summary()
