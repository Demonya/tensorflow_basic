#   MNIST数据集：
#       提供6W张28*28像素点的0-9手写数字图片和标签，用于训练。
#       提供1W张28*28像素点的0-9手写数字图片和标签，用于测试。
#   导入MNIST数据集：mnist = tf.keras.datasets.mnist
#   （x_train,y_train）,(x_test,y_test) = mnist.load_data()
#   作为输入特征,输入神经网络时，将数据拉伸为一维数组：
#   tf.keras.layers.Flatten()

import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
#
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),  (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[1])
# plt.show()
#
# print("x_train[0]:\n", x_train[0])
# print("y_train[0]:\n", y_train[0])
# print("x_test.shape:", x_test.shape)

# #   Sequential初始化model
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train/255, x_test/255   # 归一化
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['sparse_categorical_accuracy'])
#
# model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
# model.summary()



#   CLASS初始化model
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255   # 归一化

#   定义类
class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
model = MnistModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
model.summary()
