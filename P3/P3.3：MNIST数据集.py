#   MNIST数据集：
#       提供6W张28*28像素点的0-9手写数字图片和标签，用于训练。
#       提供1W张28*28像素点的0-9手写数字图片和标签，用于测试。
#   导入MNIST数据集：mnist = tf.keras.datasets.mnist
#   （x_train,y_train）,(x_test,y_test) = mnist.load_data()
#   作为输入特征,输入神经网络时，将数据拉伸为一维数组：
#   tf.keras.layers.Flatten() 

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train),  (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0])
