#   Cifar10数据集：
#   提供5W张32*32像素点的十分类彩色图片和标签,用于训练
#   提供1W张32*32像素点的十分类彩色图片和标签,用于测试
#   导入cifar10数据集：
#   cifar10 = tf.keras.dataset.cifar10
#   (x_train, y_train), (x_test, y_test) = cifar10.load_data()

import tensorflow as tf
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.imshow(x_train[1])  # 绘制图片
plt.show()

print("x_train[0]\n", x_train[0])
print("y_train[0]\n", y_train[0])
print("x_test.shape:", x_test.shape)
