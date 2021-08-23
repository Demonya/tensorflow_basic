#   用卷积神经网络训练cifar10数据集：
#   搭建一个一层卷积 两层全连接的网络：
#   使用6个5*5的卷积核,过一个步长为2且大小为2*2的池化核,过128个神经元的全连接层,
#   因label是10分类,过10个神经元的全连接层。
#   1） 5*5 conv, filters=6  2）2*2 pool， strides=2  3）Dense 128   4）Dense 10

#   C:(核：6*5*5, 步长：1, 填充：same)
#   B:（Yes）
#   A:（relu)
#   P:(max, 核：2*2, 步长：2, 填充：same)
#   D:（0.2）

#   Flatten
#   Dense(神经元：128, 激活：relu, Dropout：0.2)
#   Dense(神经元：10, 激活：softmax) #使输出符合概率分布

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import os

#   导入数据集
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#   定义网络结构
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c = Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b = BatchNormalization()
        self.a = Activation('relu')
        self.p = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d = Dropout(0.2)

        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.fd = Dropout(0.2)
        self.d2 = Dense(10, activation='softmax')
    def call(self, x):
        x = self.c(x)
        x = self.b(x)
        x = self.a(x)
        x = self.p(x)
        x = self.d(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.fd(x)
        y = self.d2(x)
        return y
#   初始化网络
model = Baseline()

#   指定优化器、损失函数、衡量指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#   断点续训
checkpoint_save_path = './checkpoint/cifar.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------------Loading Model-------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
#   训练模型
history = model.fit(x_train, y_train, batch_size=32, epochs=5,
          validation_data=(x_test, y_test),
          validation_freq=1,
          callbacks=[cp_callback])

model.summary()

#   生成参数文件
with open('./weights.txt', 'w') as f:
    for weight in model.trainable_weights:
        f.write(str(weight.name) + '\n')
        f.write(str(weight.shape) + '\n')
        f.write(str(weight.numpy()) + '\n')

#   可视化
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train ACC')
plt.plot(val_acc, label='Test ACC')
plt.title('Train and Test ACC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train loss')
plt.plot(val_acc, label='Test loss')
plt.title('Train and Test loss')
plt.legend()
plt.show()
