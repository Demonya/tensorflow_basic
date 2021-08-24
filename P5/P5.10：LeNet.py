#   LeNet：1998年Lecun提出,卷积网络的开篇之作
#   通过共享卷积核减少了网络的参数
#   在统计卷积网络的层数时,一般只统计卷积计算层和全连接计算层,其余操作可以看成是卷积操作的附属
#   LeNet一共有5层网络：
#   输入：32*32*3
#   第一层
#   C(核：6*5*5, 步长：1, 全零填充：valid)
#   B(None)  # LeNet提出时还没有Batch Normalization
#   A(sigmoid)  # LeNet时代sigmoid是主流激活函数
#   P(max, 核：2*2, strides:2, 全零填充:valid)
#   D(None)  # LeNet时代还没有Dropout

#   第二层
#   C(核：16*5*5, 步长：1, 全零填充：valid)
#   B(None)  # LeNet提出时还没有Batch Normalization
#   A(sigmoid)  # LeNet时代sigmoid是主流激活函数
#   P(max, 核：2*2, strides:2, 全零填充:valid)
#   D(None)  # LeNet时代还没有Dropout

#   Flatten()
#   第三层
#   Dense(神经元：120, 激活：sigmoid)
#   第四层
#   Dense(神经元：84, 激活：sigmoid)
#   第五层
#   Dense(神经元：10, 激活：softmax)



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
# import numpy as np
import os
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.l1_c = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='relu')
        self.l1_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.l2_c = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu')
        self.l2_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.flatten = Flatten()

        self.fc1 = Dense(120, activation='relu')
        self.fc2 = Dense(84, activation='relu')
        self.fc3 = Dense(10, activation='softmax')
    def call(self, x):
        x = self.l1_c(x)
        x = self.l1_p(x)
        x = self.l2_c(x)
        x = self.l2_p(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y

model = LeNet()


model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/cifar.ckpt'
if os.path.exists(checkpoint_save_path):
    print('------------------Loading Model-------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32,
                    epochs=5,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

with open('./weights.txt', 'w') as f:
    for weight in model.trainable_weights:
        f.write(str(weight.name) + '\n')
        f.write(str(weight.shape) + '\n')
        f.write(str(weight.numpy()) + '\n')


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train ACC')
plt.plot(val_acc, label='Test ACC')
plt.title('Train AND Test ACC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train loss')
plt.plot(val_loss, label='Test loss')
plt.title('Train AND Test Loss')
plt.legend()
plt.show()
