#   AlexNet:Hiton的代表作之一,2012年  ImageNet竞赛冠军
#   使用relu激活函数提升了训练速度,使用Dropout缓解了过拟合
#   AlexNet共八层网络：
#   第一层
#   C(核：96*3*3, 步长：1, 全零填充：valid)
#   B(Yes,LRN)  #  原论文中使用局部响应标准化LRN,由于LRN操作近些年用的较少,且功能与BN操作相似,选择当前主流的BN操作实现特征标准化
#   A(relu)
#   P(max, 核：3*3, 步长：2)
#   D(None)

#   第二层
#   C(核：256*3*3, 步长：1, 全零填充：valid)
#   B(Yes,LRN)  #  原论文中使用局部响应标准化LRN,由于LRN操作近些年用的较少,且功能与BN操作相似,选择当前主流的BN操作实现特征标准化
#   A(relu)
#   P(max, 核：3*3, 步长：2)
#   D(None)

#   第三层
#   C(核：384*3*3, 步长：1, 全零填充：same)
#   B(None)
#   A(relu)
#   P(None)
#   D(None)

#   第四层
#   C(核：384*3*3, 步长：1, 全零填充：same)
#   B(None)
#   A(relu)
#   P(None)
#   D(None)

#   第五层
#   C(核：256*3*3, 步长：1, 全零填充：same)
#   B(None)  #  原论文中使用局部响应标准化LRN,由于LRN操作近些年用的较少,且功能与BN操作相似,选择当前主流的BN操作实现特征标准化
#   A(relu)
#   P(max, 核：3*3, 步长：2)
#   D(None)

#   第6层
#   Dense(神经元：2048, activation=relu)
#   Dropout(0.5)

#   第7层
#   Dense(神经元：2048, activation=relu)
#   Dropout(0.5)

#   第8层
#   Dense(神经元：10, activation=softmax)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.l1_c = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')
        self.l1_bn = BatchNormalization()
        self.l1_a = Activation('relu')
        self.l1_p = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.l2_c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid')
        self.l2_bn = BatchNormalization()
        self.l2_a = Activation('relu')
        self.l2_p = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.l3_c = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')

        self.l4_c = Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')

        self.l5_c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.l5_p = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.flatten = Flatten()

        self.l6_fc = Dense(2048, activation='relu')
        self.l6_drop = Dropout(0.5)

        self.l7_fc = Dense(2048, activation='relu')
        self.l7_drop = Dropout(0.5)

        self.l8_fc = Dense(10, activation='softmax')

    def call(self, x):
        x = self.l1_c(x)
        x = self.l1_bn(x)
        x = self.l1_a(x)
        x = self.l1_p(x)

        x = self.l2_c(x)
        x = self.l2_bn(x)
        x = self.l2_a(x)
        x = self.l2_p(x)

        x = self.l3_c(x)

        x = self.l4_c(x)

        x = self.l5_c(x)
        x = self.l5_p(x)

        x = self.flatten(x)

        x = self.l6_fc(x)
        x = self.l6_drop(x)

        x = self.l7_fc(x)
        x = self.l7_drop(x)

        y = self.l8_fc(x)
        return y

model = AlexNet()

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
