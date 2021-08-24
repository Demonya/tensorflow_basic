#   VGGNet使用小尺寸卷积核在减少参数的同时,提高了识别准确率
#   VGGNet的网络结构规整,非常适合硬件加速
#   VGGNet 16层网络:
#   第一层：
#   C(核：64*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第二层:
#   C(核：64*3*3,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(max, 核：2*2,步长：2)
#   D(0.2)

#   第三层：
#   C(核：128*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第四层:
#   C(核：128*3*3,步长：1,,填充：same)
#   B(YES)
#   A(relu)
#   P(max, 核：2*2,步长：2)
#   D(0.2)

#   第五层：
#   C(核：256*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第六层：
#   C(核：256*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第七层:
#   C(核：256*3*3,步长：1,,填充：same)
#   B(YES)
#   A(relu)
#   P(max, 核：2*2,步长：2)
#   D(0.2)

#   第八层：
#   C(核：512*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第九层：
#   C(核：512*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第十层:
#   C(核：512*3*3,步长：1,,填充：same)
#   B(YES)
#   A(relu)
#   P(max, 核：2*2,步长：2)
#   D(0.2)

#   第十一层：
#   C(核：512*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第十二层：
#   C(核：512*3*3,步长：1,填充：same)
#   B(YES)
#   A(relu)

#   第十三层:
#   C(核：512*3*3,步长：1,,填充：same)
#   B(YES)
#   A(relu)
#   P(max, 核：2*2,步长：2)
#   D(0.2)

#   第十四层：
#   Flatten()
#   Dense(神经元:512,激活:relu,Dropout:0.2)

#   第十五层：
#   Dense(神经元:512,激活:relu,Dropout:0.2)

#   第十六层：
#   Dense(神经元:10,激活:softmax)



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

class VGGNet(Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.l1_c = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.l1_bn = BatchNormalization()
        self.l1_a = Activation('relu')

        self.l2_c = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.l2_bn = BatchNormalization()
        self.l2_a = Activation('relu')
        self.l2_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.l2_d = Dropout(0.2)

        self.l3_c = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.l3_bn = BatchNormalization()
        self.l3_a = Activation('relu')

        self.l4_c = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.l4_bn = BatchNormalization()
        self.l4_a = Activation('relu')
        self.l4_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.l4_d = Dropout(0.2)

        self.l5_c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.l5_bn = BatchNormalization()
        self.l5_a = Activation('relu')

        self.l6_c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.l6_bn = BatchNormalization()
        self.l6_a = Activation('relu')

        self.l7_c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.l7_bn = BatchNormalization()
        self.l7_a = Activation('relu')
        self.l7_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.l7_d = Dropout(0.2)

        self.l8_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l8_bn = BatchNormalization()
        self.l8_a = Activation('relu')

        self.l9_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l9_bn = BatchNormalization()
        self.l9_a = Activation('relu')

        self.l10_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l10_bn = BatchNormalization()
        self.l10_a = Activation('relu')
        self.l10_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.l10_d = Dropout(0.2)

        self.l11_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l11_bn = BatchNormalization()
        self.l11_a = Activation('relu')

        self.l12_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l12_bn = BatchNormalization()
        self.l12_a = Activation('relu')

        self.l13_c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.l13_bn = BatchNormalization()
        self.l13_a = Activation('relu')
        self.l13_p = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.l13_d = Dropout(0.2)

        self.flatten = Flatten()

        self.l14_fc = Dense(512, activation='relu')
        self.l14_drop = Dropout(0.2)

        self.l15_fc = Dense(512, activation='relu')
        self.l15_drop = Dropout(0.2)

        self.l16_fc = Dense(10, activation='softmax')

    def call(self, x):
        x = self.l1_c(x)
        x = self.l1_bn(x)
        x = self.l1_a(x)

        x = self.l2_c(x)
        x = self.l2_bn(x)
        x = self.l2_a(x)
        x = self.l2_p(x)
        x = self.l2_d(x)

        x = self.l3_c(x)
        x = self.l3_bn(x)
        x = self.l3_a(x)

        x = self.l4_c(x)
        x = self.l4_bn(x)
        x = self.l4_a(x)
        x = self.l4_p(x)
        x = self.l4_d(x)

        x = self.l5_c(x)
        x = self.l5_bn(x)
        x = self.l5_a(x)
        
        x = self.l6_c(x)
        x = self.l6_bn(x)
        x = self.l6_a(x)
        
        x = self.l7_c(x)
        x = self.l7_bn(x)
        x = self.l7_a(x)
        x = self.l7_p(x)
        x = self.l7_d(x)

        x = self.l8_c(x)
        x = self.l8_bn(x)
        x = self.l8_a(x)

        x = self.l9_c(x)
        x = self.l9_bn(x)
        x = self.l9_a(x)

        x = self.l10_c(x)
        x = self.l10_bn(x)
        x = self.l10_a(x)
        x = self.l10_p(x)
        x = self.l10_d(x)

        x = self.l11_c(x)
        x = self.l11_bn(x)
        x = self.l11_a(x)

        x = self.l12_c(x)
        x = self.l12_bn(x)
        x = self.l12_a(x)

        x = self.l13_c(x)
        x = self.l13_bn(x)
        x = self.l13_a(x)
        x = self.l13_p(x)
        x = self.l13_d(x)
        
        
        x = self.flatten(x)

        x = self.l14_fc(x)
        x = self.l14_drop(x)
        
        x = self.l15_fc(x)
        x = self.l15_drop(x)
        
        y = self.l16_fc(x)
        return y

model = VGGNet()

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
