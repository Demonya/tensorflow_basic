#   ResNet:提出了层间残差跳连,引入了前方信息,缓解梯度消失,使神经网络增加层数成为可能。
#   单纯堆叠神经网络层数,会使神经网络模型退化,以至于后面的特征丢失了前面特征的原本模样
#   用一根跳连线将前面的特征直接接到后边,使输出结果包含了堆叠卷积的非线性输出和跳过两层堆叠卷积直接连接过来的恒等映射x,
#   让它们对应的元素相加,有效缓解了神经网络模型堆叠导致的退化,使得神经网络可以向着更深层级发展。
#   ResNet中的“+” 与InceptionNet中的“+” 不同的。
#   Inception中的“+”是沿深度方向叠加,相当于千层蛋糕增加层数
#   ResNet块中的“+”是两路特征图对应元素相加,相当于两个矩阵对应元素做加法
#   ResNet块中有两种情况：
#   1）用实线表示：两层堆叠卷积不改变特征图的维度,也就是特征图的个数高、宽、深度都相同,可以直接相加。
#   2）用虚线表示：两层堆叠卷积改变了特征图的维度,需要借助1*1的卷积来调整x的维度,是w_{x}与F_{x}的维度一致。


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
# import numpy as np
import os
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.l1_c = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)
        self.l1_b = BatchNormalization()
        self.l1_a = Activation('relu')

        self.l2_c = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)
        self.l2_b = BatchNormalization()

        #residual_path为True时,对输入进行下采样,即用1*1的卷积核做卷积操作,保证x能和F(x)维度相同,顺利相加
        if residual_path:
            self.down_c = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b = BatchNormalization()
        self.l2_a = Activation('relu')
    def call(self, inputs):
        residual = inputs
        # 将输入通过卷积、BN层、激活层、计算F(x)
        x = self.l1_c(inputs)
        x = self.l1_b(x)
        x = self.l1_a(x)

        x = self.l2_c(x)
        y = self.l2_b(x)
        if self.residual_path:
            residual = self.down_c(inputs)
            residual = self.down_b(residual)

        out = self.l2_a(y + residual)  # 最后输出的是两部分的和,即F(x)+x或F(x)+W(x),再过激活函数
        return out

class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积核
        super(ResNet18, self).__init__()
        self.block_num = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.init_c = Conv2D(self.out_filters, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.init_b = BatchNormalization()
        self.init_a = Activation('relu')
        self.blocks = tf.keras.models.Sequential()

        # 构建ResNet网络结构
        for block_id in range(len(block_list)): # 第几个resnet block
            for layer_id in range(block_list[block_id]): # 第几个卷积层
                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block) # 将构建好的block加入resnet
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAvgPool2D()
        self.f1 = tf.keras.layers.Dense(10)
    def call(self, inputs):
        x = self.init_c(inputs)
        x = self.init_b(x)
        x = self.init_a(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y
model = ResNet18([2, 2, 2, 2])


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


