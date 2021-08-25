#   InceptionNet引入了Inception结构块,在同一层网络内使用不同尺寸的卷积核
#   提升了模型感知力,使用了批标准化缓解了梯度消失
#   Inception结构块：四个分支：

#   第一分支：
#   C(核:16*1*1,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)

#   第二分支：
#   C(核:16*1*1,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)

#   C(核:16*3*3,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)

#   第三分支：
#   C(核:16*1*1,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)

#   C(核:16*5*5,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)

#   第四分支
#   P(max,核：3*3,步长1, 填充：same)

#   C(核:16*1*1,步长：1, 填充：same)
#   B(YES)
#   A(relu)
#   P(None)
#   D(None)


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Dense, Flatten ,GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

class ConvBNRelu(Model):
    def __init__(self, conv_filter_num, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(conv_filter_num, kernel_size=kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])
    def call(self, x):
        x = self.model(x)
        return x

class InceptionBLK(Model):
    def __init__(self, cf_num, strides=1):
        super(InceptionBLK, self).__init__()
        self.ch = cf_num
        self.strides = strides
        self.branch1 = ConvBNRelu(cf_num, kernel_size=1, strides=strides)
        self.branch2_step1 = ConvBNRelu(cf_num, kernel_size=1, strides=strides)
        self.branch2_step2 = ConvBNRelu(cf_num, kernel_size=3, strides=1)
        self.branch3_step1 = ConvBNRelu(cf_num, kernel_size=1, strides=strides)
        self.branch3_step2 = ConvBNRelu(cf_num, kernel_size=5, strides=1)
        self.branch4_step1 = MaxPool2D(3, strides=1, padding='same')
        self.branch4_step2 = ConvBNRelu(cf_num, kernel_size=1, strides=strides)
    def call(self, x):
        branch1_x = self.branch1(x)
        branch2_x1 = self.branch2_step1(x)
        branch2_x2 = self.branch2_step2(branch2_x1)
        branch3_x1 = self.branch3_step1(x)
        branch3_x2 = self.branch3_step2(branch3_x1)
        branch4_x1 = self.branch4_step1(x)
        branch4_x2 = self.branch4_step2(branch4_x1)

        x = tf.concat([branch1_x, branch2_x2, branch3_x2, branch4_x2])
        return x

class Inception10(Model):
    def __init__(self, block_num, class_num, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channel = init_ch
        self.out_channel = init_ch
        self.block_num = block_num
        self.init_ch = init_ch
        self.init_conv = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(block_num):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBLK(self.out_channel, strides=2)
                else:
                    block = InceptionBLK(self.out_channel, strides=1)
                self.blocks.add(block)
            self.out_channel *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(10, activation='softmax')
    def call(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y



model = Inception10(block_num=2, class_num=10)

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
