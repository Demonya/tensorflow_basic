#   用tensorFlow API: tf.keras 搭建网络八股
#   六步法：
#   1）import:导入相关模块
#   2）train、test:要喂入神经网络的数据集，训练集特征x_train,训练集标签y_train，测试集特征x_test,测试集标签y_test
#   3）model = tf.keras.models.Sequential ： 在Sequential中搭建网络结构，逐层描述每层网络，相当于走了一遍前向传播。
#   4）model.compile : 配置训练方法，告知训练时选择哪种优化器，哪个损失函数，选择哪种评测指标。
#   5）model.fit : 在fit（）中执行训练过程，告知训练集和测试集的输入特征和标签。 告知每个batch是多少，告知要迭代多少次数据集
#   6）model.summary : 用summary打印出网络的结构和参数统计。


#   model = tf.keras.models.Sequential([网络结构])  #   描述各层网络
#   网络结构举例：
#   拉直层：    tf.keras.layers.Flatten()
#   全连接层    tf.keras.layers.Dense(神经元个数,activation=“激活函数”,kernel_regularizer=哪种正则化)
#   activation（字符串给出）可选：relu、softmax、sigmoid、tanh
#   kernel_regularizer可选：tf.keras.regularizers.l1()、tf.keras.regularizers.l1()
#   卷积层：    tf.keras.layers.Conv2D(filters = 卷积核个数,kernel_size=卷积核尺寸,strides=卷积步长,padding= "vaid"or"same")
#   LSTM层：  tf.keras.layers.LSTM()


#   model.compile(optimizer=优化器,loss=损失函数,metrics=[”准确率“])
#   Optimizer可选：
#      'sgd' or tf.keras.optimizers.SGD(lr=学习率,momentum=动量参数)
#      'adagrad' or tf.keras.optimizers.Adagrad(lr=学习率)
#      'adadelta' or tf.keras.optimizers.Adadelta(lr=学习率)
#      'adam' or tf.keras.optimizers.Adam(lr=学习率,beta_1=0.9,beta_2=0.999)
#   Loss可选:
#       'mse' or tf.keras.losses.MeanSquareError()
#       'sparse_catogorical_corssentropy' or tf.keras.losses.SparseCatogoricalCrossEntropy(from_logits=False)
#       常用损失函数,from_logits可看成是否是原始输出，若为原始输出为True,若经过了概率分布出书则为False


#   Metrics可选:
#       'accuracy':y_和y都是数值。如y_=[1] y=[1]
#       'categorical_accuracy':y_和y都是独热码（概率分布）,如y_=[0,1,0] y=[0.256,0.695,0.048]
#       'sparse_categorical_accuracy':y_是数值,y是独热码（概率分布）如：y_=[1] y=[0.256,0.695,0.048]


#   model.fit(训练集的输入特征,训练集的标签,batch_size=, epochs=, validation_data=(测试的输入特征,测试集的标签),
#   validation_split=从训练集划分多少比例给测试集,validation_freq=多少次epoch测试一次)


#   model.summary()  输出结果

import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np

x_train = load_iris().data
y_train = load_iris().target

np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
