#   欠拟合与过拟合
#   欠拟合的解决办法：（1）增加输入项特征（2）增加网络参数（3）减少正则化参数
#   过拟合的解决办法：（1）数据清洗（2）增大训练集（3）采用正则化（4）增大正则化参数

#   正则化缓解过拟合
#   正则化在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声。一般不正则化b
#   loss = loss(y,y_) + REGULARIZER*loss(w)
#   其中loss(y,y_)为模型中所有参数的损失函数如：交叉熵、均方误差等。
#   REGULARIZER为超参数，给出参数w在总的loss中的比例，即正则化的权重
#   loss(w) 需要正则化的参数

#   正则化的选择：
#   L1正则化大概率会使很多参数变为0，因此该方法可通过系数参数，即减少参数的数量，降低复杂度。
#   L2正则化会使参数很接近零但不为零，因此该方法可通过减小参数值的大小降低复杂度。

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])
# print(x_data.shape)


#   这里vstack后reshape有什么用？
x_train = np.vstack(x_data).reshape(-1, 2)  # reshape(-1, 2),不知道有多少行，但有2列，行数计算获得
y_train = np.vstack(y_data).reshape(-1, 1)
# print(x_train.shape)
y_c = [['red' if y else 'blue'] for y in y_train]

#   转化x的数据类型，否则后面矩阵相乘时会因数据类型问题报错

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

#   from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使传入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

#   生成神经网络的参数，输入层为2个神经元，隐藏层为11个神经元，1层隐藏层，输出层为1个神经元
#   用tf.Variable()保证参数可训练

w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01
epoch = 400


#   训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:

            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            #   均方误差MSE
            loss = tf.reduce_mean(tf.square(y_train - y))

            #   添加正则化项
            # loss_regularization= []
            # # loss.append(tf.nn.l2_loss(w1))
            # loss_regularization.append(tf.nn.l2_loss(w2))
            # loss_regularization = tf.reduce_sum(loss_regularization)
            # loss = loss_mse + 0.03 * loss_regularization
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)
        # print(grads)

        #   参数更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

#   预测部分
print('*************************predict*********************')
#   xx在-3到3之间以步长为0.1,yy在-3到3之间以步长为0.1,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]

#   将xx,yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

#   将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
#  使用训练好的参数进行预测
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

x1 = x_data[:, 0]
x2 = x_data[:, 1]

#   probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(y_c))  #   squeeze去掉纬度是1的纬度，相当于去掉【【‘red’】】
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
