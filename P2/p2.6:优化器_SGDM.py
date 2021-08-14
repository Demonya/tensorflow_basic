#   神经网络参数优化器：引导神经网络更新参数的工具
#   五种常用的神经网络优化器：1)SGD 2)SGDM 3)
#   待优化参数w，损失函数loss，学习率lr，每次迭代一个batch，t表示当前batch迭代的总次数
#   1、计算t时刻损失函数关于当前参数的梯度 g_{t} = \frac{\partial{loss}}{\partial{w}}
#   2、计算t时刻一阶动量m_{t},和二阶动量V_{t}
#   3、计算t时刻下降梯度：\eta_{t} = lr * m_{t}/\sqrt{V_{t}}
#   4、计算t+1时刻参数：w_{t+1} = w_{t} - \eta_{t} =  w_{t} -  lr * m_{t}/\sqrt{V_{t}}
#   一阶动量：与梯度相关的函数  二阶动量：与梯度平方相关的函数
#   不同的优化器只是定义了不同的一阶动量和二阶动量的公式,无二阶动量，V_{t}默认为1

#   （2）SGDM (含momentum的SGD)，在SGD基础上增加一阶动量。  参数更新时以上一个时刻的一阶动量为主
#   m_{t} = \beta * w_{t-1} + (1-\beta) * g_{t}       V_{t} = 1
#   \eta_{t} = lr * m_{t}/\sqrt{V_{t}} = lr * m_{t} = lr * (\beta * w_{t-1} + (1-\beta) * g_{t})
#   w_{t+1} = w_{t} - \eta_{t} =  w_{t} -  lr * m_{t}/\sqrt{V_{t}} = w_{t} - lr * (\beta * w_{t-1} + (1-\beta) * g_{t})


#   参数更新
#   SGDM (含momentum的SGD)
#   m_{t} = \beta * m_{t-1} + （1- \beta）* g_{t}
#   m_w , m_b = 0,0   beta = 0.9

#   m_w = \beta * m_w + (1- \beta) * grad[0]
#   m_b = \beta * m_b + (1- \beta) * grad[1]
#   w1.assign_sub(lr * m_w)
#   b1.assign_sub(lr * m_b)

# 读入数据集
from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time as time

x_data = load_iris().data
y_data = load_iris().target

# 数据集乱序
np.random.seed(116)  # 使用相同的seed,使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 划分数据集为训练集和测试集
x_train, y_train, x_test, y_test = x_data[:-30], y_data[:-30], x_data[-30:], y_data[-30:]
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
# 将特征和标签对应,每次喂入一小撮数据（batch）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络中所有可训练参数
# 生成神经网络的参数，4个输入特征故输入层为4个输入节点；3分类故输出层为3个神经元
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 嵌套循环迭代，with结构更新参数,显示当前loss
epoch = 500
lr = 0.1
test_acc = []
train_loss_results = []
loss_all = 0  # 每轮分4个step,loss_all记录四个step生成的4个loss的和
beta = 0.9
m_w , m_b = 0, 0

start_time = time.time()
for epoch in range(epoch):  # 数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别迭代
        with tf.GradientTape() as tape:  # 记录梯度信息
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])
       #############################################################
        #  SGDM更新参数
        m_w = beta * m_w + (1 - beta) * grads[0]
        m_b = beta * m_b + (1 - beta) * grads[1]
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        #############################################################

    print("Epoch {}, loss:{}".format(epoch, loss_all/4))  # 120组数据,batch 32组数据，batch级别循环4次。求平均损失
    train_loss_results.append(loss_all / 4)
    print(train_loss_results)
    loss_all = 0

    # 测试部分
    total_correct, total_number = 0, 0
    # 计算当前参数前向传播后的准确率,显示当前acc
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1  # y为预测结果
        y = tf.nn.softmax(y)  # y符合概率分布
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引,即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)  # 将每个batch中的correct数加起来
        total_correct += int(correct)  # 将所有batch中的correct数加起来
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print('-' * 50)
total_time = time.time() - start_time
print("total_time:", total_time)
# acc / loss 可视化

# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuracy$')
plt.legend()
plt.show()

