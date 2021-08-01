
# 读入数据集
from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 嵌套循环迭代，with结构更新参数,显示当前loss
epoch = 500
lr = 0.1
test_acc = []
train_loss_results = []
loss_all = 0  # 每轮分4个step,loss_all记录四个step生成的4个loss的和


for epoch in range(epoch):  # 数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别迭代
        with tf.GradientTape() as tape:  # 记录梯度信息
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
    print("Epoch {}, loss:{}".format(epoch, loss_all/4))  # 120组数据,batch 32组数据，batch级别循环4次。求平均损失
    train_loss_results.append(loss_all / 4)
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
