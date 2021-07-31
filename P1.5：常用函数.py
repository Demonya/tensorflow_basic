# 强制tensor转换为该数据类型
# tf.cast(张量名, dtype=数据类型)

# 计算张量维度上元素的最小值
# tf.reduce_min(张量名)

# 计算张量维度上元素的最大值
# tf.reduce_max(张量名)

import tensorflow as tf
x1 = tf.constant([1, 2, 3], dtype=tf.float64)
# print(x1)
x2 = tf.cast(x1, tf.int32)
# print(x2)
# print(tf.reduce_min(x1), tf.reduce_max(x1))

# axis=0 纵向操作 axis=1 横向操作

x3 = tf.constant([[1, 2, 3], [3, 4, 5]])
# print(tf.reduce_mean(x3), tf.reduce_mean(x3, axis=0), tf.reduce_mean(x3, axis=1) )

# tf.Variable() 将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。
# tf.Variable(初始值)

x4 = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print(x4)


# 数学运算：加减乘除、平方、次方、开方 矩阵乘,只有维度相同的才可以进行四则运算（加减乘除）
# tf.add(张量1, 张量2)
# tf.subtract(张量1, 张量2),
# tf.multiply(张量1, 张量2),
# tf.divide(张量1, 张量2),
# tf.square(张量),
# tf.pow(张量, n次方),
# tf.sqrt(张量),
# tf.matmul(矩阵1, 矩阵2)

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
# print(tf.add(a, b))
# print(tf.subtract(a, b))
# print(tf.multiply(a, b))
# print(tf.divide(a, b))

# print(tf.pow(b, 3), tf.square(b), tf.sqrt(b))

c = tf.ones([4, 2])
d = tf.fill([2, 3], 2.)
print(tf.matmul(c, d))


# 切分传入张量的第一维度，生成输入特征/标签对，构建数据集
# data = tf.data.Dataset.from_tensor_slices(输入特征，标签)
# numpy 和 Tensor格式都可用该语句读入数据

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)
