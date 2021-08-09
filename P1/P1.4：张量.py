# 张量：多维数组（列表） 阶：张量的维数
# 维数  阶  名字            例子
# 0-D   0  标量 scalar    s = 1 2 3
# 1-D   1  向量 vector    v = [1,2,3]
# 2-D   2  矩阵 matrix    m = [[1,2,3],[4,5,6],[7,8,9]]
# n-D   n  张量 tensor    t = [[[ n个 就是n维张量
# 数据类型 ： tf.int32, tf.float32,tf.float64,tf.bool,tf.string
import tensorflow as tf
import numpy as np

a = tf.constant([1, 2], dtype=tf.int32)
# print(a)
# print(a.shape)
# print(a.dtype)

b = tf.constant([[1, 2, 3], [2, 3, 3]], dtype=tf.float32)
# print(b)

# 将numpy的数据类型转换为Tensor数据类型
# tf.convert_to_tensor(数据名，dtype=数据类型（可选）)
c = np.arange(0, 5)
d = tf.convert_to_tensor(c, dtype=tf.int64)
# print(c, d)
 
# tf.zeros,tf.ones,tf.fill(维度，指定值)
a1 = tf.zeros([2, 3])
b1 = tf.ones(4)
c1 = tf.fill([2, 3], 1)
# print(a1, b1, c1)

# 生成正态分布随机数，默认均值为0，标准差为1 :  tf.random.normal
# 生成截断式正态分布随机数 :  tf.random.truncated_normal(维度,mean=均值,stddev=标准差) 正负两个σ
# 生成均匀分布随机数： tf.uniform(维度，minval=最小值， maxval=最大值)
a2 = tf.random.normal([3, 3], mean=0, stddev=1)
b2 = tf.random.truncated_normal([3, 3], mean=1, stddev=2)
c2 = tf.random.uniform([2, 2], minval=0, maxval=1)
print(a2)
print(b2)
print(c2)
