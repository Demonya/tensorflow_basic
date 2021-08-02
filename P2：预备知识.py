import tensorflow as tf
import numpy as np

# tf.where()
# 条件语句真，返回A，条件语句假返回B。
# tf.where(条件语句,真返回A,假返回B)
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)
# print("C {}".format(c))

# np.random.RandomState.rand(维度) 返回一个[0,1)之间的随机数
rdm = np.random.RandomState(seed=1)
a = rdm.rand()  # 返回一个随机标量
b = rdm.rand(2, 3)   # 返回维度为2行3列的随机数矩阵
# print("a:", a)
# print("b:", b)

# np.vstack(数组1, 数组2)
aa = np.array([1, 2, 3, 1, 1])
bb = np.array([0, 1, 3, 4, 5])
cc = np.vstack((aa, bb))
print(cc)


# np.mgrid[], np.ravel(), np.c_[]
# np.mgrid[起始值：结束值：步长,起始值：结束值：步长...]
# np.ravel() 变成1维数组
# np.c_[] 使返回的间隔数值点配对

x, y = np.mgrid[1:3:1, 2:4:0.5]
grid = np.c_[x.ravel(), y.ravel()]
print("x:", x)
print("y:", y)
print("grid: \n", grid)

