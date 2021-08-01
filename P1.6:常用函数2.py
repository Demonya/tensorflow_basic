# with结构记录计算过程,gradient求出张量的梯度
# with tf.GradientTape() as tape:
#    grads = tape.gradient(函数, 对谁求导)

import tensorflow as tf
import numpy as np

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant([3.0, 6.0]))
    loss = tf.pow(w, 2) + w
grads = tape.gradient(loss, w)
print("grads is:", grads)


# enumerate 遍历每个元素,增加索引, enumerate(列表名)
#
# seq = ['one', 'two', 'three']
# for i, element in enumerate(seq):
#     print(i, element)

# tf.one_hot(待转换数据,depth = 几分类):独热编码,将带转换数据转换为one-hot形式
classes = 3
labels = tf.constant([1, 0, 3])
res = tf.one_hot(labels, 4)
# print(res)


# tf.nn.softmax函数,计算各类别概率 softmax(y_{1})= e^y_{i}/Σ (e^y_{i})
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
# print("After softmax,y_pro os:", y_pro)

# assign_sub 自减赋值操作，更新参数的值并返回。前提先用tf.Variable 定义变量w为可训练。
# assgin_sub(要自减的内容)

w.assign_sub(grads)
print(w.numpy())

# tf.argmax(张量名, axis=操作轴  0：纵轴 1:横轴)

test = np.array([[1, 2, 3], [2, 3, 4], [5, 8, 3], [8, 7, 2]])
print(tf.argmax(test, axis=0), tf.argmax(test, axis=1))
