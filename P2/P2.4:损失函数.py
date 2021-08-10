#   损失函数（loss）：预测值（y）与已知答案（y_）的差距
#   NN优化的目标：loss最小化
#   loss大致分为三种类型：（1）均方误差（mse）（2）自定义（3）交叉熵（Cross Entropy）
#   均方误差mse：MSE（y_{-},y）= \frac{\sum_{i=1}^{n}{(y_{-}-y)^2}}{n}
#   loss_mse = tf.reduce_mean(tf.square(y_-y))
#   预测酸奶日销量y,x1、x2是影响日销量的因素。建模前，应先采集的数据有：每日x1、x2和销量y_（即已知答案，最佳情况：产量=销量）
#   生成数据集X,Y_: y_ = x1 + x2  噪声：-0.05 ~ 0.05 拟合可以预测销量的函数。


# loss:MSE
import tensorflow as tf
import numpy as np

rmd = np.random.RandomState(seed=23455)  # 生成【0,1）之间的随机数
x = rmd.rand(32, 2)
y_ = [[x1 + x2 + (rmd.rand()/10 - 0.05)] for (x1, x2) in x]  # 生成噪声
x = tf.cast(x, dtype=tf.float32)

w = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 10000
lr = 0.002

# for epoch in range(epoch):
#     with tf.GradientTape() as tape:
#         y = tf.matmul(x, w)
#         loss_mse = tf.reduce_mean(tf.square(y - y_))
#     grads = tape.gradient(loss_mse, w)
#     w.assign_sub(lr * grads)
#
#     if epoch % 500 == 0:
#         print("After %d epoch, w is " % (epoch))
#         print(w.numpy(), "\n")
#
# print("Final w is: ", w.numpy())

#   自定义损失函数：
#   如预测商品销量，预测多了，损失成本；预测少了，损失利润。若成本 ！= 利润，则MSE产生的loss无法利益最大化
#   loss_zdy = tf.reduce_sum(tf.greater(y,y_), cost(y-y_), profit(y_-y))
#   如预测酸奶销量，酸奶成本cost 1元，利润profit 99元。预测少了损失利润99元，大于预测多了损失成本1元。
#   预测少了损失大，希望生成的预测函数往多了预测。
#   Loss:自定义loss
cost = 1
profit = 99
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss_zdy = tf.reduce_sum(tf.where(tf.greater(y, y_), cost * (y - y_), profit * (y_ - y)))
    grads = tape.gradient(loss_zdy, w)
    w.assign_sub(lr * grads)

    if epoch % 1000 == 0:
        print("After %d epoch, w is " % (epoch))
        print(w.numpy(), "\n")

print("Final w is: ", w.numpy())
print('*' * 50)


#   交叉熵损失函数（cross Entropy）：表征两个概率分布之间的距离。 H(y_{-},y) = -\sum_{}^{}{y_{-}* \ln{y}}
#   eg：已知答案y_ = （1,0） 预测y1 = （0.6,0.4） y2 = （0.8，0.2） 哪个更接近标准答案？
#   H_{1}((1,0),(0.6,04)) = -(1 * ln0.6 + 0* ln0.4) \approx0.511
#   H_{2}((1,0),(0.8,02)) = -(1 * ln0.8 + 0* ln0.2) \approx0.223
#   因为H1 > H2,熵越大越不确定，越小越确定。故y2预测更确定。
#   tf.losses.categorical_crossentropy(y_, y)
loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
# print("loss_ce1:", loss_ce1)
# print("loss_ce2:", loss_ce2)

#   softmax与交叉熵结合：输出先过softmax函数，= np.a再计算y与y_的交叉熵损失函数。
#   tf.nn.softmax_corss_entropy_with_logits(y_, y)
y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)
print("分布计算的结果：\n", loss_ce1)
print("结合计算的结果：\n", loss_ce2)
