#  NN复杂度：多用NN层数和NN参数的个数表示
#  空间复杂度：层数 = 隐藏层的层数 + 1个输出层 输入层不算做NN的层数
#  总参数 = 总w + 总b
#  时间复杂度： 乘加运算次数
#  NN结构图中为2层结构：
#      总参数  （第一层）3*4+4 + （第二层）4*2+2 = 26
#      乘加次数 （第一层）3*4 + （第二层）4*2 = 20 线的条数

#  学习率
#  w_{t+1} = w_{t} - lr * \frac{\partial loss}{\partial w_{t}}
#  eg: loss = (w + 1)^2 \frac{\partial loss}{\partial w} = 2w + 2
#  初始化参数w=5,学习率为0.2，则：
#  1次 参数 w: 5  5 - 0.2 * (2 * 5 + 2) = 2.6
#  2次 参数 w: 2.6  2.6 - 0.2 * (2 * 2.6 + 2) = 1.16
#  3次 参数 w: 1.16  1.16 - 0.2 * (2 * 1.16 + 2) = 0.296
#  4次 参数 w: 0.296    ...


#  指数衰减学习率：可以先用较大的学习率，快速得到较优解，然后逐步减小学习率，使模型在训练后期稳定
#  指数衰减学习率 = 初始学习率 * 学习率衰减率 * （当前轮数 / 多少轮衰减一次）

import tensorflow as tf
w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40
LR_BASE = 0.2
LR_DECY = 0.99
LR_STEP = 1

for epoch in range(epoch):
    lr = LR_BASE * LR_DECY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr * grads)
    print('After %s epoch,w is %f, loss is %f,lr is %f' %(epoch, w.numpy(), loss, lr))
