#   标准化：Batch Normalization, BN
#   标准化：使数据符合0均值,1为标准差的分布。常用在卷积操作和激活操作之间。
#   批标准化：对一小批数据（batch）,做标准化处理
#   批标准化后,第K个卷积核的输出特征图（feature map）中的第i个像素点：
#   H_{i}^{'k} = (H_{i}^{k} - \mu_{batch}^{k} )/ \sigma_{batch}^{k}  （像素点-均值） /标准差
#   H_{i}^{k}：批标准化前,第k个卷积核,输出特征图中的第i个像素点
#   \mu_{batch}^{k}:批标准化前,第k个卷积核,batch张输出特征图中所有像素点平均值
#   \sigma_{batch}^{k}:批标准化前,第k个卷积核,batch张输出特征图中所有像素点标准差
#   \mu_{batch}^{k} = \frac{1}{m}\sum_{1}^{m}{H_{i}^{k}}
#   \sigma_{batch}^{k} = \sigma_{batch}^{k} = \sqrt{\delta + \frac{1}{m}\sum_{i=1}^{m}{(H_{i}^{k}-\mu_{batch}^{k})^2}}

#   BN操作将原本偏移的特征数据重新拉回到0均值,使进入激活函数的数据分布在激活函数线性区,
#   使得输入数据的微小变化更明显的体现到激活函数的输出,提升了激活函数对输入数据的区分力。
#   但是这种简单的特征数据标准化,使特征数据完全满足标准正态分布,集中在激活函数中心的线性区域,使激活函数丧失了非线性特性。
#   因此在BN操作中为每个卷积核引入了两个可训练参数,缩放因子\gamma  偏移因子\beta
#   X_{i}^{k} = \gamma_{i}^{k} * H_{i}^{'k} + \beta_{k}
#   反向传播时,缩放因子\gamma和偏移因子\beta会与其他待训练参数一同被训练优化,
#   使标准正态分布后的特征数据通过缩放因子和偏移因子优化了特征数据分布的宽窄和偏移量,保证了网络的非线性表达力。

#   BN层位于卷积层之后,激活层之前；卷积（Convolutional） ->  批标准化（Batch Normalization） ->  激活层（Activation）
#   TF描述标准化：tf.keras.layers.BatchNormalization()

import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D,  Dropout 

model = tf.keras.models.Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), padding='same'),  # 卷积层
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2)  # dropout层
])
