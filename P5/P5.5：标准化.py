#   标准化：Batch Normalization, BN
#   标准化：使数据符合0均值,1为标准差的分布。常用在卷积操作和激活操作之间。
#   批标准化：对一小批数据（batch）,做标准化处理
#   批标准化后,第K个卷积核的输出特征图（feature map）中的第i个像素点：
#   H_{i}^{'k} = (H_{i}^{k} - \mu_{batch}^{k} )/ \sigma_{batch}^{k}  （像素点-均值） /标准差
#   H_{i}^{k}：批标准化前,第k个卷积核,输出特征图中的第i个像素点
#   \mu_{batch}^{k}:批标准化前,第k个卷积核,batch张输出特征图中所有像素点平均值
#   \sigma_{batch}^{k}:批标准化前,第k个卷积核,batch张输出特征图中所有像素点标准差
#   \mu_{batch}^{k} = \frac{1}{m}\sum_{1}^{m}{H_{i}^{k}}
#   \sigma_{batch}^{k} = 
