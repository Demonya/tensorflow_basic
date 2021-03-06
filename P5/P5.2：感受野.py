#   本讲目标：用CNN实现离散数据的分类（以图像分类为例）
#   卷积计算过程：
#   1）感受野 2）全零填充（padding） 3）TF描述卷积计算层
#   4）批标准化（Batch Normalization， BN）
#   5）池化（pooling） 6）舍弃（dropout）
#   7)卷积神经网络  8）cifar数据集
#   9)卷积神经网络搭建示例
#   10）实现LeNet、AlexNet、VGGNet、InceptionNet、ResNet五个经典卷积网络


#   感受野（Receptive Field）：卷积神经网络各输出特征图中的每个像素点,在原始输入图片上映射区域的大小
#   5*5*1的输入特征,经过两次3*3*1的卷积核输出1*1*1的特征输出,
#   也可经过1次5*5*1的卷积核输出1*1*1的特征输出,其感受野均为5。选择哪一种卷积核？
#   设输入特征图宽、高为x,卷积计算步长为1：
#           经过两次3*3*1卷积核                 经过一次5*5*1卷积核
#   参数量：   9+9 = 18                             25
#   计算量：  18 * x^2 -108 * x -180        25 * x^2 - 200x + 400
#   当x > 10 时,两层3*3卷积核优于一层5*5卷积核
