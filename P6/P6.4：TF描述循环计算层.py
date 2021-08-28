#   TF描述循环计算层
#   tf.keras.layers.SimpleRNN(记忆体个数,activation='激活函数',
#   return_sequences=是否每个时刻输出h_{t}到下一层,默认值=False)
#   activation='激活函数',默认值tanh
#   return_sequences = True 各时间步输出h_{t}
#   return_sequences = False 仅最后时间步输出h_{t}
#   一般最后一层的循环核用False,中间层的循环核用True,每个时间步都把h_{t}输出给下一层
#   SimpleTNN(3,return_sequences=True)

#   API送入循环层数据格式要求：3维[送入样本数,循环核时间展开步数,每个时间步输入特征个数]
#   例：
#   y_{t}
#     ↑
#   h_{t}
#     ↑
#   x_{t}
#   [0.4,1.7,0.6]
#   [0.7,0.9,1.6]
#   RNN层输入维数：[2, 1, 3], 其中2是送入样本数为2, 1是仅有一个时间步, 3是每个样本的特征数


#   y_{t}        y_{t}          y_{t}       y_{t}
#     ↑            ↑              ↑           ↑
#   h_{t}   →    h_{t}   →      h_{t}   →   h_{t}
#     ↑            ↑              ↑           ↑
#   x_{t}        x_{t}          x_{t}       x_{t}
#   [0.4,1.7]   [0.2,1.7]     [0.1,1.1]    [1.1,0.1]
#   RNN层输入维数：[1, 4, 2], 其中1是送入样本数为1, 4是有4个时间步, 2是每个样本的特征数
