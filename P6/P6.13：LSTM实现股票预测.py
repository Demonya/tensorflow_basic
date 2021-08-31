#   传统循环网络RNN可以通过记忆体实现短期记忆进行连续数据的预测,但是当连续数据的
#   的序列变长时,会使展开时间步过长,在反向传播更新参数时,梯度要按照时间步连续相乘
#   会导致梯度消失。

#   长短记忆网络：LSTM
#   引入了三个门限:输入门 i_{t}   遗忘门 f_{t}   输出门o_{t}
#   引入了表征长期记忆的细胞态C_{t}, 引入了等待存入长期记忆的候选态  \tilde{C_{t}}
#   引入了表征短期记忆的记忆体h_{t}

#   三个门限：都是当前时刻的输入特征x_{t}和上个时刻的短期记忆h_{t-1}的函数,
#   三个公式中W_{i}、W_{f}、W_{o}都是带训练参数矩阵,b_{i}、b_{f}、b_{o}是
#   待训练偏置项,都经过sigmoid激活函数是门限的范围在0-1之间


#   输入门（门限）：i_{t} = \sigma(W_{i} * [h_{t-1}, x_{t}] + b_{i})
#   遗忘门（门限）：f_{t} = \sigma(W_{f} * [h_{t-1}, x_{t}] + b_{f})
#   输出门（门限）：i_{t} = \sigma(W_{o} * [h_{t-1}, x_{t}] + b_{o})

#   候选态（归纳出的新知识）：\tilde{C_{t}} = tanh(W_{c} * [h_{t-1}, x_{t}] + b_{c})
#   细胞态（长期记忆）：C_{t} = f_{t} * C_{t-1} + i_{t} * \tilde{C_{t}}
#   记忆体（短期记忆）：h_{t} = o_{t} * tanh(C_{t})

#   LSTM理解:
#   LSTM就是听课的过程：现在脑袋里记住的内容是今天PPT第1页到第45页的长期记忆C_{t},
#   长期记忆C_{t}由两部分组成：
#   一部分是PPT第1页到第44页的内容,也就是上一时刻的长期记忆C_{t-1},我们不可能一字不差的
#   记住全部内容,所以上个时刻的长期记忆需要乘以遗忘门,该乘积项表示留存在大脑中的对过去的记忆。
#   另一部分是现在讲的内容是新知识，是即将存在大脑中的现在的记忆。现在的记忆由两部分组成：
#   一部分是正在讲解的第45页的PPT,是当前时刻的输入x_{t],还有一部分是第44页PPT的短期记忆留存,
#   这是上一时刻的短期记忆h_{t-1},大脑把当前时刻的输入x_{t}和上一时刻的短期记忆h_{t-1}归纳形成
#   即将存入脑中的现在的记忆 \tilde{C_{t}},现在的记忆\tilde{C_{t}}乘以输入门与过去的记忆一同
#   存储为长期记忆,当把这一讲的内容复述给朋友时，有些已经被遗忘。讲述的是留存在大脑中的
#   长期记忆经过输出门筛选后的内容,这就是记忆体的输出h_{t},当有多层循环网络时,第二次循环网络的输入
#   x_{t}就是第一次循环网络的输出h_{t}。输入第二次网络的是第一次网络提取出的精华。

#   TF描述LSTM层
#   tf.keras.layers.LSTM(记忆体个数,return_sequences=是否返回输出)
#   return_sequences=True 各时间步输出h_{t}
#   return_sequences=False 仅最后时间步输出h_{t} 默认值
#   model = tf.keras.Sequential([
#       LSTM(80, return_sequences=True),
#       Dropout(0.2),
#       LSTM(100),
#       Dropout(0.2),
#       Dense(1)
#   ])

import tushare as ts
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

df1 = ts.get_k_data('600519', ktype='D', start='2010-04-26', end='2021-08-31')
df1.to_csv('./SH600519.csv')

maotai = pd.read_csv('./SH600519.csv')

training_set = maotai.iloc[0:2826 - 300, 2:3].values
test_set = maotai.iloc[2826-300:, 2:3].values

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

for i in range(60, len(test_set)):
    x_test.append(test_set[i - 60:i, 0])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')   # 损失函数为均方误差
#   该应用只观测loss数值,不观测准确率,所以删去metrics选项,在后续的epoch迭代中只显示loss值
checkpoint_save_path = './LSTM_stock/LSTM_stock.ckpt'

if os.path.exists(checkpoint_save_path + '.index'):
    print('------------Loading Model-------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=64, epochs=50,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])

model.summary()

with open('./rnn_weight1.txt', 'w') as f:
    for weight in model.trainable_weights:
        f.write(str(weight.name) + '\n')
        f.write(str(weight.shape) + '\n')
        f.write(str(weight.numpy()) + '\n')

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training AND Validation Loss')
plt.legend()
plt.show()

# predict
predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price = sc.inverse_transform(test_set_scaled[60:])

plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='MaoTai Stock Predict Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()

#   evaluate
mse = mean_squared_error(predicted_stock_price, real_stock_price)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
mae = mean_absolute_error(predicted_stock_price, real_stock_price)

print('均方误差： %.6f' % mse)
print('均方根误差： %.6f' % rmse)
print('平方绝对误差： %.6f' % mae)
