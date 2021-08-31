import tushare as ts
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
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
test_set = maotai.iloc[2826-300:, 2:3]

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

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
x_train = np.reshape(x_train, (len(x_train), 60, 1))

for i in range(60, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 60:i, 0])
    y_test.append(test_set_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    SimpleRNN(80, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')   # 损失函数为均方误差
#   该应用只观测loss数值,不观测准确率,所以删去metrics选项,在后续的epoch迭代中只显示loss值
checkpoint_save_path = './rnn_stock/stock.ckpt'

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
real_stock_price = sc.inverse_transform(test_set[60:])

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
