#   报错记录：NotImplementedError: Cannot convert a symbolic Tensor (sequential/simple_rnn/strided_slice:0) to a numpy array.
#   This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
#   TensorFlow版本：2.3.0
#   python版本：3.8.10
#   numpy：1.20.3
#   解决方法：将numpy版本调整为1.19.5

#   用RNN实现输入一个字母,预测下一个字母(One hot 编码)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
import os

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}  # 单词映射到数值id的词典

x_train = [w_to_id['a'], w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e']]
y_train = [w_to_id['b'], w_to_id['c'], w_to_id['d'], w_to_id['e'], w_to_id['a']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

#   调整x_train符合Embedding的输入格式要求：[送入样本数, 循环核时间展开步, 每个输入的特征个数]
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)


# 搭建具体三个记忆体的循环层
model = tf.keras.Sequential([
    Embedding(5, 2),  # 需要表示的单词量为5,用2个数字表示每一个单词
    SimpleRNN(3),
    Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/alphabet.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------------Loading Model------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss'  # 由于没有输出测试集,不计算测试集准确率,根据loss保存最优模型
)

history = model.fit(x_train, y_train, batch_size=32, epochs=500, callbacks=[cp_callback])

model.summary()

#   参数写入文件
with open('./weights.txt', 'w') as f:
    for weight in model.trainable_weights:
        f.write(str(weight.name) + '\n')
        f.write(str(weight.shape) + '\n')
        f.write(str(weight.numpy()) + '\n')

#   可视化ACC&Loss
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']


plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuray')
plt.title('Training Accuray')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()

#   predict

preNum = int(input("input the number of test alphabet"))

for i in range(preNum):
    alphabet1 = input("input test alphabet:")
    alphabet = [w_to_id[alphabet1]]

    alphabet = np.reshape(alphabet, (1, 1))
    result = model.predict([alphabet])  # 得到预测结果
    pred = tf.argmax(result, axis=1)  # 选出预测结果概率最大的一个
    pred = int(pred)
    print('第{}次输入预测概率：{}'.format(i, result))
    tf.print(alphabet1 + '->' + input_word[pred])
