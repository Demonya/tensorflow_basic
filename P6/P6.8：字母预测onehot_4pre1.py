#   用RNN实现输入连续四个字母,预测下一个字母（One hot编码）
#   输入abcd 输出e 输入bcde 输出a 输入cdea 输出b 输入deab 输出c 输入eabc 输出d
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import os

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_to_onehot = {
    0: [1., 0., 0., 0., 0.],
    1: [0., 1., 0., 0., 0.],
    2: [0., 0., 1., 0., 0.],
    3: [0., 0., 0., 1., 0.],
    4: [0., 0., 0., 0., 1.]
}
x_train = [
    [id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']]],
    [id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']]],
    [id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']]],
    [id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']]],
    [id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']]]
]
y_train = [w_to_id['e'], w_to_id['a'], w_to_id['b'], w_to_id['c'], w_to_id['d']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

#   将x_train调整为SimpleRNN要求的格式：（输入样本数, 循环核时间展开步数, 每个输入样本的特征数）
#   此处整个数据集送入所以送入,送入样本数为len(x_train)；输入4个字母出结果,所以 循环核展开时间步为4,
#   表示为独热码有5个输入特征,每个时间步输入特征个数为5
x_train = np.reshape(x_train, (len(x_train), 4, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    SimpleRNN(3),  # 3个记忆体
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './rnn_checkpoint/rnn.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('--------------Loading Model--------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 monitor='loss'  # 由于fit没有输出测试集,不计算测试集准确率,根据loss选择最优模型
                                                 )

history = model.fit(x_train, y_train, batch_size=32, epochs=100,callbacks=[cp_callback])

model.summary()

with open('./rnn_weight.txt', 'w') as f:
    for weight in model.trainable_weights:
        f.write(str(weight.name) + '\n')
        f.write(str(weight.shape) + '\n')
        f.write(str(weight.numpy()) + '\n')

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training ACC')
plt.title('Training ACC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()

#   predict
preNum = int(input("input the number of test alphabet:"))
for i in range(preNum):
    alphabet1 = input("input test alphabet:")
    alphabet = [id_to_onehot[w_to_id[a]] for a in alphabet1]
    alphabet = np.reshape(alphabet, (1, 4, 5))
    result = model.predict([alphabet])
    print(result)
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])
