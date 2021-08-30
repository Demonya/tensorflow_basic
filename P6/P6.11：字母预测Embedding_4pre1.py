import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
import matplotlib.pyplot as plt
import os

input_word = ['abcdefghijklmnopqrstuvwxyz']
w_to_id = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
    'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15,
    'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
    'y': 24, 'z': 25
}
training_set_scaled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 24]
x_train = []
y_train = []

for i in range(4, 26):
    x_train.append(training_set_scaled[i - 4:i])
    y_train.append(training_set_scaled[i])



np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

#   调整x_train符合Embedding的输入格式要求：[送入样本数, 循环核时间展开步, 每个输入的特征个数]
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)


# 搭建具体三个记忆体的循环层
model = tf.keras.Sequential([
    Embedding(26, 2),  # 生成一个26行2列的可训练参数矩阵,实现编码可训练
    SimpleRNN(10),
    Dense(26, activation='softmax')  #  输出会是26个字母之一,所以是26
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/alphabet_embedding.ckpt'
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
with open('./Embedding_weights.txt', 'w') as f:
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

    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict([alphabet])  # 得到预测结果
    pred = tf.argmax(result, axis=1)  # 选出预测结果概率最大的一个
    pred = int(pred)
    print('第{}次输入预测概率：{}'.format(i, result))
    tf.print(alphabet1 + '->' + input_word[pred])
