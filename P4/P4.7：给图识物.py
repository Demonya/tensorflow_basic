#   前向传播执行应用
#   predict(输入特征, batch_size=)
#   返回前向传播计算结果
#   复现模型：
#   model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

#   加载参数: model.load_weights(model_save_path)
#   预测结果: result = model.predict(x_predict)

from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

prenum = int(input("input the number of test_pictures:\n"))

for i in range(prenum):
    image_path = input("the name of test pictures:")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)  # x_predict输入格式：(1,28,28)  img_arr:(28,28)
    img_arr = np.array(img.convert('L'))

#   特征处理:Method1
    img_arr = 255 - img_arr

#   特征处理:Method2
#     for i in range(28):
#         for j in range(28):
#             if img_arr[i][j] < 200:
#                 img_arr[i][j] = 255
#             else:
#                 img_arr[i][j] = 0
    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)
    print('\n')
