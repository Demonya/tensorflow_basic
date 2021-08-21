#   提取可训练参数
#   model.trainable_variables 返回模型中可训练参数
#   设置print输出格式: np.set_printoptions(
#   threshold=np.inf  #超过多少省略显示 )
#   print(model.trainable_variables)
#   with open(./'weights.txt', w) as f:
#       for v in model.trainable_variables:
#           f.write(str(v.name), + '\n')
#           f.write(str(v.shape), + '\n')
#           f.write(str(v.numpy()), + '\n')


import tensorflow as tf
import os
import numpy as np
np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------Loading Model----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True
                                                 )
history = model.fit(x_train, y_train, batch_size=32,epochs=5,
                    validation_data=(x_test,y_test),validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

print(model.trainable_weights)
with open('./weights.txt', 'w') as f:
    for v in model.trainable_weights:
        f.write(str(v.name) + '\n')
        f.write(str(v.shape) + '\n')
        f.write(str(v.numpy()) + '\n')
