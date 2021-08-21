#   读取保存模型
#   读取模型：load_weights（路径文件名）
#   checkpoint_save_path = "./checkpoint/mnist.ckpt"
#   判断是否存在索引表,判断是否保存过参数:
#   if os.path.exists(checkpoint_save_path + '.index'):
#       print('---------------Loading Model---------------')
#       model.load_weights(checkpoint_save_path)

#   保存模型：tf.keras.callbacks.ModelCheckpoint(
#   filepath = 路径文件名,
#   save_weights_only=Ture/False
#   save_best_only = True/False
#   )
#   history = model.fit(callbacks=[cp_callback])

#   cp_callback = tf.keras.callbacks.ModelCheckpoint(
#   filepath=checkpoint_save_path,
#   save_weights_only=True   #   是否只保留模型参数
#   save_best_only=True  #   是否只保留最优结果
# )

#   history = model.fit(x_train, y_train, batchsize=32, epochs=10,
#   validation_data=(x_test, y_test), validation_freq=2,
#   callbacks=[cp_callback])


import tensorflow as tf
import os

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
