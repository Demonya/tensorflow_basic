#   import 导入相关包   mnist = tf.keras.datasets.mnist  (x_train,y_train), (x_test, y_test) = mnist.load_data()
#   train、test  自己本领域的数据和标签，如何给x_train,y_train,x_test,y_test赋值。自制数据集解决本领域的实际应用。训练数据少，数据增强提升泛化能力。
#   Sequential/Class
#   model.compile
#   model.fit   断点续训,实时保存最有模型。神经网络训练的目的就是获得最优的参数。参数提取。
#   ACC和Loss曲线可以见证模型的优化过程，给出ACC和Loss曲线绘制代码。给图识物
#   model.summary()

#   本讲目标：神经网络八股功能扩展
#   1）自制数据集,解决本领域应用
#   2）数据增强,扩充数据集
#   3）断点续训,存取模型
#   4）参数提取,把参数存入文本
#   5）acc/loss可视化,查询训练效果
#   6)应用程序，给图识物
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
)

model.fit(x_train, y_train,batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=2)
model.summary()

