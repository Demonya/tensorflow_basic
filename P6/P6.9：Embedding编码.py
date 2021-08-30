#   Embedding编码方法
#   独热码：位宽要与词汇量一致,数据量大,过于稀疏,映射之间是独立的,没有表现出关联性
#   Embedding:是一种单词编码方式,用低维向量实现了编码,这种编码通过神经网络训练优化,能表达出单词间的相关性
#   tf.keras.layers.Embedding(词汇表大小:也就是编码一共要表示多少个单词,编码维度：用几个数字表达一个单词)
#   对1-100进行编码,[4]编码为[0.25, 0.1, 0.11]
#   tf.keras.layers.Embedding(100, 3)
#   进入到Embedding层的x_train维度要求：[送入样本数, 循环核时间展开步数]
