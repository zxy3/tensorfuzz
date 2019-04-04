from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
import keras
from keras.models import Model
from keras.utils import plot_model
import pydot
import matplotlib.pyplot as plt
from keras import utils

import numpy as np
import tensorflow as tf

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("x_shape:", x_train.shape)
# print("y_shape", y_train.shape)

# 重新定义数据格式，归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()

model.add(Dense(200, activation='relu', input_dim=784, name="input_layer"))
model.add(Dense(100, activation='relu', name='Dense_1'))
model.add(Dense(10, activation='softmax', name='Dense_2'))
# model.add(Dense(10, name='Dense_2'))
# model.add(Activation(activation='softmax', name='output'))

# 定义loss
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss)
print('accuracy:', accuracy)


model.save('attack_model_weights.h5')

