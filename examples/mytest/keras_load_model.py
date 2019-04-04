import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
from examples.mytest import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 重新定义数据格式，归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 载入模型
model = load_model('model_weights.h5')

# 取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('input_layer').output)
dense2_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Dense_1').output)
dense3_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Dense_2').output)

# 获取各层输出信息
dense1_output = dense1_layer_model.predict(x_train)
dense2_output = dense2_layer_model.predict(x_train)
dense3_output = dense3_layer_model.predict(x_train)


print(dense1_output)
print(len(dense1_output))
print(len(dense1_output[0]))



# 翻转矩阵
reverse_dense1_output = utils.reverse_list(dense1_output)
reverse_dense2_output = utils.reverse_list(dense2_output)
reverse_dense3_output = utils.reverse_list(dense3_output)


# 得到各层各个神经元最大值和最小值
dense1_boundary = utils.get_boundary(reverse_dense1_output)
dense2_boundary = utils.get_boundary(reverse_dense2_output)
dense3_boundary = utils.get_boundary(reverse_dense3_output)

# 将最大值最小值保存为csv文件
utils.save_boundary_list(dense1_boundary, 'dense1_boundary.csv')
utils.save_boundary_list(dense2_boundary, 'dense2_boundary.csv')
utils.save_boundary_list(dense3_boundary, 'dense3_boundary.csv')

# 获取各层神经元边界值(csv文件)
dense1_boundary_list = utils.get_boundary_from_file('dense1_boundary.csv')
dense2_boundary_list = utils.get_boundary_from_file('dense2_boundary.csv')
dense3_boundary_list = utils.get_boundary_from_file('dense3_boundary.csv')

print("dense1_boundary_list", len(dense1_boundary_list), ":::", len(dense1_boundary_list[0]))
print("dense2_boundary_list", len(dense2_boundary_list), ":::", len(dense2_boundary_list[0]))
print("dense3_boundary_list", len(dense3_boundary_list), ":::", len(dense3_boundary_list[0]))
print(dense3_boundary_list)





