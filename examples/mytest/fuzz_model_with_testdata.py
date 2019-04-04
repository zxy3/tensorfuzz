import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
from examples.mytest import utils
from examples.mytest import coverage_functions
import time
from keras import utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 重新定义数据格式，归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 载入模型
model = load_model('model_weights.h5')
# utils.plot_model(model, to_file='model_weights.png', show_shapes=False, show_layer_names=True)

# 取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('input_layer').output)
dense2_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Dense_1').output)
dense3_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Dense_2').output)

# 获取各层输出信息
dense1_output = dense1_layer_model.predict(x_test)
dense2_output = dense2_layer_model.predict(x_test)
dense3_output = dense3_layer_model.predict(x_test)

print(type(dense1_output))
print(len(dense1_output))
print(len(dense1_output[0]))
print(dense1_output[0])

# print(dense1_output)
# print(len(dense1_output))
# print(len(dense1_output[0]))
# print(len(dense2_output))
# print(len(dense2_output[0]))
# print(len(dense3_output))
# print(len(dense3_output[0]))

input_list = []
for size in range(len(dense1_output)):
    data_size_input_list = []
    dense1_sub_list = dense1_output[size]
    dense2_sub_list = dense2_output[size]
    dense3_sub_list = dense3_output[size]

    dense1_sub_input_list = []
    dense2_sub_input_list = []
    dense3_sub_input_list = []
    for neuron_sum in range(len(dense1_sub_list)):
        dense1_sub_input_list.append([dense1_sub_list[neuron_sum]])
    data_size_input_list.append(dense1_sub_input_list)

    for neuron_sum in range(len(dense2_sub_list)):
        dense2_sub_input_list.append([dense2_sub_list[neuron_sum]])
    data_size_input_list.append(dense2_sub_input_list)

    for neuron_sum in range(len(dense3_sub_list)):
        dense3_sub_input_list.append([dense3_sub_list[neuron_sum]])
    data_size_input_list.append(dense3_sub_input_list)

    input_list.append(data_size_input_list)


# print(len(input_list))
# print(len(input_list[0]))
# print(len(input_list[0][0]))
# print(len(input_list[1][0]))

time1 = time.time()
coverage1 = coverage_functions.k_multisection_neuron_coverage(10, ['dense1_boundary.csv', 'dense2_boundary.csv', 'dense3_boundary.csv'],
                                                           input_list)
print("k_multisection coverage:", coverage1)


time2 = time.time()
coverage2 = coverage_functions.neuron_boundary_coverage(['dense1_boundary.csv', 'dense2_boundary.csv', 'dense3_boundary.csv'],
                                                        input_list)
print("neuron boundary coverage:", coverage2)


time3 = time.time()
coverage3 = coverage_functions.strong_neuron_activation_coverage(['dense1_boundary.csv', 'dense2_boundary.csv', 'dense3_boundary.csv'],
                                                        input_list)
print("strong neuron activation coverage:", coverage3)


time4 = time.time()
coverage4 = coverage_functions.top_k_neuron_coverage(2, input_list)
print("top-k neuron coverage:", coverage4)


time5 = time.time()
coverage5 = coverage_functions.top_neuron_patterns(2, input_list)
print("top-k patterns:", coverage5)

'''
k_multisection coverage: 0.8403225806451613
neuron boundary coverage: 0.21451612903225806
strong neuron activation coverage: 0.4290322580645161
top-k neuron coverage: 0.8096774193548387
top-k patterns: 7438
'''






