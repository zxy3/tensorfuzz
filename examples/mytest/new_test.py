from examples.mytest import utils
from examples.mytest import coverage_functions
import numpy as np
from scipy import special
import copy
import tensorflow as tf
from keras.utils import np_utils

from keras.datasets import mnist

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
from examples.mytest import utils
from examples.mytest import coverage_functions
import time


# 对MNIST数据做简单修改，添加噪音
def do_basic_mutations(element, a_min=-1.0, a_max=1.0):

    image, label = element
    sigma = 0.5
    noise = np.random.normal(size=image.shape, scale=sigma)

    mutated_image = noise + image

    mutated_image = np.clip(
        mutated_image, a_min=a_min, a_max=a_max
    )


    mutated_element = [mutated_image, label]
    return mutated_element




(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 重新定义数据格式，归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# # 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# x_test = x_test.tolist()
# y_test = y_test.tolist()
# x_test = x_test * 2
# y_test = y_test * 2
#
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# print(len(x_test))

data = []
for i in range(len(x_test)):
    image_info = []
    image_info.append(x_test[i])
    image_info.append(y_test[i])
    data.append(image_info)

x_test = x_test.tolist()
y_test = y_test.tolist()


# mutated_image_data = []
for i in range(len(data)):
    image_info = data[i]
    # mutated_image_info = do_basic_mutations(image_info)
    for itr in range(10):
        image_info = do_basic_mutations(image_info)
    # x_test.append(mutated_image_info[0])
    # y_test.append(mutated_image_info[1])
    mutated_image_info = image_info
    x_test[i] = mutated_image_info[0]
    y_test[i] = mutated_image_info[1]
    # print(len(mutated_image_info[0]))
    # print(mutated_image_info[0])
    # print(len(mutated_image_info[1]))
    # print(mutated_image_info[1])

x_test = np.array(x_test)
y_test = np.array(y_test)



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
dense1_output = dense1_layer_model.predict(x_test)
dense2_output = dense2_layer_model.predict(x_test)
dense3_output = dense3_layer_model.predict(x_test)

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
# print(time.time() - time1)


time2 = time.time()
coverage2 = coverage_functions.neuron_boundary_coverage(['dense1_boundary.csv', 'dense2_boundary.csv', 'dense3_boundary.csv'],
                                                        input_list)
print("neuron boundary coverage:", coverage2)
# print(time.time() - time2)


time3 = time.time()
coverage3 = coverage_functions.strong_neuron_activation_coverage(['dense1_boundary.csv', 'dense2_boundary.csv', 'dense3_boundary.csv'],
                                                        input_list)
print("strong neuron activation coverage:", coverage3)
# print(time.time() - time3)

time4 = time.time()
coverage4 = coverage_functions.top_k_neuron_coverage(2, input_list)
print("top-k neuron coverage:", coverage4)


time5 = time.time()
patterns_sum = coverage_functions.top_neuron_patterns(2, input_list)
print("top-k patterns:", patterns_sum)

'''
k_multisection coverage: 0.9035483870967742
neuron boundary coverage: 0.25967741935483873
strong neuron activation coverage: 0.5193548387096775
top-k neuron coverage: 0.9290322580645162
top-k patterns: 9700
'''





