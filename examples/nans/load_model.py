"""load model and store boundary of each neuron"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import tflearn
import numpy as np
# import lib.dataset as mnist
from keras.datasets import mnist
from keras.utils import np_utils
import random
from lib import fuzz_utils  # 测试工具
from lib.corpus import InputCorpus
from lib.corpus import seed_corpus_from_numpy_arrays  # 建立seed corpus
from lib.corpus import seed_corpus_from_numpy_arrays_1  # 建立seed corpus
from lib.coverage_functions import all_logit_coverage_function  # 覆盖计算方法
from lib.fuzzer import Fuzzer  # 测试器对象
from lib.mutation_functions import do_basic_mutations  # 变异方法
from lib.sample_functions import recent_sample_function  # 选择样本方法
import pyflann


tf.flags.DEFINE_string(
    "checkpoint_dir", "/tmp/nanfuzzer", "Dir containing checkpoints of model to fuzz."  # 包含模型检查点的目录
)
tf.flags.DEFINE_string(
    "data_dir", "/tmp/mnist", "The directory in which we store the MNIST data"
)

FLAGS = tf.flags.FLAGS

def metadata_function(metadata_batches):   # 获取元数据,[[1, 2, 3], [4, 5, 6], [7, 8, 9]]==>[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    """Gets the metadata."""
    metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
    return metadata_list
# input_layer = tflearn.variables.get_layer_variables_by_name("input_layer")
# hidden_layer_1 = tflearn.variables.get_layer_variables_by_name("hidden_layer_1")

# dataset = mnist.train(FLAGS.data_dir)
# print(dataset.cache().shuffle(buffer_size=50000))
# dataset = dataset.cache().shuffle(buffer_size=50000)
# iterator = dataset.make_one_shot_iterator()
# images, integer_labels = iterator.get_next()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#
# # 重新定义数据格式，归一化
# x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
# x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
print(x_train.shape)
#
# # 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)  # 为将要被记录的日志设置开始入口

    coverage_function = all_logit_coverage_function  # 覆盖计算方法，所有logit的绝对值之和

    image1, label1 = fuzz_utils.basic_mnist_input_corpus(
        choose_randomly=FLAGS.random_seed_corpus  # 这里为False, 返回第一张图片和标签, 图片为28*28*1
    )
    numpy_arrays1 = [[image1, label1]]


    image, label = fuzz_utils.mnist_input_corpus_by_index(index=0)  # 获取训练数据
    # print(image)
    numpy_arrays = [[image, label]]

    print(numpy_arrays == numpy_arrays1)

    with tf.Session() as sess:
        tensor_map = fuzz_utils.get_tensors_from_checkpoint(  # 载入checkpoints
            sess, FLAGS.checkpoint_dir
        )  # 返回字典，包括input,coverage,metadata
        fetch_function = fuzz_utils.build_fetch_function(sess, tensor_map)

        seed_corpus = seed_corpus_from_numpy_arrays(
            # 建立seed corpus，输入集 numpy_array是一张图片,返回的 seed corpus包含一个元素，有metada和coverage信息
            numpy_arrays, coverage_function, metadata_function, fetch_function
        )




        # 构造网络图
        # new_saver = tf.train.import_meta_graph(FLAGS.checkpoint_dir + "/fuzz_checkpoint-34000.meta")
        # new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        # graph = sess.graph

if __name__ == "__main__":
    tf.app.run()






