# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fuzz a neural network to get a NaN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from lib import fuzz_utils  # 测试工具
from lib.corpus import InputCorpus
from lib.corpus import seed_corpus_from_numpy_arrays  # 建立seed corpus
from lib.corpus import seed_corpus_from_numpy_arrays_1
from lib.coverage_functions import all_logit_coverage_function  # 覆盖计算方法
from lib.fuzzer import Fuzzer  # 测试器对象
from lib.mutation_functions import do_basic_mutations  # 变异方法
from lib.sample_functions import recent_sample_function  # 选择样本方法
import pyflann
from lib import utils


tf.flags.DEFINE_string(
    "checkpoint_dir", "/tmp/nanfuzzer", "Dir containing checkpoints of model to fuzz."  # 包含模型检查点的目录
)
tf.flags.DEFINE_integer(
    "total_inputs_to_fuzz", 100000, "Loops over the whole corpus."  # 循环数
)
tf.flags.DEFINE_integer(
    "mutations_per_corpus_item", 100, "Number of times to mutate corpus item."  # 每个输入变异的次数
)
tf.flags.DEFINE_float(
    "ann_threshold",
    1.0,
    "Distance below which we consider something new coverage.",  # 低于阈值则认为是新的覆盖
)
tf.flags.DEFINE_integer("seed", None, "Random seed for both python and numpy.")
tf.flags.DEFINE_boolean(
    "random_seed_corpus", False, "Whether to choose a random seed corpus."  # 随机选择
)
FLAGS = tf.flags.FLAGS


def metadata_function(metadata_batches):   # 获取元数据,[[1, 2, 3], [4, 5, 6], [7, 8, 9]]==>[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    """Gets the metadata."""
    metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
    return metadata_list


def objective_function(corpus_element):  # 目标方法，检查元数据是否为inf或NaN
    """Checks if the metadata is inf or NaN."""
    metadata = corpus_element.metadata  # 元素的metadata
    if all([np.isfinite(d).all() for d in metadata]):  # 所有元素不为0，FALSE为True。metadata中所有数都为finite满足条件，返回False
        return False

    tf.logging.info("Objective function satisfied: non-finite element found.")  # 找到错误
    return True


def main(_):
    """Constructs the fuzzer and performs fuzzing."""

    # Log more
    tf.logging.set_verbosity(tf.logging.INFO)  # 为将要被记录的日志设置开始入口
    # Set the seeds!
    if FLAGS.seed:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    coverage_function = all_logit_coverage_function  # 覆盖计算方法，所有logit的绝对值之和
    # image, label = fuzz_utils.basic_mnist_input_corpus(
    #     choose_randomly=FLAGS.random_seed_corpus  # 这里为False, 返回第一张图片和标签, 图片为28*28*1
    # )
    # image, label = fuzz_utils.mnist_input_corpus_by_index(index=0)
    # print(len(image))
    # numpy_arrays = [[image, label]]
    #
    # print(len(numpy_arrays))
    # print(len(numpy_arrays[0]))
    with tf.Session() as sess:

        tensor_map = fuzz_utils.get_tensors_from_checkpoint(  # 载入checkpoints
            sess, FLAGS.checkpoint_dir
        )  # 返回字典，包括input,coverage,metadata

        # fetch_function = fuzz_utils.build_fetch_function(sess, tensor_map)  # 返回一个方法还是
        fetch_function = fuzz_utils.build_fetch_function_1(sess, tensor_map)

        size = FLAGS.mutations_per_corpus_item  # 每次变异数量
        mutation_function = lambda elt: do_basic_mutations(elt, size)  # 变异方法
        images, labels = fuzz_utils.all_mnist_input_corpus()
        # numpy_arrays = [[images, labels]]
        elements = []
        for idx in range(50000):
            image, label = images[idx], labels[idx]
            # print(len(image))
            numpy_arrays = [[image, label]]
            # print(len(numpy_arrays))
            # print(len(numpy_arrays[0]))
            seed_corpus = seed_corpus_from_numpy_arrays_1(   # 建立seed corpus，输入集 numpy_array是一张图片,返回的 seed corpus包含一个元素，有metada和coverage信息
                numpy_arrays, coverage_function, metadata_function, fetch_function
            )  # 返回seed corpus，里面包含一个CorpusElement元素
            elements.append(seed_corpus[0])
        hidden_1_list = []
        hidden_2_list = []
        output_list = []
        # hidden_1_list.append(elements[0].hidden_layer_1_before_activation[0][0].tolist())
        # hidden_1_list.append(elements[1].hidden_layer_1_before_activation[0][0].tolist())
        # # print(elements[0].hidden_layer_1_before_activation[0][0].tolist())
        # print(hidden_1_list)
        # print(len(hidden_1_list))
        # print(len(hidden_1_list[0]))
        # reverse_list = utils.reverse_list(hidden_1_list)
        # print(reverse_list)
        # print(len(reverse_list))
        # print(len(reverse_list[0]))

        # for index in range(1):
        for index in range(len(elements)):
            element = elements[index]
            # hidden_1_before_activation = element.hidden_layer_1_before_activation[0][0].tolist()
            # hidden_2_before_activation = element.hidden_layer_2_before_activation[0][0].tolist()
            # output_before_activation = element.output_layer_before_activation[0][0].tolist()
            hidden_1_before_activation = element.hidden_layer_1_after_activation[0][0].tolist()
            hidden_2_before_activation = element.hidden_layer_2_after_activation[0][0].tolist()
            # print(element.hidden_layer_2_after_activation[0][0])
            # print(len(element.hidden_layer_2_after_activation[0][0]))
            # print(type(element.hidden_layer_2_after_activation[0][0]))
            # print('==============================================')
            # print('==========================', type(element.output_layer_before_activation))
            # print(len(element.output_layer_before_activation))
            # print(len(element.output_layer_before_activation[0]))
            # output_before_activation = element.bad_softmax[0].tolist()
            output_before_activation = element.output_layer_before_activation[0][0].tolist()
            # print(output_before_activation[0][0])
            # print(type(output_before_activation[0][0]))

            hidden_1_list.append(hidden_1_before_activation)
            hidden_2_list.append(hidden_2_before_activation)
            output_list.append(output_before_activation)

        # print(hidden_1_list)
        # print(len(hidden_1_list))
        # print(len(hidden_1_list[0]))
        #
        # # print(hidden_2_list)
        # print(len(hidden_2_list))
        # print(hidden_2_list[0])
        #
        # print(len(output_list))
        # print(len(hidden_2_list[0]))


        reverse_hidden_1_list = utils.reverse_list(hidden_1_list)
        reverse_hidden_2_list = utils.reverse_list(hidden_2_list)
        reverse_output_list = utils.reverse_list(output_list)


        # 得到各层各个神经元最大值和最小值
        hidden_1_boundary = utils.get_boundary(reverse_hidden_1_list)
        hidden_2_boundary = utils.get_boundary(reverse_hidden_2_list)
        output_boundary = utils.get_boundary(reverse_output_list)
        #
        # # 将最大值最小值保存为csv文件
        utils.save_boundary_list(hidden_1_boundary, 'hidden_1_boundary.csv')
        utils.save_boundary_list(hidden_2_boundary, 'hidden_2_boundary.csv')
        utils.save_boundary_list(output_boundary, 'output_boundary.csv')









        # seed_corpus = seed_corpus_from_numpy_arrays_1(   # 建立seed corpus，输入集 numpy_array是一张图片,返回的 seed corpus包含一个元素，有metada和coverage信息
        #             numpy_arrays, coverage_function, metadata_function, fetch_function
        # )  # 返回seed corpus，里面包含一个CorpusElement元素

        # corpus = InputCorpus(  # 建立input corpus，一开始只包含一个元素
        #     seed_corpus, recent_sample_function, FLAGS.ann_threshold, "kdtree"   # recent_sample_function用于选择下一个元素
        # )
        # fuzzer = Fuzzer(
        #     corpus,
        #     coverage_function,
        #     metadata_function,
        #     objective_function,
        #     mutation_function,
        #     fetch_function,
        # )
        # result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)
        # if result is not None:
        #     tf.logging.info("Fuzzing succeeded.")
        #     tf.logging.info(
        #         "Generations to make satisfying element: %s.",
        #         result.oldest_ancestor()[1],
        #     )
        # else:
        #     tf.logging.info("Fuzzing failed to satisfy objective function.")


if __name__ == "__main__":
    tf.app.run()
