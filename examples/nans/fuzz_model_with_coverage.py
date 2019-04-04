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
from lib import coverage_functions_1
import time


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
    0.5,
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

    with tf.Session() as sess:

        tensor_map = fuzz_utils.get_tensors_from_checkpoint(  # 载入checkpoints
            sess, FLAGS.checkpoint_dir
        )  # 返回字典，包括input,coverage,metadata

        # fetch_function = fuzz_utils.build_fetch_function(sess, tensor_map)  # 返回一个方法还是
        fetch_function = fuzz_utils.build_fetch_function_1(sess, tensor_map)

        size = FLAGS.mutations_per_corpus_item  # 每次变异数量
        mutation_function = lambda elt: do_basic_mutations(elt, size)  # 变异方法
        images, labels = fuzz_utils.mnist_test_input_corpus()
        # numpy_arrays = [[images, labels]]
        elements = []
        mutated_elements = []
        for idx in range(10000):
            image, label = images[idx], labels[idx]
            numpy_arrays = [[image, label]]
            seed_corpus = seed_corpus_from_numpy_arrays_1(   # 建立seed corpus，输入集 numpy_array是一张图片,返回的 seed corpus包含一个元素，有metada和coverage信息
                numpy_arrays, coverage_function, metadata_function, fetch_function
            )  # 返回seed corpus，里面包含一个CorpusElement元素
            elements.append(seed_corpus[0])
            # print(elements[0].output_layer_before_activation[0][0].tolist())
            # print(elements[0].bad_softmax[0].tolist())

        def generate_mutated_elements(new_elements):
            mutated_inter_elements = []
            for i in range(len(new_elements)):
                mutated_batches = do_basic_mutations(new_elements[i], 1)
                mutated_numpy_arrays = [[mutated_batches[0][0].tolist(), mutated_batches[1].tolist()[0]]]
                mutated_seed_corpus = seed_corpus_from_numpy_arrays_1(
                    # 建立seed corpus，输入集 numpy_array是一张图片,返回的 seed corpus包含一个元素，有metada和coverage信息
                    mutated_numpy_arrays, coverage_function, metadata_function, fetch_function
                )  # 变异后的CorpusElement
                mutated_seed_corpus[0].parent = new_elements[i]
                mutated_inter_elements.append(mutated_seed_corpus[0])
            return mutated_inter_elements

        mutated_elements = generate_mutated_elements(elements)
        iteration = 0
        coverage_functions_1.compute_coverage(mutated_elements)

        # while True:
        #     if iteration % 100 == 0:
        #         print("iteration: ", iteration)
        #     # print(mutated_elements[0].output_layer_before_activation[0][0].tolist())
        #     for index in range(len(mutated_elements[0].output_layer_before_activation[0][0].tolist())):
        #         logit = mutated_elements[0].output_layer_before_activation[0][0].tolist()[index]
        #         if logit >= 88:
        #             print("fuzzed succeed")
        #             break
        #     iteration += 1
        #     mutated_elements = generate_mutated_elements(mutated_elements)






        hidden_1_list = []
        hidden_2_list = []
        output_list = []
        # mutated_hidden_1_list = []
        # mutated_hidden_2_list = []
        # mutated_output_list = []
        #
        #
        #
        # for index in range(len(mutated_elements)):
        #     mutated_hidden_1_list.append(mutated_elements[index].hidden_layer_1_before_activation[0][0].tolist())
        #     mutated_hidden_2_list.append(mutated_elements[index].hidden_layer_2_before_activation[0][0].tolist())
        #     mutated_output_list.append(mutated_elements[index].output_layer_before_activation[0][0].tolist())
        #
        # input_list = []
        # for size in range(len(mutated_hidden_1_list)):
        #     data_size_input_list = []
        #     dense1_sub_list = mutated_hidden_1_list[size]
        #     dense2_sub_list = mutated_hidden_2_list[size]
        #     dense3_sub_list = mutated_output_list[size]
        #
        #     dense1_sub_input_list = []
        #     dense2_sub_input_list = []
        #     dense3_sub_input_list = []
        #     for neuron_sum in range(len(dense1_sub_list)):
        #         dense1_sub_input_list.append([dense1_sub_list[neuron_sum]])
        #     data_size_input_list.append(dense1_sub_input_list)
        #
        #     for neuron_sum in range(len(dense2_sub_list)):
        #         dense2_sub_input_list.append([dense2_sub_list[neuron_sum]])
        #     data_size_input_list.append(dense2_sub_input_list)
        #
        #     for neuron_sum in range(len(dense3_sub_list)):
        #         dense3_sub_input_list.append([dense3_sub_list[neuron_sum]])
        #     data_size_input_list.append(dense3_sub_input_list)
        #
        #     input_list.append(data_size_input_list)
        #
        # output_layer_input_list = []
        # for size in range(len(mutated_output_list)):
        #     data_size_input_list = []
        #     dense3_sub_list = mutated_output_list[size]
        #     dense3_sub_input_list = []
        #
        #     for neuron_sum in range(len(dense3_sub_list)):
        #         dense3_sub_input_list.append([dense3_sub_list[neuron_sum]])
        #     data_size_input_list.append(dense3_sub_input_list)
        #
        #     output_layer_input_list.append(data_size_input_list)
        #
        # time6 = time.time()
        # coverage6 = coverage_functions_1.k_multisection_neuron_coverage(10, ['output_boundary.csv'], output_layer_input_list)
        # print("耗时", time.time() - time6)
        # print("output_layer k_multisection coverage:", coverage6)
        #
        # time7 = time.time()
        # coverage7 = coverage_functions_1.neuron_boundary_coverage(['output_boundary.csv'], output_layer_input_list)
        # print("耗时", time.time() - time7)
        # print("output_layer neuron_boundary coverage:", coverage7)
        #
        # time1 = time.time()
        # coverage1 = coverage_functions_1.k_multisection_neuron_coverage(10, ['hidden_1_boundary.csv', 'hidden_2_boundary.csv', 'output_boundary.csv'],
        #                                                               input_list)
        # print("耗时", time.time() - time1)
        # print("k_multisection coverage:", coverage1)
        #
        # time2 = time.time()
        # coverage2 = coverage_functions_1.neuron_boundary_coverage(
        #     ['hidden_1_boundary.csv', 'hidden_2_boundary.csv', 'output_boundary.csv'],
        #     input_list)
        # print("耗时", time.time() - time2)
        # print("neuron boundary coverage:", coverage2)
        #
        # time3 = time.time()
        # coverage3 = coverage_functions_1.strong_neuron_activation_coverage(
        #     ['hidden_1_boundary.csv', 'hidden_2_boundary.csv', 'output_boundary.csv'],
        #     input_list)
        # print("耗时", time.time() - time3)
        # print("strong neuron activation coverage:", coverage3)
        #
        # time4 = time.time()
        # coverage4 = coverage_functions_1.top_k_neuron_coverage(2, input_list)
        # print("耗时", time.time() - time4)
        # print("top-k neuron coverage:", coverage4)
        #
        # time5 = time.time()
        # coverage5 = coverage_functions_1.top_neuron_patterns(2, input_list)
        # print("耗时", time.time() - time5)
        # print("top-k patterns:", coverage5)






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
