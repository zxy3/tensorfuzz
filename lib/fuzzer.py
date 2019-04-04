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
"""Defines the actual Fuzzer object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.corpus import CorpusElement
import tensorflow as tf
from lib import generate_input

# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
class Fuzzer(object):
    """Class representing the fuzzer itself."""  # 测试器对象

    def __init__(
        self,
        corpus,  # 指Input Corpus
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
    ):
        """Init the class.

    Args:
      corpus: An InputCorpus object. 输入集对象
      coverage_function: a function that does CorpusElement -> Coverage.覆盖计算方法
      metadata_function: a function that does CorpusElement -> Metadata.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).目标方法
      mutation_function: a function that does CorpusElement -> Metadata. 变异方法 变为元数据
      fetch_function: grabs numpy arrays from the TF runtime using the relevant 获取相应信息
        tensors.
    Returns:
      Initialized object.
    """
        self.corpus = corpus
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""  # 循环测试

        for iteration in range(iterations):
            if iteration % 100 == 0:
                tf.logging.info("fuzzing iteration: %s", iteration)
            parent = self.corpus.sample_input()   # 选择样本输入，返回一个CorpusElement元素

            # Get a mutated batch for each input tensor
            mutated_data_batches = self.mutation_function(parent)  # 生成变异后的数据,100个数据
            # print('===================================')
            # print(mutated_data_batches)

            # Grab the coverage and metadata for mutated batch from the TF runtime. 将变异数据放入网络中运行获取信息
            coverage_batches, metadata_batches = self.fetch_function(
                mutated_data_batches
            )

            # Get the coverage - one from each batch element
            mutated_coverage_list = self.coverage_function(coverage_batches)

            # Get the metadata objects - one from each batch element
            mutated_metadata_list = self.metadata_function(metadata_batches)

            # Check for new coverage and create new corpus elements if necessary.
            # pylint: disable=consider-using-enumerate

            #  生成100个元素，判断是否为新的覆盖以及是否满足objective function
            for idx in range(len(mutated_coverage_list)):
                new_element = CorpusElement(
                    [batch[idx] for batch in mutated_data_batches],
                    mutated_metadata_list[idx],
                    mutated_coverage_list[idx],
                    parent,
                )
                if self.objective_function(new_element):
                    return new_element, iteration
                self.corpus.maybe_add_to_corpus(new_element)
            # tf.logging.info("this iteration completed")
        return None

class generation_based_Fuzzer(object):
    '''
    基于自动生成数据的模糊器
    '''

    def __init__(
            self,
            # inputs,
            coverage_function,
            metadata_function,
            objective_function,
            mutation_function,
            fetch_function,
            generation_function
    ):
        # self.inputs = inputs
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.generation_function = generation_function

    def loop(self, iterations):
        low, high = -1, 1
        for iteration in range(iterations):
            if iteration % 100 == 0:
                tf.logging.info("fuzzing iteration: %s", iteration)
            generated_data_batches = generate_input.generate_mnist_input(low=low, high=high, size=100)
            # generated_data_batches = self.generation_function(low=low, high=high, size=100)
            low, high = low - 10, high + 10
            # Grab the coverage and metadata for mutated batch from the TF runtime. 将变异数据放入网络中运行获取信息
            coverage_batches, metadata_batches = self.fetch_function(
                generated_data_batches
            )

            # Get the coverage - one from each batch element
            generated_coverage_list = self.coverage_function(coverage_batches)

            # Get the metadata objects - one from each batch element
            generated_metadata_list = self.metadata_function(metadata_batches)

            # Check for new coverage and create new corpus elements if necessary.
            # pylint: disable=consider-using-enumerate

            #  每次对100个随机输入做检测
            for idx in range(len(generated_coverage_list)):
                new_element = CorpusElement(
                    [batch[idx] for batch in generated_data_batches],
                    generated_metadata_list[idx],
                    generated_coverage_list[idx],
                    None,
                )
                if self.objective_function(new_element):
                    tf.logging.info('fuzzing succeed iteration %s', iteration)
                    return new_element
                # self.corpus.maybe_add_to_corpus(new_element)
            # tf.logging.info("this iteration completed")
        return None