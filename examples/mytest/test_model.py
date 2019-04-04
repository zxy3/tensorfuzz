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
"""Train a model with MNIST"""

from __future__ import absolute_import  # 绝对引入
from __future__ import division  # 精确除法
from __future__ import print_function  # 输出需要加括号

import os
import lib.dataset as mnist
import tensorflow as tf


tf.flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/testfuzzer",
    "The overall dir in which we store experiments",
)  # 存储模型
tf.flags.DEFINE_string(
    "data_dir", "/tmp/mnist", "The directory in which we store the MNIST data"
)  # 存储MNIST
tf.flags.DEFINE_integer(
    "training_steps", 35000, "Number of mini-batch gradient updates to perform"
)  # 训练步数
tf.flags.DEFINE_float(
    "init_scale", 0.25, "Scale of weight initialization for classifier"
)  # 初始化权重

FLAGS = tf.flags.FLAGS


def classifier(images, init_func):
    """Builds TF graph for clasifying images. 分类

    Args:
        images: TensorFlow tensor corresponding to a batch of images. 一组图片张量
        init_func: Initializer function to use for the layers. 网络中层的初始化方法

    Returns:
      A TensorFlow tensor corresponding to a batch of logits.
    """
    image_input_tensor = tf.identity(images, name="image_input_tensor")  # 与输入图片的shape一致
    net = tf.layers.flatten(image_input_tensor)  # 对图片进行展平操作
    net = tf.layers.dense(net, 200, tf.nn.relu, kernel_initializer=init_func)  # 添加全连接层, 输入，输出个数，激活函数，初始化权重
    net = tf.layers.dense(net, 100, tf.nn.relu, kernel_initializer=init_func)  # 添加全连接层，100个输出
    net = tf.layers.dense(
        net, 10, activation=None, kernel_initializer=init_func
    )  # 添加全连接层
    return net, image_input_tensor


def unsafe_softmax(logits):  # 用于多分类，分类器最后的输出单元需要softmax进行数值处理
    """Computes softmax in a numerically unstable way."""  # 可能会出现数值溢出
    return tf.exp(logits) / tf.reduce_sum(
        tf.exp(logits), axis=1, keepdims=True  # axis=1,keep_dims=True表示按照行的维度求和
    )


def unsafe_cross_entropy(probabilities, labels):  # 描述两个概率分布之间的距离
    """Computes cross entropy in a numerically unstable way."""
    return -tf.reduce_sum(labels * tf.log(probabilities), axis=1)


# pylint: disable=too-many-locals
def main(_):
    """Trains the model."""

    dataset = mnist.train(FLAGS.data_dir)
    dataset = dataset.cache().shuffle(buffer_size=50000).batch(100).repeat()  # 维持一个50000大小的shuffle buffer，每轮取出100个数据
    iterator = dataset.make_one_shot_iterator()  # 通过迭代器读取数据，数据输出一次后就丢弃了
    images, integer_labels = iterator.get_next()  # 获得图片和标签
    images = tf.reshape(images, [-1, 28, 28, 1])  # 改变形状，-1表示不知道大小
    label_input_tensor = tf.identity(integer_labels)  # 标签输入张量
    labels = tf.one_hot(label_input_tensor, 10)  # 使用独热编码对标签编码
    init_func = tf.random_uniform_initializer(  # 初始化权重，生成均匀分布的随机数
        -FLAGS.init_scale, FLAGS.init_scale
    )
    logits, image_input_tensor = classifier(images, init_func)  # 分类,得到logits和图片输入张量
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))  # 比较是否相等，argmax返回最大值的索引号
    accuracy = tf.reduce_mean(tf.to_float(equality))  # 计算正确率


    # This will NaN if abs of any logit >= 88.
    # bad_softmax = unsafe_softmax(logits)
    # softmax = tf.nn.softmax(logits)
    # This will NaN if max_logit - min_logit >= 88.
    # bad_cross_entropies = unsafe_cross_entropy(bad_softmax, labels)
    # loss = tf.reduce_mean(bad_cross_entropies)  # 损失函数

    soft_softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(soft_softmax_cross_entropy)  # 损失函数
    optimizer = tf.train.GradientDescentOptimizer(0.01)  # 优化器

    tf.add_to_collection("input_tensors", image_input_tensor)
    tf.add_to_collection("input_tensors", label_input_tensor)
    tf.add_to_collection("coverage_tensors", logits)  # 覆盖张量
    # tf.add_to_collection("metadata_tensors", bad_softmax)
    # tf.add_to_collection("metadata_tensors", bad_cross_entropies)
    tf.add_to_collection("metadata_tensors", soft_softmax_cross_entropy)
    tf.add_to_collection("metadata_tensors", logits)

    train_op = optimizer.minimize(loss)  # 训练

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    sess = tf.Session()
    sess.run(tf.initialize_all_tables())
    sess.run(tf.global_variables_initializer())

    # train classifier on these images and labels
    for idx in range(FLAGS.training_steps):
        sess.run(train_op)
        if (idx + 1) % 1000 == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy])
            print(idx + 1, ":loss: {}, accuracy: {}".format(loss_val, accuracy_val))
            # print(os.path)
            saver.save(
                sess,
                os.path.join(FLAGS.checkpoint_dir, "fuzz_checkpoint"),
                global_step=idx + 1,
            )  # 保存模型


if __name__ == "__main__":
    tf.app.run()
