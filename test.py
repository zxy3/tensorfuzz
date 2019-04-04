import tensorflow as tf
import numpy as np

# tf.flags.DEFINE_string('username', 'tom', 'this is an username')
# tf.flags.DEFINE_integer('age', 12, 'this is an age')
# tf.flags.DEFINE_boolean('gender', True, 'this is the gender')
#
# flags = tf.flags.FLAGS
# print(flags.username)


# x = np.array([[[0], [1], [2]]])
# print(x)
# print(x.shape)  # (1, 3, 1)
# x1 = np.squeeze(x)  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
# print(x1)  # [0 1 2]
# print(x1.shape)  # (3,)

# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# x_shape = x.get_shape()
# print(x_shape)
# x_shape = x.get_shape().as_list()[0]
# print(x_shape)

# x = [1, 2, 3]
# y = [4, 5]
# z = list(zip(x, y))
# print(z)

# x = [1, 3, 2]
# # print(sorted(x))
# print(x[0:])

# print(all([True, True, False]))

# import random
#
#
# random.seed(0)
# for i in range(10):
#     print(random.random())
#
# print("==================>")
# random.seed(0)
# for i in range(10):
#     print(random.random())


# dataset = tf.data.Dataset.range(5)
# iterator = dataset.make_one_shot_iterator()
#
# with tf.Session() as sess:
#     while True:
#         try:
#             print(sess.run(iterator.get_next()))
#         except tf.errors.OutOfRangeError:
#             break

# shape = [2, 3]
# initial = tf.truncated_normal(shape=shape, stddev=0.01)
# print(tf.Variable(initial))


# scores = np.array([123, 456, 789])
# print(scores)
# print("max:", np.max(scores))
# scores -= np.max(scores)
# print(scores)
# p = np.exp(scores) / np.sum(np.exp(scores))
# print(p)
# print(max(p))

# batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# # print(len(batches))
# metadata_list = [
#     [batches[i][j] for i in range(len(batches))]
#     for j in range(len(batches[0]))
# ]
#
# print(metadata_list)

list1 = [1, 2, 3]
list2 = [4, 5, 6]
# print(list1 + list2)
# list3 = [x for x in range(5)] + list1
# print(list3)
feed_dict = {}
for l1, l2 in list(zip(list1, list2)):
    feed_dict[l1] = l2

print(feed_dict)