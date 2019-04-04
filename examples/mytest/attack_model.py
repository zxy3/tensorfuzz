from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
import pydot
import matplotlib.pyplot as plt

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

# model = load_model('attack_model_weights.h5')
model = load_model('attack_model_weights.h5')

no_softmax_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense_2').output)
softmax_layer_model = Model(inputs=model.input, outputs=model.get_layer('output').output)

# logits = no_softmax_layer_model.predict(x_train[0:1])
# y = softmax_layer_model.predict(x_train[0:1])
# print("no_softmax:", logits)
# print("softmax:", y)
# print("y_train:", y_train[0:1])


def cw(x, y=None, eps=1.0, ord_=2, T=2, optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9, sess=None,
       min_prob=0, clip=(0.0, 1.0), pred=None, logit=None):
    """CarliniWagner (CW) attack.

    Only CW-L2 and CW-Linf are implemented since I do not see the point of
    embedding CW-L2 in CW-L1.  See https://arxiv.org/abs/1608.04644 for
    details.

    The idea of CW attack is to minimize a loss that comprises two parts: a)
    the p-norm distance between the original image and the adversarial image,
    and b) a term that encourages the incorrect classification of the
    adversarial images.

    Please note that CW is a optimization process, so it is tricky.  There are
    lots of hyper-parameters to tune in order to get the best result.  The
    binary search process for the best eps values is omitted here.  You could
    do grid search to find the best parameter configuration, if you like.  I
    demonstrate binary search for the best result in an example code.

    :param model: The model wrapper.
    :param x: The input clean sample, usually a placeholder.  NOTE that the
              shape of x MUST be static, i.e., fixed when constructing the
              graph.  This is because there are some variables that depends
              upon this shape.
    :param y: The target label.  Set to be the least-likely label when None.
    :param eps: The scaling factor for the second penalty term.
    :param ord_: The p-norm, 2 or inf.  Actually I only test whether it is 2
        or not 2.
    :param T: The temperature for sigmoid function.  In the original paper,
              the author used (tanh(x)+1)/2 = sigmoid(2x), i.e., t=2.  During
              our experiment, we found that this parameter also affects the
              quality of generated adversarial samples.
    :param optimizer: The optimizer used to minimize the CW loss.  Default to
        be tf.AdamOptimizer with learning rate 0.1. Note the learning rate is
        much larger than normal learning rate.
    :param alpha: Used only in CW-L0.  The decreasing factor for the upper
        bound of noise.
    :param min_prob: The minimum confidence of adversarial examples.
        Generally larger min_prob wil lresult in more noise.
    :param clip: A tuple (clip_min, clip_max), which denotes the range of
        values in x.

    :return: A tuple (train_op, xadv, noise).  Run train_op for some epochs to
             generate the adversarial image, then run xadv to get the final
             adversarial image.  Noise is in the sigmoid-space instead of the
             input space.  It is returned because we need to clear noise
             before each batched attacks.
    """
    x = tf.reshape(x, (28, 28, 1))
    x_numpy = x.eval()
    # print(x_numpy)
    x_list = x_numpy.tolist()
    # print(len(x_list[0][0]))
    for size1 in range(len(x_list)):
        sub_list = x_list[size1]
        for size2 in range(len(sub_list)):
            sub_list[size2][0] = 0

    xshape = x.get_shape().as_list()
    # print(xshape)
    noise = tf.get_variable('noise', xshape, tf.float64,
                            initializer=tf.initializers.zeros)
    sess.run(noise.initializer)
    # print(noise.eval())
    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])
    # print(sess.run(x_scaled))
    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
    xinv = tf.log(z / (1 - z)) / T
    # print(type(xinv))
    #
    # # add noise in sigmoid-space and map back to input domain
    xadv = tf.sigmoid(T * (xinv + noise))
    xadv = xadv * (clip[1] - clip[0]) + clip[0]
    #
    # ybar, logits = model(xadv, logits=True)
    ybar, logits = pred, logit
    # print(ybar, logits)
    ydim = tf.convert_to_tensor(ybar).get_shape().as_list()[1]
    # print(ydim)
    # print(type(ybar))
    #
    ybar = tf.convert_to_tensor(ybar)
    logits = tf.convert_to_tensor(logits)
    print(ybar, logits)
    if y is not None:
        y = tf.cond(tf.equal(tf.rank(y), 0),
                    lambda: tf.fill([xshape[0]], y),
                    lambda: tf.identity(y))
    else:
        # we set target to the least-likely label
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)
    print(yt, yo)

    # encourage to classify to a wrong category
    loss0 = tf.nn.relu(yo - yt + min_prob)
    print(loss0)

    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)
    print(axis, ord_)
    # make sure the adversarial images are visually close
    if 2 == ord_:
        # CW-L2 Original paper uses the reduce_sum version.  These two
        # implementation does not differ much.

        # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
        loss1 = tf.reduce_mean(tf.square(xadv-x))
        print(loss1)
        loss1 = tf.cast(loss1, tf.float32)
        print(loss1)
    else:
    #     # CW-Linf
        tau0 = tf.fill([xshape[0]] + [1]*len(axis), clip[1])
        tau = tf.get_variable('cw8-noise-upperbound', dtype=tf.float32,
                              initializer=tau0, trainable=False)
        diff = xadv - x - tau

        # if all values are smaller than the upper bound value tau, we reduce
        # this value via tau*0.9 to make sure L-inf does not get stuck.
        tau = alpha * tf.to_float(tf.reduce_all(diff < 0, axis=axis))
        loss1 = tf.nn.relu(tf.reduce_sum(diff, axis=axis))
    #
    loss = eps*loss0 + loss1
    print("==========================loss", loss)
    # print(type(loss))
    train_op = optimizer.minimize(loss, var_list=[noise])
    print(type(train_op))
    # #
    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        train_op = tf.group(train_op, tau)

    return train_op, xadv, noise




# sess = tf.InteractiveSession()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    x = tf.convert_to_tensor(x_train[0:1])
    pred = softmax_layer_model.predict(x_train[0:1])
    logit = no_softmax_layer_model.predict(x_train[0:1])
    train_op, xadv, noise = cw(x=x, sess=sess, pred=pred, logit=logit)
    print(train_op, xadv, noise.eval())

#
# pred = model.predict(x_train[0:1])
# y = tf.nn.softmax(pred, axis=1)
# y = sess.run(y)
# # print(pred)
# # print(y)
#
# x_adv = fgm(x=x_train[0:1], sess=sess)
# print(softmax_layer_model.predict(x_train[0:1]))
# print(model.predict(x_train[0:1]))







# def fgm(x=None, eps=0.01, epochs=1,sess=None, sign=True, clip_min=0., clip_max=1.):
#     """
#     Fast gradient method.
#
#     See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
#     for details.  This implements the revised version since the original FGM
#     has label leaking problem (https://arxiv.org/abs/1611.01236).
#
#     :param model: A wrapper that returns the output as well as logits.
#     :param x: The input placeholder.
#     :param eps: The scale factor for noise.
#     :param epochs: The maximum epoch to run.
#     :param sign: Use gradient sign if True, otherwise use gradient value.
#     :param clip_min: The minimum value in output.
#     :param clip_max: The maximum value in output.
#
#     :return: A tensor, contains adversarial samples for each input.
#     """
#     # xadv = tf.identity(x)
#     xadv = x
#     # ybar = model(xadv)
#     ybar = softmax_layer_model.predict(xadv)
#     print(xadv)
#     print(ybar)
#     print(y_train[0:1])
#     yshape = tf.convert_to_tensor(ybar).get_shape().as_list()
#     print(yshape)
#     ydim = yshape[1]
#     print(ydim)
#     #
#     indices = sess.run(tf.argmax(ybar, axis=1))
#     print(indices)
#
#     target = tf.cond(
#         tf.equal(ydim, 1),
#         lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
#         lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))
#
#     if 1 == ydim:
#         loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
#     else:
#         loss_fn = tf.nn.softmax_cross_entropy_with_logits
#
#     if sign:
#         noise_fn = tf.sign
#     else:
#         noise_fn = tf.identity
#
#     eps = tf.abs(eps)
#     print("eps:::", eps)
#
#     def _cond(xadv, i):
#         # return sess.run(tf.less(i, epochs))
#         print("比较======================：》")
#         return i < epochs
#
#     def _body(xadv, i):
#         print("body==========================>")
#         ybar = softmax_layer_model.predict(xadv)
#         print("===========================>", ybar)
#         logits = no_softmax_layer_model.predict(xadv)
#
#
#         loss = loss_fn(labels=target, logits=logits)
#         dy_dx, = tf.gradients(loss, xadv)
#         xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
#         xadv = tf.clip_by_value(xadv, clip_min, clip_max)
#         return xadv, i+1
#
#     #
#     # i = 0
#     # while i < epochs:
#     #     ybar, logits = softmax_layer_model.predict(xadv), no_softmax_layer_model.predict(xadv)
#     #     loss = loss_fn(labels=target, logits=logits)
#     #     # print(loss)
#     #     # print(type(xadv))
#     #     dy_dx = tf.gradients(loss, tf.convert_to_tensor(xadv))
#     #     print(dy_dx)
#     #     i += 1
#     xadv, _ = sess.run(tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
#                             name='fast_gradient'))
#
#
#     return xadv


