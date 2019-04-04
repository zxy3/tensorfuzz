import os
import numpy as np
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

from urllib.request import urlretrieve
import os
import json
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import mnist

model = load_model('model_weights.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 重新定义数据格式，归一化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# # 转one-hot标签
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None,targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)  邻域
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    # if y is None:
    #     # Using model predictions as ground truth to avoid label leaking
    #     preds_max = tf.reduce_max(preds, 1, keep_dims=True)
    #     y = tf.to_float(tf.equal(preds, preds_max))
    #     y = tf.stop_gradient(y)
    # y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    # loss = utils_tf.model_loss(y, preds, mean=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=preds)
    # if targeted:
    #     loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

# print(x_train[0])
# print(y_train[0])
#
# print(len(x_train[0]))
#
pred = model.predict(x_train[0:1])
# print(pred)
# print(y_train[0:1])
# print(type(pred))
# print(type(x_train[0]))
# print(type(pred[0]))
# print(type(y_train[0]))

x = tf.convert_to_tensor(x_train[0])
pred = tf.convert_to_tensor(pred[0])
y = tf.convert_to_tensor(y_train[0])
print(type(x), type(pred), type(y))
adv_input = fgm(x=x, preds=pred, y=y, clip_min=-1.0, clip_max=1.0)
print(adv_input)


