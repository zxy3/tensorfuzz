import tensorflow as tf
import tflearn

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_xor = [[0.], [1.], [1.], [0.]]

# Graph definition
with tf.Graph().as_default():
    tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
    net = tflearn.input_data(shape=[None, 2], name='inputLayer')
    net = tflearn.fully_connected(net, 2, activation='sigmoid', weights_init=tnorm, name='layer1')
    net = tflearn.fully_connected(net, 1, activation='softmax', weights_init=tnorm, name='layer2')
    regressor = tflearn.regression(net, optimizer='sgd', learning_rate=2., loss='mean_square', name='layer3')

    # Training
    m = tflearn.DNN(regressor)
    m.fit(X, Y_xor, n_epoch=100, snapshot_epoch=False)

    # Testing
    # print("Testing XOR operator")
    # print("0 xor 0:", m.predict([[0., 0.]]))
    # print("0 xor 1:", m.predict([[0., 1.]]))
    # print("1 xor 0:", m.predict([[1., 0.]]))
    # print("1 xor 1:", m.predict([[1., 1.]]))

    layer1_var = tflearn.variables.get_layer_variables_by_name('layer1')
    layer2_var = tflearn.variables.get_layer_variables_by_name('layer2')
    inputLayer_var = tflearn.variables.get_layer_variables_by_name('inputLayer')

    #result = tf.matmul(inputLayer_var, layer1_var[0]) + layer1_var[1]

    with m.session.as_default():
        print(tflearn.variables.get_value(layer1_var[0]))
        print(tflearn.variables.get_value(layer1_var[1]))
        # print(tflearn.variables.get_value(layer2_var[0]))
        # print(tflearn.variables.get_value(layer2_var[1]))