import numpy as np

def generate_mnist_input(low, high, size):
    inputs, images, labels = [], [], []
    for i in range(size):

        image = np.random.uniform(low=low, high=high, size=(28, 28, 1)).tolist()
        label = np.random.randint(low=1, high=10, dtype='int32')
        images.append(image)
        labels.append(label)
    images, labels= np.array(images), np.array(labels)
    inputs = [images, labels]
    return inputs


    # image = np.random.rand(28, 28, 1)
    # print(image)
    # r = np.random.uniform(size=(28, 28, 1))
    # print(r)
    # print(type(r))
    # print(image[0])
    # print(type(image))

    # print(label)
    # print(type(label))
# inputs = generate_mnist_input(low=-1, high=1, size=2)
# print(inputs)
# print(inputs[0][0])
