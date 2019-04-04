import tensorflow as tf
import numpy as np


# input_batches = []
#
# input_batches.append(np.expand_dims(np.array([1,2,3]), axis=0))
# input_batches.append(np.expand_dims(1, axis=0))
# print(input_batches)


metadata_batches = np.array([[1,2,3],[4,5,6]])
metadata_list = [
        [metadata_batches[i][j] for i in range(len(metadata_batches))]
        for j in range(metadata_batches[0].shape[0])
    ]
print(metadata_list)
