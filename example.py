import mlx.nn as nn
from mlx.utils import tree_map
import mlx.core as mx

# model = nn.Linear(10, 10)
# print(model.parameters())
# # dict_keys(['weight', 'bias'])
# print (nn.MultiHeadAttention.create_additive_causal_mask(10))


# a = mx.random.uniform(shape=(1, 4, 2))
# k = mx.random.uniform(shape=(1, 2,2, 2))
# print(k.shape)
# v = mx.random.uniform(shape=(1, 2,2, 2))
# q = mx.random.uniform(shape=(1, 2,4, 2))
# print(q)
# k = mx.concatenate([mx.expand_dims(k, 2)] * 2, axis=2).reshape([1, 2, 4, -1])
# print(k.shape)
# k =k.transpose(0,2,1,3)

# q = q.transpose(0,2,1,3)

# print(q)
# print(k)

# print( (q @ k.transpose(0,1,3,2)).shape)


# k = nn.Linear(2, 4, bias=False)
# print(k.parameters()['weight'].shape
import numpy as np

# Suppose we have the following batch of sequences (already padded)
# inputs = np.array([[1, 2, 3, 0, 0],
#                    [4, 5, 6, 7, 0],
#                    [8, 9, 10, 11, 12]])

# # And these are the actual lengths of the sequences
# lengths = np.array([3, 4, 5])

# # We can create a mask for the valid (non-padding) tokens like this
# print(np.arange(inputs.shape[1])[None,:])
# print(lengths[:, None])
# print(np.arange(inputs.shape[1])[None,:] < [2])
# length_mask = np.arange(inputs.shape[1]) < lengths[:, None]

# print(length_mask)


# logits = [ 0.3, 0.7, 0.1, 0.2, 0.2]
# temp = 3

# logits = mx.array(logits) / temp
# print(logits)
# print(len(logits))
# print(mx.random.categorical(logits, shape=(1, 1)))
import mlx.core as mx

# Define the logits and temperature
logits = mx.array([1.0, 2.0, 2.0,2.0])
temp = 0.5

# Generate a random categorical sample
sample = mx.random.categorical(logits * (1 / temp))

print(sample)
