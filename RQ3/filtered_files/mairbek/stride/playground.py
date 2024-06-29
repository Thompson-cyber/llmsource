from functools import reduce
from operator import mul


# Playing with examples from
# https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20


def decompose(n, shape):
    """Generated by ChatGPT"""
    size = 1
    for s in shape:
        size *= s

    # Check if the given index is out of bounds
    if n >= size or n < 0:
        raise ValueError("Index out of bounds")

    result = []
    for s in reversed(shape):
        index = n % s
        result.append(index)
        n = (n - index) // s

    # Return the tuple of indices
    return tuple(reversed(result))


def reshape(lst, shape):
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i*n:(i+1)*n], shape[1:]) for i in range(len(lst)//n)]


def _flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _flatten(i):
                yield j
        else:
            yield i


def flatten(container):
    return list(_flatten(container))


def as_strided(xx, strides, shape):
    x = flatten(xx)
    n_shape=reduce(mul, shape)

    result=[]
    for i in range(n_shape):
        ns=decompose(i, shape)
        idx=sum([ni*si for ni, si in zip(ns, strides)])
        result.append(x[idx])

    return reshape(result, shape)

def strided_reshape(xx, shape):
    x = flatten(xx)
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = strides[::-1]
    print("strides", strides)
    return as_strided(x, strides, shape)

def strided_slice(xx, current_shape, slice):
    x = flatten(xx)

    # offsets
    # strides
    # shape