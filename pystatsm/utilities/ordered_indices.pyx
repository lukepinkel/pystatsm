from itertools import product
from libc.stdlib cimport malloc, free

cdef inline bint all_ascending(int *indices, int length):
    cdef int i
    for i in range(length - 1):
        if indices[i] > indices[i + 1]:
            return False
    return True

cdef inline bint all_descending(int *indices, int length):
    cdef int i
    for i in range(length - 1):
        if indices[i] < indices[i + 1]:
            return False
    return True

def ascending_indices(tuple shape):
    cdef int i, length, *c_indices
    cdef list all_indices, filtered_indices, sorted_indices

    length = len(shape)
    all_indices = list(product(*[range(s) for s in shape]))
    filtered_indices = []

    for indices in all_indices:
        c_indices = <int *> malloc(length * sizeof(int))
        if c_indices is NULL:
            raise MemoryError()

        for i in range(length):
            c_indices[i] = indices[i]

        if all_ascending(c_indices, length):
            filtered_indices.append(indices)

        free(c_indices)

    sorted_indices = sorted(filtered_indices)
    return sorted_indices


def descending_indices(tuple shape):
    cdef int i, length, *c_indices
    cdef list all_indices, filtered_indices, sorted_indices

    length = len(shape)
    all_indices = list(product(*[range(s) for s in shape]))
    filtered_indices = []

    for indices in all_indices:
        c_indices = <int *> malloc(length * sizeof(int))
        if c_indices is NULL:
            raise MemoryError()

        for i in range(length):
            c_indices[i] = indices[i]

        if all_descending(c_indices, length):
            filtered_indices.append(indices)

        free(c_indices)

    sorted_indices = sorted(filtered_indices, reverse=True)
    return sorted_indices
