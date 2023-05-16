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

cdef inline bint monotonic(int *indices, int length, bint ascending, bint strict):
    cdef int i
    for i in range(length - 1):
        if ascending:
            if strict:
                if indices[i] >= indices[i + 1]:
                    return False
            else:
                if indices[i] > indices[i + 1]:
                    return False
        else:
            if strict:
                if indices[i] <= indices[i + 1]:
                    return False
            else:
                if indices[i] < indices[i + 1]:
                    return False
    return True

def generate_indices(tuple shape, bint ascending=True, bint first_indices_change_fastest=True, bint strict=False):
    cdef int i
    cdef int length
    cdef int *c_indices
    cdef list all_indices, filtered_indices

    length = len(shape)
    all_indices = list(product(*[range(s) for s in shape]))

    filtered_indices = []

    for indices in all_indices:
        c_indices = <int *> malloc(length * sizeof(int))
        if c_indices is NULL:
            raise MemoryError()

        for i in range(length):
            c_indices[i] = indices[i]

        if monotonic(c_indices, length, ascending, strict):
            filtered_indices.append(indices)

        free(c_indices)

    if not first_indices_change_fastest:
        filtered_indices = sorted(filtered_indices, key=lambda x: tuple(reversed(x)))

    return filtered_indices

# def generate_indices(tuple shape, bint ascending=True, bint first_indices_change_fastest=True, bint strict=False):
#     cdef int i
#     cdef int length
#     cdef int *c_indices
#     cdef list all_indices, filtered_indices

#     length = len(shape)
#     if first_indices_change_fastest:
#         all_indices = list(product(*[range(s) for s in shape]))
#     else:
#         all_indices = list(product(*[range(s) for s in reversed(shape)]))
#         all_indices = [tuple(reversed(idx)) for idx in all_indices]

#     filtered_indices = []

#     for indices in all_indices:
#         c_indices = <int *> malloc(length * sizeof(int))
#         if c_indices is NULL:
#             raise MemoryError()

#         for i in range(length):
#             c_indices[i] = indices[i]

#         if monotonic(c_indices, length, ascending, strict):
#             filtered_indices.append(indices)

#         free(c_indices)

#     return filtered_indices
# def generate_indices(tuple shape, bint ascending=True, bint first_indices_change_fastest=True, bint strict=False):
#     cdef int i
#     cdef int length
#     cdef int *c_indices
#     cdef list all_indices, filtered_indices

#     length = len(shape)
#     all_indices = list(product(*[range(s) for s in shape]))
#     filtered_indices = []

#     for indices in all_indices:
#         c_indices = <int *> malloc(length * sizeof(int))
#         if c_indices is NULL:
#             raise MemoryError()

#         for i in range(length):
#             c_indices[i] = indices[i]

#         if monotonic(c_indices, length, ascending, strict):
#             filtered_indices.append(indices)

#         free(c_indices)

#     if first_indices_change_fastest:
#         return filtered_indices
#     else:
#         return [tuple(reversed(idx)) for idx in filtered_indices]
    
    