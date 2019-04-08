# -*- coding: utf-8 -*
import numpy as np
from typing import Dict, List, Tuple, Sequence

# from functools import lru_cache
from cachetools import cachedmethod, LRUCache
import operator

import logging
logger = logging.getLogger(__name__)

class Weightable:
    def get_weight(self):
        pass


class SimpleWeight(Weightable):
    def __init__(self, w) -> None:
        self.w = w

    def get_weight(self):
        return self.w


class LCS:

    def __init__(self, x, y, threshold=0.4, compare=lambda x, y: SimpleWeight(1.) if x == y else SimpleWeight(0.), orig_x = None, orig_y = None):
        self.compare = compare

        self.m = len(x) + 1
        self.n = len(y) + 1
        self.x, self.y = x, y

        self.orig_x, self.orig_y = orig_x, orig_y

        self.threshold = threshold

        self.cache = LRUCache(maxsize=len(x) * len(y))
        # max_size = max(len(x), len(y))

        self.matrix = np.zeros((self.m, self.n))

        # for i in range(0, self.m):
        #     self.matrix[i, 0] = 0
        # for j in range(0, self.n):
        #     self.matrix[0, j] = 0

        for i in range(1, self.m):
            for j in range(1, self.n):
                wi = self.__compare(i - 1, j - 1)
                w = wi.get_weight()
                if w > self.threshold:
                    self.matrix[i, j] = self.matrix[i - 1, j - 1] + w
                else:
                    self.matrix[i, j] = max(self.matrix[i, j - 1], self.matrix[i - 1, j])

                # matrix[x, y] = min(
                #     matrix[x - 1, y] + 1,  # deletion
                #     matrix[x, y - 1] + 1,  # insertion
                #     matrix[x - 1, y - 1] + cost  # substitution
                # )

        # print(self.matrix)
        self.print_matrix()
        self.lcs_length = self.matrix[self.m - 1, self.n - 1]
        logger.info(f"lcs length: {self.lcs_length}")

    def print_matrix(self):
        if self.orig_x is None and type(self.x[0])!=str:
            return
        
        mm = np.array(self.matrix, dtype=np.dtype(object))

        x = self.x if self.orig_x is None else self.orig_x
        y = self.y if self.orig_y is None else self.orig_y

        for i in range(0, len(self.x)):
            mm[i+1, 0] = self.x[i]

        for i in range(0, len(self.y)):
            mm[0, i+1] = self.y[i]

        logging.info(mm)


    # @lru_cache(maxsize=1024)
    @cachedmethod(operator.attrgetter("cache"))
    def __compare(self, i, j):
        return self.compare(self.x[i], self.y[j])

    def backtrack_list(self):
        return self.__backtrack_list(self.m - 1, self.n - 1)

    def __backtrack_list(self, i, j):
        if i == 0 or j == 0:
            return []
        wi = self.__compare(i - 1, j - 1)
        w = wi.get_weight()
        if w > self.threshold:
            bck = self.__backtrack_list(i - 1, j - 1)
            bck.append(self.x[i - 1])
            return bck
        if self.matrix[i, j - 1] > self.matrix[i - 1, j]:
            return self.__backtrack_list(i, j - 1)

        return self.__backtrack_list(i - 1, j)

    # def backtrack_indexes(self):
    #     return self.__backtrack_indexes(self.m - 1, self.n - 1)

    def backtrack_indexes(self, i=None, j=None):
        if i is None:
            i = self.m - 1
        if j is None:
            j = self.n - 1

        if i == 0 or j == 0:
            return []
        wi = self.__compare(i - 1, j - 1)
        w = wi.get_weight()
        if w > self.threshold:
            bck = self.backtrack_indexes(i - 1, j - 1)
            bck.append((i - 1, j - 1, wi))
            return bck
        if self.matrix[i, j - 1] > self.matrix[i - 1, j]:
            return self.backtrack_indexes(i, j - 1)
        else:
            return self.backtrack_indexes(i - 1, j)

    def backtrack_full(self):
        indexes = self.backtrack_indexes()
        logger.info(indexes)
        return self.get_full_info(indexes)

    def __backtrack_all(self, i, j):
        if i == 0 or j == 0:
            return [[]]
        wi = self.__compare(i - 1, j - 1)
        w = wi.get_weight()
        if w > self.threshold:
            Zs = self.__backtrack_all(i - 1, j - 1)
            for z in Zs:
                z.append((i - 1, j - 1))
            return Zs
        R = []
        if self.matrix[i, j - 1] >= self.matrix[i - 1, j]:
            R.extend(self.__backtrack_all(i, j - 1))

        if self.matrix[i - 1, j] >= self.matrix[i, j - 1]:
            R.extend(self.__backtrack_all(i - 1, j))

        return R

    def backtrack_all_sequences(self):
        return self.__backtrack_all(self.m - 1, self.n - 1)

    def get_full_info(self, indexes):
        s1, s2 = get_spans(indexes)
        # return s1, s2, self.lcs_length / max(s1[1] - s1[0], s2[1] - s2[0])
        return s1, s2, self.lcs_length


def get_spans(indexes: list):
    (i1, j1, w1) = indexes[0]
    (i2, j2, w2) = indexes[-1]
    return (i1, i2 + 1), (j1, j2 + 1)


def gather_array(x, indexes, axis=0, delimiter=''):
    return delimiter.join([x[j[axis]] for j in indexes])


def compile_arrays(x, y, indexes):
    (i1, j1) = indexes[0]
    (i2, j2) = indexes[-1]
    return x[i1:i2 + 1], y[j1:j2 + 1]


def get_list_span(indexes, axis=0):
    start = indexes[0]
    ends = indexes[-1]
    return start[axis], ends[axis] + 1


