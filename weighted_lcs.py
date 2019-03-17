# -*- coding: utf-8 -*
import numpy as np


class LCS:
    threshold = 0.4

    def __init__(self, x, y, compare=lambda x, y: 1. if x == y else 0.):
        self.compare = compare

        self.m = len(x) + 1
        self.n = len(y) + 1
        self.x, self.y = x, y

        # max_size = max(len(x), len(y))

        self.matrix = np.zeros((self.m, self.n))

        # for i in range(0, self.m):
        #     self.matrix[i, 0] = 0
        # for j in range(0, self.n):
        #     self.matrix[0, j] = 0

        for i in range(1, self.m):
            for j in range(1, self.n):
                w = compare(x[i - 1], y[j - 1])
                if w > self.threshold:
                    self.matrix[i, j] = self.matrix[i - 1, j - 1] + w
                else:
                    self.matrix[i, j] = max(self.matrix[i, j - 1], self.matrix[i - 1, j])

                # matrix[x, y] = min(
                #     matrix[x - 1, y] + 1,  # deletion
                #     matrix[x, y - 1] + 1,  # insertion
                #     matrix[x - 1, y - 1] + cost  # substitution
                # )

        print(self.matrix)
        self.lcs_length = self.matrix[self.m - 1, self.n - 1]
        print('lcs length:', self.lcs_length)


    def backtrack_list(self):
        return self.__backtrack_list(self.m - 1, self.n - 1)

    def __backtrack_list(self, i, j):
        if i == 0 or j == 0:
            return []
        w = self.compare(self.x[i - 1], self.y[j - 1])
        if w > self.threshold:
            bck = self.__backtrack_list(i - 1, j - 1)
            bck.append(self.x[i - 1])
            return bck
        if self.matrix[i, j - 1] > self.matrix[i - 1, j]:
            return self.__backtrack_list(i, j - 1)

        return self.__backtrack_list(i - 1, j)

        # improve(?):
        # a = np.argmax([self.matrix[i-1, j - 1], self.matrix[i, j - 1], self.matrix[i - 1, j]])
        # if a == 0:
        #     bck = self.backtrack_list_internal(i - 1, j - 1)
        #     bck.append(self.x[i - 1])
        #     return bck
        # if a == 1:
        #     return self.backtrack_list_internal(i, j - 1)
        #
        # return self.backtrack_list_internal(i - 1, j)

    def backtrack_indexes(self):
        return self.__backtrack_indexes(self.m - 1, self.n - 1)

    def __backtrack_indexes(self, i, j):
        if i == 0 or j == 0:
            return []
        w = self.compare(self.x[i - 1], self.y[j - 1])
        if w > self.threshold:
            bck = self.__backtrack_indexes(i - 1, j - 1)
            bck.append((i - 1, j - 1))
            return bck
        if self.matrix[i, j - 1] > self.matrix[i - 1, j]:
            return self.__backtrack_indexes(i, j - 1)

        return self.__backtrack_indexes(i - 1, j)

    def backtrack_full(self):
        indexes = self.backtrack_indexes()
        print(indexes)
        return self.get_full_info(indexes)

    def get_full_info(self, indexes):
        s1, s2 = self.get_spans(indexes)
        return s1, s2, self.lcs_length / max(s1[1] - s1[0], s2[1] - s2[0])

    @staticmethod
    def get_spans(indexes):
        (i1, j1) = indexes[0]
        (i2, j2) = indexes[-1]
        return (i1, i2+1), (j1, j2+1)

    @staticmethod
    def compile_arrays(x, y, indexes):
        (i1, j1) = indexes[0]
        (i2, j2) = indexes[-1]
        return x[i1:i2+1], y[j1:j2+1]

    @staticmethod
    def get_list_span(indexes, axis = 0):
        start = indexes[0]
        ends = indexes[-1]
        return start[axis], ends[axis]+1

