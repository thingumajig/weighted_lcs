# -*- coding: utf-8 -*
import numpy as np

class LCS:
    threshold = 0.4
    def __init__(self, x, y, compare = lambda x, y: 1. if x == y else 0.):
        self.compare =  compare

        self.m = len(x) + 1
        self.n = len(y) + 1
        self.x, self.y = x, y

        # max_size = max(len(x), len(y))

        self.matrix = np.zeros((self.m, self.n))


        for i in range(0, self.m):
            self.matrix[i, 0] = 0
        for j in range(0, self.n):
            self.matrix[0, j] = 0

        for i in range(1, self.m):
            for j in range(1, self.n):
                w = compare(x[i-1], y[j-1])
                if w > self.threshold:
                    self.matrix[i,j] = self.matrix[i-1, j-1] + w
                else:
                    self.matrix[i,j] = max(self.matrix[i, j-1], self.matrix[i-1, j])

        print(self.matrix)
        self.lcs_length = self.matrix[self.m-1,self.n-1]

    def backtrack(self, i, j):
        if i==0 or j==0:
            return ''
        w = self.compare(self.x[i-1], self.y[j-1])
        if w > self.threshold:
            return self.backtrack(i-1, j-1) + self.x[i-1]
        if self.matrix[i, j-1] > self.matrix[i-1, j]:
            return self.backtrack(i, j-1)

        return self.backtrack(i-1, j)


