# -*- coding: utf-8 -*-
# @Date    : 2016/11/17
# @Author  : hrwhisper

import numpy as np


class Simplex(object):
    def __init__(self, obj):
        self.mat = np.array([[0] + obj[:]])

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])

    def _find_theta(self, col):
        return np.array([self.mat[i][0] / self.mat[i][col] if self.mat[i][col] > 0
                         else 0x7fffffff for i in range(1, self.mat.shape[0])]).argmin() + 1

    def _gaussian_elimination(self, row, col):
        self.mat[row] /= self.mat[row][col]
        for i in range(len(self.mat)):
            if i != row:
                self.mat[i] = self.mat[i] - self.mat[i][col] * self.mat[row]

    def solve(self):
        n = len(self.mat) - 1  # to add slack varible
        temp = np.vstack([np.zeros((1, n)), np.eye(n)])  # add a diagonal array
        self.mat = np.hstack([self.mat, temp])
        # print(self.mat)
        while self.mat[0].min() < 0:
            col = self.mat[0].argmin()
            row = self._find_theta(col)
            self._gaussian_elimination(row, col)
            print(row, col, '\n', self.mat)
        return self.mat[0][0]


if __name__ == '__main__':
    """
         maximize z: 3*x1 + 2*x2;
            2*x1 + x2 <= 100;
            x1 + x2 <= 80;
    """
    s = Simplex([-3, -2])
    s.add_constraint([2, -1], 100)
    s.add_constraint([1, 1], 80)
    s.add_constraint([1, 0], 40)
    print(s.solve())
    print(s.mat)

    """
       max z = 2x + 3y + 2z
       st
       2x + y + z <= 4
       x + 2y + z <= 7
       z          <= 5
       x,y,z >= 0
    """
    t = Simplex([-2, -3, -2])
    t.add_constraint([2, 1, 1], 4)
    t.add_constraint([1, 2, 1], 7)
    t.add_constraint([0, 0, 1], 5)
    print(t.solve())
    print(t.mat)
