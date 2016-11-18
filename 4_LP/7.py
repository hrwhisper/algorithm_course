# -*- coding: utf-8 -*-
# @Date    : 2016/11/17
# @Author  : hrwhisper

import numpy as np


class Simplex(object):
    def __init__(self, obj):
        self.mat = np.array([[0] + obj])

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])

    def solve(self):
        n = len(self.mat) - 1  # the number slack variables we should add
        temp = np.vstack([np.zeros((1, n)), np.eye(n)])  # add a diagonal array
        mat = self.mat = np.hstack([self.mat, temp])  # combine them!
        while mat[0].min() < 0:
            col = mat[0].argmin()
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # find the theta
            if mat[row][col] <= 0: return None  # no answer. it equals the theta is ∞
            mat[row] /= mat[row][col]
            # for each i!= row do : self.mat[i] = self.mat[i] - self.mat[row] * self.mat[i][col]
            mat[np.arange(mat.shape[0]) != row] -= mat[row] * mat[np.arange(mat.shape[0]) != row, col:col + 1]
        return mat[0][0]


if __name__ == '__main__':
    """
         maximize z: 3*x1 + 2*x2;
            2*x1 + x2 <= 100;
            x1 + x2 <= 80;
            answer :180
    """
    s = Simplex([-3, -2])
    s.add_constraint([2, 1], 100)
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
       answer :11
    """
    t = Simplex([-2, -3, -2])
    t.add_constraint([2, 1, 1], 4)
    t.add_constraint([1, 2, 1], 7)
    t.add_constraint([0, 0, 1], 5)
    print(t.solve())
    print(t.mat)