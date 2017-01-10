# -*- coding: utf-8 -*-
# @Date    : 2016/11/17
# @Author  : hrwhisper
# more details : https://www.hrwhisper.me/introduction-to-simplex-algorithm/

import numpy as np


class Simplex(object):
    def __init__(self, obj, max_mode=False):  # default is solve min LP, if want to solve max lp,should * -1
        self.mat, self.max_mode = np.array([[0] + obj]) * (-1 if max_mode else 1), max_mode

    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])

    def _simplex(self, mat, B, m, n):
        while mat[0, 1:].min() < 0:
            col = np.where(mat[0, 1:] < 0)[0][0] + 1  # use Bland's method to avoid degeneracy.
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  # find the theta index
            if mat[row][col] <= 0: return None  # the theta is ∞, the problem is unbounded
            self._pivot(mat, B, row, col)
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n}

    def _pivot(self, mat, B, row, col):
        mat[row] /= mat[row][col]
        ids = np.arange(mat.shape[0]) != row
        mat[ids] -= mat[row] * mat[ids, col:col + 1]  # for each i!= row do: mat[i]= mat[i] - mat[row] * mat[i][col]
        B[row] = col

    def solve(self):
        m, n = self.mat.shape  # m - 1 is the number slack variables we should add
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]), list(range(n - 1, n + m - 1))  # add diagonal array
        mat = self.mat = np.hstack([self.mat, temp])  # combine them!
        if mat[1:, 0].min() < 0:  # is the initial basic solution feasible?
            row = mat[1:, 0].argmin() + 1  # find the index of min b
            temp, mat[0] = np.copy(mat[0]), 0  # set first row value to zero, and store the previous value
            mat = np.hstack([mat, np.array([1] + [-1] * (m - 1)).reshape((-1, 1))])
            self._pivot(mat, B, row, mat.shape[1] - 1)
            if self._simplex(mat, B, m, n)[0] != 0: return None  # the problem has no answer

            if mat.shape[1] - 1 in B:  # if the x0 in B, we should pivot it.
                self._pivot(mat, B, B.index(mat.shape[1] - 1), np.where(mat[0, 1:] != 0)[0][0] + 1)
            self.mat = np.vstack([temp, mat[1:, :-1]])  # recover the first line
            for i, x in enumerate(B[1:]):
                self.mat[0] -= self.mat[0, x] * self.mat[i + 1]
        return self._simplex(self.mat, B, m, n)


if __name__ == '__main__':
    print("""
         maximize z: 3*x1 + 2*x2;
            2*x1 + x2 <= 100;
            x1 + x2 <= 80;
            x1 <= 40
            answer :180
    """)
    s = Simplex([3, 2], max_mode=True)
    s.add_constraint([2, 1], 100)
    s.add_constraint([1, 1], 80)
    s.add_constraint([1, 0], 40)
    print(s.solve())
    print(s.mat)

    print("""
         min z: x1 + 2x2;
            x1 + x2 >= 2; =》 -x1 - x2 <= -2
            x1 + x2 <= 1;
            x1 , x2 >= 0
            answer : None
    """)
    s = Simplex([1, 2])
    s.add_constraint([-1, -1], -2)
    s.add_constraint([1, 1], 1)
    print(s.solve())
    print(s.mat)

    print("""
             min z: x1 + 2x2; # -x1 + 2x2  则答案-2 x1=2
                x1 + x2 >= 1;=》 -x1 - x2 <= -1
                x1 + x2 <= 2;
                x1 , x2 >= 0
                answer : 1   x1=1
    """)
    s = Simplex([1, 2])
    s.add_constraint([-1, -1], -1)
    s.add_constraint([1, 1], 2)
    print(s.solve())
    print(s.mat)

    print("""
             max z: 2x1 - x2;
                2x1 + x2 <= 2
                x1 - 5x2 <= -4
                x1 , x2 >= 0
                answer : 0.181818181818182   x1=0.545454545454545 x2 = 0.909090909090909
    """)
    s = Simplex([2, -1], max_mode=True)
    s.add_constraint([2, 1], 2)
    s.add_constraint([1, -5], -4)
    print(s.solve())
    print(s.mat)

    print("""
        max 2x1 + 3x2 + 2x3
        st
        2x1 + x2 + x3 <= 4
        x1 + 2x2 + x3 <= 7
        x3          <= 5
        x1,x2,x3>= 0
        answer :11 x1 = 0 x2 = 3 x3 = 1
    """)
    t = Simplex([2, 3, 2], max_mode=True)
    t.add_constraint([2, 1, 1], 4)
    t.add_constraint([1, 2, 1], 7)
    t.add_constraint([0, 0, 1], 5)
    print(t.solve())
    print(t.mat)

    print("""
       max z = 2x1 - x2
       st
        2x1 - x2 <= 2
        x1 - 5x2 <= -4
        x1 ,x2 >= 0
       answer :2 x1 = 1.5555556 x2 = 1.1111
    """)
    t = Simplex([2, -1], max_mode=True)
    t.add_constraint([2, -1], 2)
    t.add_constraint([1, -5], -4)
    print(t.solve())
    print(t.mat)

    print("""
       max x1 + 14x2 +6x3
       st
        x1 + x2 + x3 <=4
        x1 <= 2
        x3 <= 3
        3x2 + x3 >= 6
        x1 ,x2 ,x3 >= 0
       answer :56  x2=4
    """)
    t = Simplex([1, 14, 6], max_mode=True)
    t.add_constraint([1, 1, 1], 4)
    t.add_constraint([1, 0, 0], 2)
    t.add_constraint([0, 0, 1], 3)
    t.add_constraint([0, -3, -1], -6)
    print(t.solve())
    print(t.mat)

    print("""
               minimize -x1 - 14x2 - 6x3
               st
                x1 + x2 + x3 <=4
                x1 <= 2
                x3 <= 3
                3x2 + x3 <= 6
                x1 ,x2 ,x3 >= 0
               answer :-32 x2=1 x3 = 3
            """)
    t = Simplex([-1, -14, -6])
    t.add_constraint([1, 1, 1], 4)
    t.add_constraint([1, 0, 0], 2)
    t.add_constraint([0, 0, 1], 3)
    t.add_constraint([0, 3, 1], 6)
    print(t.solve())
    print(t.mat)

    print("""
          maximize x1 + 14x2 + 6x3
          st
           x1 + x2 + x3 <=4
           x1 <= 2
           x3 <= 3
           3x2 + x3 <= 6
           x1 ,x2 ,x3 >= 0
          answer :32 x2 x2=1 x3=3
    """)
    t = Simplex([1, 14, 6], max_mode=True)
    t.add_constraint([1, 1, 1], 4)
    t.add_constraint([1, 0, 0], 2)
    t.add_constraint([0, 0, 1], 3)
    t.add_constraint([0, 3, 1], 6)
    print(t.solve())
    print(t.mat)

    print("""
              maximize x3
              st
               x1 <= 2
               x2 <= 2
               x3 <= 2
               x1 ,x2 ,x3 >= 0
              answer :2 x2 x2=1 x3=3
        """)
    t = Simplex([0, 0, 1], max_mode=True)
    t.add_constraint([1, 0, 0], 2)
    t.add_constraint([0, 1, 0], 2)
    t.add_constraint([0, 0, 1], 2)
    print(t.solve())
    print(t.mat)

    print("""
                  minimize 5x1 + 35x2 + 20x3
                  st
                   x1 - x2 - x3 <= -2
                   -x1 - 3x2   <=-3
                   x1 ,x2 ,x3 >= 0
                  answer :2 x2 x2=1 x3=3
            """)
    t = Simplex([5, 35, 20])
    t.add_constraint([1, -1, -1], -2)
    t.add_constraint([-1, -3, 0], -3)
    print(t.solve())
    print(t.mat)
