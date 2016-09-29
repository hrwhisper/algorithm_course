# -*- coding: utf-8 -*-
# @Date    : 2016/9/27
# @Author  : hrwhisper
from math import ceil, log


def matrix_brute_mul(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_add_or_sub(A, B, add=True):
    n = len(A)
    return [[A[i][j] + B[i][j] if add else A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def _strassen_mul(A, B):
    n = len(A)
    if n == 1: return [[A[0][0] * B[0][0]]]
    if n == 2: matrix_brute_mul(A, B)
    half_n = n >> 1
    A11, A12, A21, A22 = [], [], [], []
    B11, B12, B21, B22 = [], [], [], []
    for i in range(half_n):
        A11.append(A[i][:half_n][:])
        A12.append(A[i][half_n:][:])
        B11.append(B[i][:half_n][:])
        B12.append(B[i][half_n:][:])
        A21.append(A[i + half_n][:half_n][:])
        A22.append(A[i + half_n][half_n:][:])
        B21.append(B[i + half_n][:half_n][:])
        B22.append(B[i + half_n][half_n:][:])

    P1 = _strassen_mul(A11, matrix_add_or_sub(B12, B22, False))
    P2 = _strassen_mul(matrix_add_or_sub(A11, A12), B22)
    P3 = _strassen_mul(matrix_add_or_sub(A21, A22), B11)
    P4 = _strassen_mul(A22, matrix_add_or_sub(B21, B11, False))
    P5 = _strassen_mul(matrix_add_or_sub(A11, A22), matrix_add_or_sub(B11, B22))
    P6 = _strassen_mul(matrix_add_or_sub(A12, A22, False), matrix_add_or_sub(B21, B22))
    P7 = _strassen_mul(matrix_add_or_sub(A11, A21, False), matrix_add_or_sub(B11, B12))

    C11 = matrix_add_or_sub(matrix_add_or_sub(matrix_add_or_sub(P4, P5), P6), P2, False)
    C12 = matrix_add_or_sub(P1, P2)
    C21 = matrix_add_or_sub(P3, P4)
    C22 = matrix_add_or_sub(matrix_add_or_sub(matrix_add_or_sub(P1, P5), P3, False), P7, False)

    C = [[] for _ in range(n)]
    for i in range(half_n):
        C[i].extend(C11[i])
        C[i].extend(C12[i])
        C[i + half_n].extend(C21[i])
        C[i + half_n].extend(C22[i])
    return C


def strassen_matrix_mul(A, B):
    before_n = len(A)

    n = 2 ** ceil(log(before_n, 2))
    for i in range(before_n):
        A[i].extend([0] * (n - before_n))
        B[i].extend([0] * (n - before_n))
    for i in range(before_n, n):
        A.append([0] * n)
        B.append([0] * n)

    C = _strassen_mul(A, B)[:before_n]
    return [row[:before_n] for row in C]


def test(test_cnt=1000, max_n=100, L=1, R=1000):
    print('start test cnt={} , max_n={}'.format(test_cnt, max_n))
    import random
    import numpy as np
    for _ in range(test_cnt):
        n = random.randint(1, max_n)
        A = []
        B = []
        for i in range(n):
            A.append([random.randint(L, R) for _ in range(n)])
            B.append([random.randint(L, R) for _ in range(n)])
        C2 = (np.matrix(A) * np.matrix(B)).tolist()
        C = strassen_matrix_mul(A[:], B[:])
        if C != C2:
            print('Wrong answer')
            print(A)
            print(B)
            print(C)
            print(C2)
            return
    print('ok')


if __name__ == '__main__':
    # from datetime import datetime
    # import random
    #
    # n = 1024
    # A, B = [], []
    # for i in range(n):
    #     A.append([random.randint(0, n ** 2) for _ in range(n)])
    #     B.append([random.randint(0, n ** 2) for _ in range(n)])
    #
    # start = datetime.now()
    # matrix_brute_mul(A, B)
    # print('complete grade-school method in {}s'.format((datetime.now() - start).total_seconds()))
    #
    # start = datetime.now()
    # strassen_matrix_mul(A[:], B[:])
    # print('complete Strassen method in {}s'.format((datetime.now() - start).total_seconds()))
    #
    # import numpy as np
    # start = datetime.now()
    # np.matrix(A) * np.matrix(B).tolist()
    # print('complete numpy matrix mul in {}s'.format((datetime.now() - start).total_seconds()))
    test()
