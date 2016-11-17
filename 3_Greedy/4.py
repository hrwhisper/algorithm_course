# -*- coding: utf-8 -*-
# @Date    : 2016/10/17
# @Author  : hrwhisper


def max_payoff(A, B):
    if not A or not B or len(A) != len(B): return 0
    A.sort()
    B.sort()
    ans = 1
    for i in range(len(A)):
        ans *= A[i] ** B[i]
    return ans


if __name__ == '__main__':
    a = [1, 3, 4, 2, 1, 6, 2]
    b = [2, 3, 4, 5, 6, 7, 2]
    print(max_payoff(a,b))
