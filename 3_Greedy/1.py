# -*- coding: utf-8 -*-
# @Date    : 2016/10/18
# @Author  : hrwhisper


def can_be_a_graph(degrees):
    d_sum, n = sum(degrees), len(degrees)
    if d_sum & 1 or n * (n - 1) < d_sum or max(degrees) > n - 1: return False
    for n in range(n, -1, -1):
        degrees.sort(reverse=True)
        for i in range(1, n):
            if degrees[0] <= 0: break
            degrees[i] -= 1
            if degrees[i] < 0: return False
            degrees[0] -= 1
        if degrees[0] != 0: return False
    return True


if __name__ == '__main__':
    test_case = [
        [1, 1, 2, 2, 4],  # True
        [1, 1, 2, 2, 2],  # True
        [1, 2, 2, 3, 4],  # True
        [1, 2, 2, 2, 4],  # False
        [1, 2, 3, 4, 4],  # False
        [0],  # True
        [1],  # False
    ]

    for t in test_case:
        print(can_be_a_graph(t[:]))
