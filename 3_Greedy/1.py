# -*- coding: utf-8 -*-
# @Date    : 2016/10/18
# @Author  : hrwhisper


def can_be_a_graph(degrees):
    d_sum, n = sum(degrees), len(degrees)
    if d_sum & 1 or n * (n - 1) < d_sum or max(degrees) > n - 1: return False
    for n in range(n, -1, -1):
        degrees.sort(reverse=True)  # 可以每一次算完类似合并排序合并过程使得总复杂度为O(n^2)
        for i in range(1, n):
            if degrees[0] <= 0: break
            degrees[i] -= 1
            if degrees[i] < 0: return False
            degrees[0] -= 1
        if degrees[0] != 0: return False
    return True


def merge(a, ls, le, re):
    t = []
    _ls = ls
    rs = le
    while ls < le and rs < re:
        if a[ls] >= a[rs]:
            t.append(a[ls])
            ls += 1
        else:
            t.append(a[rs])
            rs += 1
    for i in range(ls, le):
        t.append(a[i])
    for i in range(rs, re):
        t.append(a[i])

    for i in range(_ls, re):
        a[i] = t[i - _ls]


def can_be_a_graph2(degrees):
    d_sum, n = sum(degrees), len(degrees)
    if d_sum & 1 or n * (n - 1) < d_sum or max(degrees) > n - 1: return False
    degrees.sort(reverse=True)
    while degrees:
        k = degrees[0]
        for i in range(1, n):
            if degrees[0] <= 0: break
            degrees[i] -= 1
            if degrees[i] < 0: return False
            degrees[0] -= 1
        if degrees[0] != 0: return False
        n -= 1
        degrees.pop(0)
        merge(degrees, 0, k, n)
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
        print(can_be_a_graph(t[:]) , can_be_a_graph2(t[:]))
