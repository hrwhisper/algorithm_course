# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def count_small_than_mid(rocks, mid, n):
    start = cnt = 0
    for i in range(1, n):
        if rocks[i] - rocks[start] <= mid:
            cnt += 1
        else:
            start = i
    return cnt


def binary_search(left, right, rocks, M, N):
    while left < right:
        mid = (left + right) >> 1
        if count_small_than_mid(rocks, mid, N) <= M:
            left = mid + 1
        else:
            right = mid
    return left


def solve_largest_minimum_spacing(L, M, N, rocks):
    rocks = [0] + rocks + [L]
    N += 2
    rocks.sort()
    left = min(rocks[i] - rocks[i - 1] for i in range(1, N))
    return binary_search(left, L + 1, rocks, M, N)


if __name__ == '__main__':
    L, N, M = 25, 5, 2
    rocks = [2, 14, 11, 21, 17]
    print(solve_largest_minimum_spacing(L, M, N, rocks))
