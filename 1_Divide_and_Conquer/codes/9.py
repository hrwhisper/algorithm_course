# -*- coding: utf-8 -*-
# @Date    : 2016/9/24
# @Author  : hrwhisper
from functools import cmp_to_key
import math


def euclidean_dis_pow(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def solve_closest_pair_n_logn2(points):
    def closest_pair(L, R, points):
        if L == R: return 0x7fffffff, points[L], points[R]  # return int max
        if R - L == 1: return euclidean_dis_pow(points[L], points[R]), points[L], points[R]
        mid = (L + R) >> 1
        d, p1, p2 = closest_pair(L, mid, points)
        d2, p3, p4 = closest_pair(mid + 1, R, points)
        if d > d2:
            d, p1, p2 = d2, p3, p4

        min_x = points[mid][0] - d
        max_x = points[mid][0] + d

        suspect = [points[i] for i in range(L, R + 1) if min_x <= points[i][0] <= max_x]

        suspect.sort(key=lambda x: x[1])
        n = len(suspect)
        for i in range(n):
            for j in range(i + 1, n):
                if suspect[j][1] - suspect[i][1] > d: break
                t = euclidean_dis_pow(suspect[i], suspect[j])
                if t < d:
                    d = t
                    p1, p2 = suspect[i], suspect[j]
        return d, p1, p2

    points.sort(key=cmp_to_key(lambda x, y: x[0] - y[0] if x[0] != y[0] else x[1] - y[1]))
    return closest_pair(0, len(points) - 1, points)


def solve_closest_pair_n_logn(points):
    def merge(ls, le, re, a):
        start = ls
        rs = le + 1
        b = []
        while ls <= le and rs <= re:
            if a[ls][1] < a[rs][1]:
                b.append(a[ls])
                ls += 1
            else:
                b.append(a[rs])
                rs += 1

        for i in range(ls, le + 1):
            b.append(a[i])

        for i in range(rs, re + 1):
            b.append(a[i])

        for i in range(start, re + 1):
            a[i] = b[i - start]

    def closest_pair(L, R, points, y_sorted):
        if L == R: return 0x7fffffff, points[L], points[R]  # return int max
        if R - L == 1:
            if y_sorted[L][1] > y_sorted[R][1]:
                y_sorted[L], y_sorted[R] = y_sorted[R], y_sorted[L]
            return euclidean_dis_pow(points[L], points[R]), points[L], points[R]
        mid = (L + R) >> 1
        d, p1, p2 = closest_pair(L, mid, points, y_sorted)
        d2, p3, p4 = closest_pair(mid + 1, R, points, y_sorted)
        merge(L, mid, R, y_sorted)
        if d > d2:
            d, p1, p2 = d2, p3, p4

        min_x = points[mid][0] - d
        max_x = points[mid][0] + d

        suspect = [y_sorted[i] for i in range(L, R + 1) if min_x <= y_sorted[i][0] <= max_x]
        n = len(suspect)
        for i in range(n):
            for j in range(i + 1, n):
                if suspect[j][1] - suspect[i][1] > d: break
                t = euclidean_dis_pow(suspect[i], suspect[j])
                if t < d:
                    d = t
                    p1, p2 = suspect[i], suspect[j]
        return d, p1, p2

    points.sort(key=cmp_to_key(lambda x, y: x[0] - y[0] if x[0] != y[0] else x[1] - y[1]))
    y_sorted = points[:]
    return closest_pair(0, len(points) - 1, points, y_sorted)


def brute_closest_pair(points):
    d = 0x7ffffffff
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = min(d, euclidean_dis_pow(points[i], points[j]))
    return d


def test(test_cnt=3000, array_num=1500, L=1, R=15100):
    import random
    for i in range(test_cnt):
        points = [(random.randint(L, R), random.randint(L, R)) for _ in range(random.randint(2, array_num))]
        d1 = brute_closest_pair(points[:])
        d2, p1, p2 = solve_closest_pair_n_logn2(points[:])
        d3, p3, p4 = solve_closest_pair_n_logn(points[:])

        if d1 != d2 or d1 != d3:
            print(d1, d2, d3)
            return
    print('ok')


if __name__ == '__main__':
    test()
    # a = [[1, 2], [1, 1], [0, 1]]
    # print(brute_closest_pair(a))
    #
    # a = [(127860, 86521), (30732, 71007), (4991, 11841), (52612, 123297)]
    # a = [(3280, 6524), (974, 2708), (9442, 13129), (6876, 5971), (14190, 8614), (14278, 13317), (7126, 7101)]
    # print(solve_closest_pair_n_logn(a[:]))
    # print(solve_closest_pair_n_logn2(a[:]))
    # print(brute_closest_pair(a))
