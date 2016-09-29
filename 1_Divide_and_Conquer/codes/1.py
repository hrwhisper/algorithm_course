# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def binary_search(A, la, ra, B, lb, rb, k):
    m, n = ra - la, rb - lb
    if n == 0: return A[la + k - 1]
    if k == 1: return min(A[la], B[lb])

    b_m = k >> 1
    a_m = k - b_m
    if A[la + a_m - 1] < B[lb + b_m - 1]:
        return binary_search(A, la + a_m, ra, B, lb, lb + b_m, k - a_m)
    else:  # A[la + a_m - 1] > B[lb + b_m - 1]
        return binary_search(A, la, la + a_m, B, lb + b_m, rb, k - b_m)


def find_median(A, B):
    return binary_search(A, 0, len(A), B, 0, len(A), ((len(A) << 1) + 1) >> 1)


def test(test_cnt=100000, array_num=10):
    import random
    L = 1
    R = array_num ** 2
    for i in range(test_cnt):
        A = []
        B = []
        vis = set()
        for _ in range(random.randint(1, array_num)):
            t = random.randint(L, R)
            while t in vis:
                t = random.randint(L, R)
            vis.add(t)
            A.append(t)
        for _ in range(len(A)):
            t = random.randint(L, R)
            while t in vis:
                t = random.randint(L, R)
            vis.add(t)
            B.append(t)

        A.sort()
        B.sort()
        C = A + B
        n = len(A) << 1
        C.sort()
        median = C[(n - 1) >> 1]
        median2 = find_median(A[:], B[:])
        if median2 != median:
            print(A)
            print(B)
            print(median, median2)
            return
    print('ok')


if __name__ == '__main__':
    test()
    # s = Solution()
    # a = [1,2]
    # b = [3,4]
    # print(find_median(a, b), s.findMedianSortedArrays(a, b))
