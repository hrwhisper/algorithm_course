# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def merge_sort(L, R, a):
    if L >= R - 1: return 0
    mid = (L + R) >> 1
    cnt_left = merge_sort(L, mid, a)
    cnt_right = merge_sort(mid, R, a)
    return cnt_left + cnt_right + merge(L, mid, R, a)


def _count_by_merge(i, n, j, m, a):
    cnt = 0
    while i < n and j < m:
        if a[i] <= 3 * a[j]:
            i += 1
        else:
            cnt += n - i
            j += 1
    return cnt


def merge(L, le, R, a):
    rs = le
    ls = L
    x = []

    cnt = _count_by_merge(ls, le, rs, R, a)

    while ls < le and rs < R:
        if a[ls] <= a[rs]:
            x.append(a[ls])
            ls += 1
        else:
            x.append(a[rs])
            rs += 1

    for ls in range(ls, le):
        x.append(a[ls])
    for rs in range(rs, R):
        x.append(a[rs])
    for i in range(L, R):
        a[i] = x[i - L]
    return cnt


def brute_inversions(a):
    n = len(a)
    _cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > 3 * a[j]:
                _cnt += 1
    return _cnt


def test(test_cnt=1010, array_num=500, L=1, R=125551):
    import random
    for i in range(test_cnt):
        A = [random.randint(L, R) for _ in range(random.randint(1, array_num))]
        B = A[:]
        t = brute_inversions(B)
        cnt = merge_sort(0, len(A), A)
        B.sort()
        if cnt != t:
            print(cnt, t)
            return
    print('ok')


if __name__ == '__main__':
    test()
    A = [38, 27, 43, 3, 9, 82, 10]
    B = A[:]
    print(brute_inversions(B))
    print(merge_sort(0, len(A), A))
