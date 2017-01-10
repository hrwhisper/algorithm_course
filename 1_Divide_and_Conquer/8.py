# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def merge_sort(L, R, a):
    if L >= R - 1: return 0
    mid = (L + R) >> 1
    cnt = merge_sort(L, mid, a)
    cnt += merge_sort(mid, R, a)
    return cnt + merge(L, mid, R, a)


def merge(L, le, R, a):
    cnt = 0
    rs = le
    ls = L
    x = []
    while ls < le and rs < R:
        if a[ls] <= a[rs]:
            x.append(a[ls])
            ls += 1
        else:
            x.append(a[rs])
            rs += 1
            cnt += le - ls

    for i in range(ls, le):
        x.append(a[i])
    for i in range(rs, R):
        x.append(a[i])
    for i in range(L, R):
        a[i] = x[i - L]
    return cnt


def quick_sort(L, R, a):
    if L >= R: return 0
    i, cnt = partition(L, R, a)
    cnt += quick_sort(L, i - 1, a)
    cnt += quick_sort(i + 1, R, a)
    return cnt


def partition(L, R, a):
    cnt = 0
    t = []
    base = a[L]
    i = L + 1
    while i <= R:
        if a[i] < base:
            cnt += i - L - len(t)
            t.append(a[i])
        i += 1
    j = len(t) + L
    t.append(base)  # or delete this line, and let i = L

    for i in range(L + 1, R + 1):
        if base <= a[i]:
            t.append(a[i])

    for i in range(L, R + 1):
        a[i] = t[i - L]

    return j, cnt


def brute_inversions(a):
    n = len(a)
    _cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                _cnt += 1
    return _cnt


def test(test_cnt=1000, array_num=1000, L=1, R=100000):
    import random
    for i in range(test_cnt):
        A = [random.randint(L, R) for _ in range(random.randint(1, array_num))]
        # t = brute_inversions(A[:])
        cnt_merge_sort = merge_sort(0, len(A), A[:])
        cnt_quick_sort = quick_sort(0, len(A) - 1, A[:])
        if cnt_quick_sort != cnt_merge_sort:  # cnt_merge_sort != t or:
            print(A)
            print(cnt_merge_sort, cnt_quick_sort)
            return
    print('ok')


if __name__ == '__main__':
    from datetime import datetime

    with open('./Q8.txt') as f:
        a = list(map(int, f.read().split()))
    start = datetime.now()
    print(merge_sort(0, len(a), a[:]))
    print('complete merge sort in {}s'.format((datetime.now() - start).total_seconds()))
    start = datetime.now()
    print(quick_sort(0, len(a) - 1, a))
    print('complete quick sort in {}s'.format((datetime.now() - start).total_seconds()))

    test()
