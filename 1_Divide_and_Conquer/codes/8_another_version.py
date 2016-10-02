# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def merge_sort(a):
    if len(a) == 1: return a, 0
    mid = len(a) >> 1
    left, left_cnt = merge_sort(a[:mid])
    right, right_cnt = merge_sort(a[mid:])
    a, cnt = merge(left, right)
    return a, left_cnt + right_cnt + cnt


def merge(left, right):
    cnt = 0
    x = []
    left = left[::-1]
    right = right[::-1]
    while left or right:
        if left and right and left[-1] <= right[-1]:
            x.append(left.pop())
        elif right:
            x.append(right.pop())
            cnt += len(left)
        else:
            x.append(left.pop())
    return x, cnt


def find_inversions_by_merge_sort(a):
    return merge_sort(a)[1]


def quick_sort(a):
    if not a: return 0
    if len(a) == 1: return 0
    i, cnt = partition(a)
    left_cnt = quick_sort(a[:i])
    right_cnt = quick_sort(a[i + 1:])
    return left_cnt + right_cnt + cnt


def partition(a):
    if len(a) == 0: return 0, 0
    cnt = 0
    t = []
    base = a[0]
    for i in range(len(a)):
        if a[i] < base:
            cnt += i - len(t)
            t.append(a[i])
    j = len(t)
    for i in range(len(a)):
        if base <= a[i]:
            t.append(a[i])

    for i in range(len(a)):
        a[i] = t[i]
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
        cnt_merge_sort = find_inversions_by_merge_sort(A[:])
        cnt_quick_sort = quick_sort(A[:])
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
    print(find_inversions_by_merge_sort(a[:]))
    print('complete merge sort in {}s'.format((datetime.now() - start).total_seconds()))
    start = datetime.now()
    print(quick_sort(a[:]))
    print('complete quick sort in {}s'.format((datetime.now() - start).total_seconds()))

    test()
