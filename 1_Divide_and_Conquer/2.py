# -*- coding: utf-8 -*-
# @Date    : 2016/9/22
# @Author  : hrwhisper

def find_kth_element(L, R, A, k):
    if L == R: return A[L]
    i = partition(L, R, A)
    left_element = i - L + 1
    if left_element == k: return A[i]
    if left_element < k:
        return find_kth_element(i + 1, R, A, k - left_element)
    else:
        return find_kth_element(L, i - 1, A, k)


def partition(L, R, A):
    i = L + 1
    j = R
    base = A[L]
    while True:
        while i < j and A[i] > base: i += 1
        while j > L and A[j] < base: j -= 1
        if i >= j: break
        A[i], A[j] = A[j], A[i]  # swap

    A[L], A[j] = A[j], A[L]  # swap
    return j


def test(test_cnt=1000000, array_num=500, L=1, R=125111):
    import random
    for i in range(test_cnt):
        a = [random.randint(L, R) for _ in range(array_num)]
        _a = sorted(a[:], reverse=True)
        k = random.randint(1, array_num)
        if find_kth_element(0, len(a) - 1, a, k) != _a[k - 1]:
            print(a)
            print(k)
            print(find_kth_element(0, len(a) - 1, a, k), _a[k - 1])
            return
    print('ok')


if __name__ == '__main__':
    test()
    # a = [700, 597, 91, 541, 242, 451, 538, 351, 585, 700, 728, 711, 752, 777, 1194, 1240, 804, 948, 1201, 843]
    # _a = sorted(a[:], reverse=True)
    # k = 4
    # print(find_kth_element(0, len(a) - 1, a, k), _a[k - 1])
    # a = [3, 2, 1, 5, 6, 4]
    # _a = sorted(a[:], reverse=True)
    # k = 2
    # print(find_kth_element(0, len(a) - 1, a, k), _a[k - 1])
