# -*- coding: utf-8 -*-
# @Date    : 2016/10/11
# @Author  : hrwhisper


def longest_increasing_subsequence(nums):
    if not nums: return 0
    n = len(nums)
    dp = [1] * n
    update_from = [-1] * n

    lis_len = 1
    index = 0
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if nums[j] < nums[i] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                update_from[i] = j
        if dp[i] > lis_len:
            lis_len, index = dp[i], i

    ans = []
    while index != -1:
        ans.append(nums[index])
        index = update_from[index]
    return lis_len, ans[::-1]


def binary_search(g, x, L, R):
    while L < R:
        mid = (L + R) >> 1
        if g[mid] < x:
            L = mid + 1
        else:
            R = mid
    return L


def longest_increasing_subsequence_nlogn(nums):
    if not nums: return 0
    n = len(nums)
    dp = [1] * n
    g = [0x7fffffff] * (n + 1)
    update_from = [-1] * (n + 1)
    indexs = [-1] * (n + 1)
    lis_len = 1
    index = 0
    for i in range(n):
        k = binary_search(g, nums[i], 1, n)
        g[k] = nums[i]
        dp[i] = k
        indexs[k] = i
        update_from[i] = indexs[k - 1]
        if dp[i] > lis_len:
            lis_len, index = dp[i], i

    ans = []
    while index != -1:
        ans.append(nums[index])
        index = update_from[index]
    return lis_len, ans[::-1]


def test(test_cnt=1000, array_num=1000):
    import random
    L = 1
    R = array_num ** 2
    for i in range(test_cnt):
        A = []
        for _ in range(random.randint(1, array_num)):
            A.append(random.randint(L, R))
        try:
            t1 = longest_increasing_subsequence_nlogn(A[:])
            t2 = longest_increasing_subsequence(A[:])
            if t1 != t2:
                print(t1, t2)
                return
        except Exception as e:
            print(e)
            print(A)
            return
    print('ok')


if __name__ == '__main__':
    nums = [1, 3, 4, 2, 1, 6, 2]
    print(longest_increasing_subsequence(nums))
    print(longest_increasing_subsequence_nlogn(nums))
    test()
