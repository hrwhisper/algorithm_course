# -*- coding: utf-8 -*-
# @Date    : 2016/10/11
# @Author  : hrwhisper


def partition(s):
    def helper(i, j):
        while j >= 0 and i < n:
            if s[i] != s[j]:
                break
            dp[i] = min(dp[i], dp[j - 1] + 1 if j > 0 else 0)
            i, j = i + 1, j - 1

    n = len(s)
    dp = [0] + [0x7fffffff] * n
    for k in range(1, n):
        helper(k, k)  # odd case
        helper(k, k - 1)  # even case

    return dp[n - 1]


if __name__ == '__main__':
    print(partition('aab'))
    print(partition('aaba'))
    print(partition('aba'))
    print(partition('a'))
