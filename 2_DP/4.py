# -*- coding: utf-8 -*-
# @Date    : 2016/10/7
# @Author  : hrwhisper


def decoding_ways(s):
    """
    :type s: str
    :rtype: int
    """
    if not s: return 0
    n = len(s)
    dp = [0] * n
    dp[0] = 1 if s[0] != '0' else 0
    for i in range(1, n):
        if 10 <= int(s[i - 1:i + 1]) <= 26:
            dp[i] += dp[i - 2] if i >= 2 else 1
        if s[i] != '0':
            dp[i] += dp[i - 1]
    return dp[n - 1]


if __name__ == '__main__':
    print(decoding_ways('0'))  # 0
    print(decoding_ways('10'))  # 1
    print(decoding_ways('1211'))  # 5
    print(decoding_ways('12011'))  # 2
    print(decoding_ways('12345'))  # 3
    print(decoding_ways('01'))  # 0
