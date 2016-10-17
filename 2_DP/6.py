# -*- coding: utf-8 -*-
# @Date    : 2016/10/7
# @Author  : hrwhisper


def max_profit(prices):
    if not prices or len(prices) < 2: return 0
    n = len(prices)
    min_price = prices[0]
    dp = [0] * n
    for i in range(1, n):
        dp[i] = max(dp[i - 1], prices[i] - min_price)
        min_price = min(prices[i], min_price)

    max_price = prices[n - 1]
    ans = dp[n - 1]
    max_profit = 0
    for i in range(n - 1, 0, -1):
        max_profit = max(max_profit, max_price - prices[i])
        max_price = max(max_price, prices[i])
        ans = max(ans, max_profit + dp[i - 1])
    return ans


if __name__ == '__main__':
    print(max_profit([3, 2, 6, 5, 0, 3]))  # 7
    print(max_profit([1, 2, 5, 2, 3, 10, 2, 15]))  # 22
    print(max_profit([7, 1, 5, 3, 6, 4]))  # 7
    print(max_profit([1, 2]))  # 1
    print(max_profit([1, 2, 3]))  # 2
    print(max_profit([1, 10, 1, 5]))  # 13
    print(max_profit([2, 1]))  # 0
    print(max_profit([1]))
    print(max_profit([]))
