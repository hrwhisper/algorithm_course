# -*- coding: utf-8 -*-
# @Date    : 2016/10/7
# @Author  : hrwhisper


def largest_divisible_subset(nums):
    if len(nums) <= 1: return nums
    nums.sort()
    n = len(nums)
    dp = [1] * n
    update_from = [-1] * n
    max_len, max_index = 1, 0
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                update_from[i] = j

        if dp[i] > max_len:
            max_len = dp[i]
            max_index = i

    ans = []
    while max_index != -1:
        ans.append(nums[max_index])
        max_index = update_from[max_index]
    return ans


if __name__ == '__main__':
    # print(s.largestDivisibleSubset([1]))
    # print(s.largestDivisibleSubset([1, 2, 3]))
    # print(s.largestDivisibleSubset([1, 2, 4, 8]))
    print(largest_divisible_subset([3, 4, 6, 12, 18, 54]))
