# -*- coding: utf-8 -*-
# @Date    : 2016/10/7
# @Author  : hrwhisper


def rob_circle(nums):
    if not nums: return 0
    n = len(nums)
    if n <= 2: return max(nums)
    dp1 = [0] * n
    dp2 = [0] * n
    dp1[0] = dp1[1] = nums[0]
    dp2[1] = nums[1]
    for i in range(2, n):
        dp1[i] = max(dp1[i - 1], dp1[i - 2] + nums[i])
        dp2[i] = max(dp2[i - 1], dp2[i - 2] + nums[i])
    return max(dp1[n - 2], dp2[n - 1])


def rob_no_circle(nums):
    if not nums: return 0
    n = len(nums)
    if n <= 2: return max(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[n - 1]


def rob_circle2(self, nums):
    if not nums: return 0
    if len(nums) <= 2: return max(nums)
    return max(rob_no_circle(nums[:-1]), rob_no_circle(nums[1:]))


if __name__ == '__main__':
    s = Solution()
    print(s.rob([1, 2, 3]))
    print(rob_circle([1,2,3]))
