# -*- coding: utf-8 -*-
# @Date    : 2016/10/11
# @Author  : hrwhisper
import collections


def can_cross2(stones):
    n = len(stones)
    val2id = {stone: i for i, stone in enumerate(stones)}
    dp = collections.defaultdict(lambda: collections.defaultdict(int))
    dp[1][0] = True
    for j in range(1, n):
        for i in dp[j]:  # the same as dp[j].keys()
            step = stones[j] - stones[i]
            for k in [step + 1, step, step - 1]:
                _next = stones[j] + k
                if _next in val2id:
                    _id = val2id[_next]
                    if _id == n - 1:
                        return True
                    if _id != j:
                        dp[_id][j] = True
    return False


def can_cross(stones):
    dp = {stone: {} for stone in stones}
    dp[0][0] = 0
    for stone in stones:
        for step in dp[stone].values():
            for k in [step + 1, step, step - 1]:
                if k > 0 and stone + k in dp:
                    dp[stone + k][stone] = k
    return len(dp[stones[-1]].keys()) > 0


if __name__ == '__main__':
    print(can_cross([0, 2]))
    print(can_cross([0, 1, 3, 5, 6, 8, 12, 17]))
    print(can_cross([0, 1, 2, 3, 4, 8, 9, 11]))