# -*- coding: utf-8 -*-
# @Date    : 2016/10/19
# @Author  : hrwhisper
from functools import cmp_to_key


def min_complete_time(p, f):
    n = len(p)
    t = list(zip(range(n), list(zip(p, f))))  # it will be like this: [_id,(pi,fi)]
    t.sort(key=cmp_to_key(lambda x, y: y[1][1] - x[1][1]))
    order = []
    min_time = timecost = 0
    for i in range(n):
        order.append(t[i][0])
        timecost += t[i][1][0]
        min_time = max(min_time, timecost + t[i][1][1])
    return min_time, order


def cal_time(order, p, f):
    cur_p_time = 0
    total_time = 0
    for i in order:
        cur_p_time += p[i]
        total_time = max(total_time, cur_p_time + f[i])
    return total_time


def brute_judge(p, f):
    order = list(range(len(p)))
    min_time = cal_time(order, p, f)
    min_order = order[:]
    while next_permutation(order):
        cur_time = cal_time(order, p, f)
        if cur_time < min_time:
            min_time = cur_time
            min_order = order[:]
    return min_time, min_order


def next_permutation(num):
    j, k = len(num) - 2, len(num) - 1
    while j >= 0:
        if num[j] < num[j + 1]: break
        j -= 1

    if j < 0:
        return False

    while k > j:
        if num[k] > num[j]: break
        k -= 1
    num[j], num[k] = num[k], num[j]
    num[:] = num[:j + 1] + num[:j:-1]
    return True


def min_complete_time2(p, f):
    n = len(p)
    if n == 0: return 0, []
    if n == 1: return p[0] + f[0], [0]
    pf = list(zip(p, f))  # it will be like this: [pi,fi]
    order = []
    vis = [False] * n
    for _ in range(n):
        cur_min_cost = 0x7fffffff
        min_id = -1
        for _id, (pi, fi) in enumerate(pf):
            if vis[_id]: continue
            # if min_id == -1:
            #     min_id = _id
            t = max([pi2 + fi2 for _id2, (pi2, fi2) in enumerate(pf) if _id != _id2])
            t = max(t, fi)
            if cur_min_cost > t + pi:
                cur_min_cost = t + pi
                min_id = _id
        vis[min_id] = True
        order.append(min_id)
    min_time = cal_time(order, p, f)
    return min_time, order


def test(test_cnt=10000, array_num=4, L=1, R=10):
    import random
    for i in range(test_cnt):
        n = random.randint(0, array_num)
        p = [random.randint(L, R) for _ in range(n)]
        f = [random.randint(L, R) for _ in range(n)]

        min1, order1 = min_complete_time(p[:], f[:])
        min2, order2 = brute_judge(p[:], f[:])
        # min3, order3 = min_complete_time2(p[:], f[:])
        if min1 != min2 :# or min2 != min3:
            print(min1, order1)
            print(min2, order2)
            # print(min3, order3)
            print(p)
            print(f)
            return
    print('ok')


if __name__ == '__main__':
    # test_case = [
    #     [  # 6
    #         [1, 2],
    #         [3, 4]
    #     ],
    #     [  # 14
    #         [2, 4],
    #         [4, 10]
    #     ],
    #     [  # 14
    #         [4, 2],
    #         [10, 6]
    #     ],
    #     [  # 14
    #         [4, 2, 1],
    #         [10, 6, 3],
    #     ],
    #     [  # 8
    #         [1, 4],
    #         [5, 3],
    #     ],
    #     [[], []],  # 0
    #     [[1], [1]]  # 2
    # ]
    # for p, f in test_case:
    #     print(min_complete_time(p[:], f[:]), min_complete_time2(p[:], f[:]), brute_judge(p[:], f[:]))
    test()
