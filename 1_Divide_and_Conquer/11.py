# -*- coding: utf-8 -*-
# @Date    : 2016/9/24
# @Author  : hrwhisper


def quick_mul2(x, y):
    x, y = str(x), str(y)
    if len(x) < len(y): x, y = y, x
    n = len(x)
    if n == 1: return int(x) * int(y)
    if n & 1:
        x = '0' + x
        n += 1
    y = '0' * (n - len(y)) + y

    half_n = n >> 1
    xh = int(x[:half_n])
    xl = int(x[half_n:])
    yh = int(y[:half_n])
    yl = int(y[half_n:])

    p = quick_mul2((xh + xl), (yh + yl))
    h = quick_mul2(xh, yh)
    l = quick_mul2(xl, yl)
    return h * (10 ** n) + (p - h - l) * (10 ** half_n) + l


def quick_mul(x, y):
    s_x, s_y = str(x), str(y)
    if len(s_x) == 1 or len(s_y) == 1: return x * y
    n = max(len(s_x), len(s_y))
    half_n = n >> 1
    pow_half_n = 10 ** half_n
    xh = x // pow_half_n
    xl = x % pow_half_n
    yh = y // pow_half_n
    yl = y % pow_half_n
    p = quick_mul(xh + xl, yh + yl)
    h = quick_mul(xh, yh)
    l = quick_mul(xl, yl)
    if n & 1: n -= 1
    return h * (10 ** n) + (p - h - l) * (10 ** half_n) + l


def test(test_cnt=10001, L=0, R=11122231):
    import random
    for i in range(test_cnt):
        x, y = random.randint(L, R), random.randint(L, R)
        t = x * y
        res1 = quick_mul(x, y)
        res2 = quick_mul2(x, y)
        if t != res1 or t != res2:
            print(x, y, t, res1, res2)
            return
    print('ok')


if __name__ == '__main__':
    test()
    # print(quick_mul(0, 67), 120 * 67)
