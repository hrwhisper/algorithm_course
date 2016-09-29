# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper


def search_local_minimum(sx, sy, ex, ey, g):
    if sx == ex and sy == ey:
        return sx, sy
    elif ex - sx == 1 and ey - sy == 1:
        temp = [g[sx][sy], g[sx + 1][sy], g[sx][sy + 1], g[sx + 1][sy + 1]]
        return [(sx, sy), (sx + 1, sy), (sx, sy + 1), (sx + 1, sy + 1)][temp.index(min(temp))]

    mx, my = (sx + ex) >> 1, (sy + ey) >> 1
    min_x, min_y = mx, my
    for i in range(sx, ex + 1):
        if g[min_x][min_y] > g[i][sy]: min_x, min_y = i, sy
        if g[min_x][min_y] > g[i][ey]: min_x, min_y = i, ey
        if g[min_x][min_y] > g[i][my]: min_x, min_y = i, my

    for i in range(sy, ey + 1):
        if g[min_x][min_y] > g[sx][i]: min_x, min_y = sx, i
        if g[min_x][min_y] > g[ex][i]: min_x, min_y = ex, i
        if g[min_x][min_y] > g[mx][i]: min_x, min_y = mx, i

    if min_x < mx and min_y < my:  # 左上角
        case = 0
    elif min_x > mx and min_y < my:  # 左下角
        case = 1
    elif min_x < mx and min_y > my:  # 右上角
        case = 2
    elif min_x > mx and min_y > my:  # 右下角
        case = 3
    else:
        if min_x > sx and g[min_x - 1][min_y] < g[min_x][min_y]:
            case = 0 if min_y < my else 2  # 上半部分 区分左上右上
        elif min_x < ex and g[min_x + 1][min_y] < g[min_x][min_y]:
            case = 1 if min_y < my else 3  # 下半部分 区分左下右下
        elif min_y > sy and g[min_x][min_y - 1] < g[min_x][min_y]:
            case = 0 if min_x < mx else 1  # 左半部分 区分左上左下
        elif min_y < ey and g[min_x][min_y + 1] < g[min_x][min_y]:
            case = 2 if min_x < mx else 3  # 右半部分 区分右上右下
        else:
            return min_x, min_y

    if case == 0:
        return search_local_minimum(sx, sy, mx, my, g)
    elif case == 1:
        return search_local_minimum(mx, sy, ex, my, g)
    elif case == 2:
        return search_local_minimum(sx, my, mx, ey, g)
    else:
        return search_local_minimum(mx, my, ex, ey, g)


def solve_search_local_minimum(g):
    x, y = search_local_minimum(0, 0, len(g) - 1, len(g) - 1, g)
    return g[x][y]

# solve_search_local_minimum(g)
# def search_local_minimum(sx, sy, ex, ey, g):
#     if sx == ex and sy == ey:
#         return sx, sy
#     elif ex - sx == 1 and ey - sy == 1:
#         temp = [g[sx][sy], g[sx + 1][sy], g[sx][sy + 1], g[sx + 1][sy + 1]]
#         return [(sx, sy), (sx + 1, sy), (sx, sy + 1), (sx + 1, sy + 1)][temp.index(min(temp))]
#
#     mx, my = (sx + ex) >> 1, (sy + ey) >> 1
#     min_x, min_y = mx, my
#     for i in range(sx, ex + 1):
#         if g[min_x][min_y] > g[i][sy]:
#             min_x, min_y = i, sy
#         if g[min_x][min_y] > g[i][ey]:
#             min_x, min_y = i, ey
#         if g[min_x][min_y] > g[i][my]:
#             min_x, min_y = i, my
#
#     for i in range(sy, ey + 1):
#         if g[min_x][min_y] > g[sx][i]:
#             min_x, min_y = sx, i
#         if g[min_x][min_y] > g[ex][i]:
#             min_x, min_y = ex, i
#         if g[min_x][min_y] > g[mx][i]:
#             min_x, min_y = mx, i
#
#     if min_x == mx and min_y == my:
#         return mx, my
#
#     case = 0  # 0 左上 1 左下 2 右上 3 右下
#     if min_x < mx and min_y < my:  # 左上角
#         case = 0
#     elif min_x > mx and min_y < my:  # 左下角
#         case = 1
#     elif min_x < mx and min_y > my:  # 右上角
#         case = 2
#     elif min_x > mx and min_y > my:  # 右下角
#         case = 3
#     elif min_x < mx and min_y == my:  # 上半部分
#         if g[min_x][min_y - 1] > g[min_x][min_y] and g[min_x][min_y + 1] > g[min_x][min_y]:
#             return min_x, min_y
#         elif g[min_x][min_y - 1] < g[min_x][min_y]:
#             case = 0
#         else:
#             case = 2
#     elif min_x > mx and min_y == my:  # 下半部分
#         if g[min_x][min_y - 1] > g[min_x][min_y] and g[min_x][min_y + 1] > g[min_x][min_y]:
#             return min_x, min_y
#         elif g[min_x][min_y - 1] < g[min_x][min_y]:
#             case = 1
#         else:
#             case = 3
#     elif min_x == mx and min_y < my:  # 左半部分
#         if g[min_x - 1][min_y] > g[min_x][min_y] and g[min_x + 1][min_y] > g[min_x][min_y]:
#             return min_x, min_y
#         elif g[min_x - 1][min_y] < g[min_x][min_y]:
#             case = 0
#         else:
#             case = 1
#     elif min_x == mx and min_y > my:  # 右半部分
#         if g[min_x - 1][min_y] > g[min_x][min_y] and g[min_x + 1][min_y] > g[min_x][min_y]:
#             return min_x, min_y
#         elif g[min_x - 1][min_y] < g[min_x][min_y]:
#             case = 2
#         else:
#             case = 3
#
#     if case == 0:
#         return search_local_minimum(sx, sy, mx, my, g)
#     elif case == 1:
#         return search_local_minimum(mx, sy, ex, my, g)
#     elif case == 2:
#         return search_local_minimum(sx, my, mx, ey, g)
#     else:
#         return search_local_minimum(mx, my, ex, ey, g)


def test(test_cnt=10000, max_n=320):
    print('start test cnt={} , max_n={}'.format(test_cnt, max_n))

    def ok(i, j, n, g):
        if i > 0 and g[i][j] > g[i - 1][j]: return False
        if j > 0 and g[i][j] > g[i][j - 1]: return False
        if i + 1 < n and g[i][j] > g[i + 1][j]: return False
        if j + 1 < n and g[i][j] > g[i][j + 1]: return False
        return True

    L = 1
    R = max_n ** 4
    import random
    for i in range(test_cnt):
        vis = set()
        n = random.randint(1, max_n)
        g = []
        for i in range(n):
            row = []
            for j in range(n):
                t = random.randint(L, R)
                while t in vis:
                    t = random.randint(L, R)
                row.append(t)
                vis.add(t)
            g.append(row)
        # print(g)
        i, j = search_local_minimum(0, 0, n - 1, n - 1, g)
        try:
            if not ok(i, j, n, g):
                print('Wrong Answer')
                print(i, j)
                print(g)
                return
        except IndexError as e:
            print('Except')
            print(g)
            print(i, j)
            return
    print('ok')


if __name__ == '__main__':
    # g = [
    #     [60, 58, 56, 54, 52],
    #     [59, 57, 55, 53, 51],
    #     [42, 44, 46, 48, 50],
    #     [41, 43, 45, 47, 49],
    #     [39, 38, 37, 36, 35]]
    # # g = [[79, 21, 17], [33, 73, 67], [74, 57, 23]]
    # print(search_local_minimum(0, 0, len(g) - 1, len(g) - 1, g))
    # g = [[14, 514, 556], [114, 0, 307], [501, 332, 528]]
    # print(search_local_minimum(0, 0, len(g) - 1, len(g) - 1, g))
    g = [
        [60, 58, 56, 54, 52],
        [59, 54, 55, 51, 51],
        [42, 44, 77, 48, 50],
        [41, 43, 45, 47, 49],
        [39, 38, 37, 36, 35]]
    test()
