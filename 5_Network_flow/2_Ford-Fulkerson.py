# -*- coding: utf-8 -*-
# @Date    : 2016/11/30
# @Author  : hrwhisper

import collections


class Edge(object):
    def __init__(self, to, cap, rev):
        self.to = to
        self.cap = cap
        self.rev = rev


def add_edge(from_, to, cap):
    g[from_].append(Edge(to, cap, len(g[to])))
    g[to].append(Edge(from_, 0, len(g[from_]) - 1))


def dfs(s, t, flow, vis):
    if s == t: return flow
    vis[s] = True
    for edge in g[s]:
        if not vis[edge.to] and edge.cap > 0:
            f = dfs(edge.to, t, min(flow, edge.cap), vis)
            if f:
                edge.cap -= f
                g[edge.to][edge.rev].cap += f
                return f
    return 0


def max_flow(m, n, s, t):
    flow = 0
    while True:
        vis = [False] * (m + n + 2)
        f = dfs(s, t, 0x7fffffff, vis)
        if not f: return flow
        flow += f


g = collections.defaultdict(list)


def test(row, col):
    global g
    g = collections.defaultdict(list)

    m = len(row)
    n = len(col)
    s = 0
    t = m + n + 1
    for i in range(1, m + 1):  # s link to [1,m]
        add_edge(0, i, row[i - 1])
        for j in range(m + 1, m + n + 1):  # row link to column
            add_edge(i, j, 1)

    for i in range(m + 1, m + n + 1):
        add_edge(i, t, col[i - m - 1])

    print(max_flow(m, n, s, t), sum(row), sum(col))
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(1, m + 1):
        for edge in g[i]:
            if edge.to != 0:
                matrix[i - 1][edge.to - m - 1] = 1 - edge.cap

    # for i in matrix:
    #     print(i)

    import numpy as np
    row_1 = np.sum(matrix, axis=1).tolist()
    col_1 = np.sum(matrix, axis=0).tolist()
    print(row_1 == row)
    print(col_1 == col)


if __name__ == '__main__':
    # test()
    with open('./problem2.data') as f:
        for i in range(3):
            f.readline()

        lines = f.read().split('\n')[::-1]
        while lines:
            line = lines.pop().strip()
            if line == '': break
            m, n = line.split(' ')
            row = list(map(int, lines.pop().strip().split(' ')))
            col = list(map(int, lines.pop().strip().split(' ')))
            test(row, col)
            # 2 1
            # 0 1
            # test([3, 1], [2, 2])
