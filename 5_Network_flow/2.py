# -*- coding: utf-8 -*-
# @Date    : 2016/11/30
# @Author  : hrwhisper

import collections


class Edge(object):
    def __init__(self, to, cap, rev):
        self.to = to
        self.cap = cap
        self.rev = rev


class Vertex(object):
    def __init__(self, id_, flow):
        self.id = id_
        self.flow = flow


def add_edge(from_, to, cap):
    g[from_].append(Edge(to, cap, len(g[to])))
    g[to].append(Edge(from_, 0, len(g[from_]) - 1))


def found_vertex_non_zero(n, nodes):
    for i in range(n - 1):
        if nodes[i].flow > 0: return i
    return -1


def max_flow(m, n, s, t):
    N = m + n + 2
    h = [0] * N
    nodes = [Vertex(i, 0) for i in range(N)]
    h[s] = N
    for edge in g[s]:
        g[edge.to][edge.rev].cap += edge.cap
        nodes[edge.to].flow = edge.cap
        edge.cap = 0

    while True:
        cur = found_vertex_non_zero(N, nodes)
        if cur == -1:  break

        found = False
        for edge in g[cur]:
            if edge.cap > 0 and h[cur] > h[edge.to]:
                found = True
                f = min(nodes[cur].flow, edge.cap)
                edge.cap -= f
                g[edge.to][edge.rev].cap += f
                nodes[cur].flow -= f
                nodes[edge.to].flow += f
                if nodes[cur].flow == 0:
                    break
        if not found:
            h[cur] += 1
    return nodes[-1].flow


g = collections.defaultdict(list)


def test(row, col):
    global g
    g = collections.defaultdict(list)

    m = len(row)
    n = len(col)
    s = 0
    t = m + n + 1
    for i in range(1, m + 1):  # link source s to [1,m]
        add_edge(0, i, row[i - 1])
        for j in range(m + 1, m + n + 1):  # row link to column
            add_edge(i, j, 1)

    for i in range(m + 1, m + n + 1):
        add_edge(i, t, col[i - m - 1])  # link column to sink t

    print(max_flow(m, n, s, t), sum(row), sum(col))

    matrix = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(1, m + 1):
        for edge in g[i]:
            if edge.to != 0:
                matrix[i - 1][edge.to - m - 1] = 1 - edge.cap

    print(is_correct(row, col, matrix))
    return matrix


def is_correct(row, col, matrix):
    import numpy as np
    row_1 = np.sum(matrix, axis=1).tolist()
    col_1 = np.sum(matrix, axis=0).tolist()
    return row_1 == row and col_1 == col


if __name__ == '__main__':
    # test()
    with open('./problem2.data') as f, open('./2.out', 'w+') as fw:
        for i in range(3):
            f.readline()

        lines = f.read().split('\n')[::-1]
        while lines:
            line = lines.pop().strip()
            if line == '': break
            m, n = line.split(' ')
            row = list(map(int, lines.pop().strip().split(' ')))
            col = list(map(int, lines.pop().strip().split(' ')))
            matrix = test(row, col)
            for row in matrix:
                for x in row:
                    fw.write(str(x) + ' ')
                fw.write('\n')
            fw.write('\n\n')
            # 2 1
            # 0 1
            # test([3, 1], [2, 2])
