# -*- coding: utf-8 -*-
# @Date    : 2016/11/30
# @Author  : hrwhisper

import collections
import copy


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


def test():
    with open('./problem1.data', 'r') as f, open('./1.out', 'w+') as fw:
        for i in range(3):
            f.readline()

        lines = f.read().split('\n')[::-1]
        while lines:
            global g
            g = collections.defaultdict(list)
            line = lines.pop().strip()
            if line == '': continue
            m, n = list(map(int, line.split(' ')))
            s, t = n + m, n + m + 1
            for i in range(m):
                x, y = list(map(int, lines.pop().strip().split(' ')))
                add_edge(i, x + m - 1, 1)
                add_edge(i, y + m - 1, 1)
                add_edge(s, i, 1)
            ans = binary_search(m, n, s, t)
            print(ans)
            fw.write(str(ans) + '\n')


def binary_search(m, n, s, t):
    L, R = 1, m + 1
    global g
    g2 = copy.deepcopy(g)
    while L < R:
        mid = (L + R) >> 1
        for i in range(m, m + n):
            add_edge(i, t, mid)

        if max_flow(m, n, s, t) == m:
            R = mid
        else:
            L = mid + 1
        g = copy.deepcopy(g2)
    return L


if __name__ == '__main__':
    test()
