# -*- coding: utf-8 -*-
# @Date    : 2016/10/18
# @Author  : hrwhisper

import heapq
import collections
import random


class Node(object):
    def __init__(self, to, val):
        self.to = to
        self.val = val

    def __lt__(self, other):
        return self.val < other.val


def dijkstra(s, t, g):
    q = []
    dis = collections.defaultdict(lambda: 0x7fffffff)  # [0x7fffffff] * len(g)
    vis = collections.defaultdict(bool)  # [False] * len(g)
    dis[s] = 0
    heapq.heappush(q, Node(s, 0))

    while q:
        cur = heapq.heappop(q).to
        if vis[cur]: continue
        vis[cur] = True
        for to, val in g[cur]:
            if not vis[to] and dis[cur] + val < dis[to]:
                dis[to] = dis[cur] + val
                heapq.heappush(q, Node(to, dis[to]))
    return dis


def count_node_path(s, t, dis, g):
    cnt = collections.defaultdict(int)  # [0] * len(g)
    q = [to for to, val in g[t] if dis[t] == dis[to] + val]
    while q:
        cur = q.pop()
        if cur == s: continue
        cnt[cur] += 1
        for to, val in g[cur]:
            if dis[cur] == dis[to] + val:
                q.append(to)
    return cnt


if __name__ == '__main__':
    g = collections.defaultdict(list)

    with open('./graph.txt') as f:
        for i, line in enumerate(f):
            if i < 6: continue
            x, y, val = list(map(int, line.strip().split()))
            g[x].append((y, val))
            g[y].append((x, val))

    s, t = random.randint(0, len(g)), random.randint(0, len(g))
    dis = dijkstra(s, t, g)
    print(s, t, dis[t])
    cnt = count_node_path(s, t, dis, g)
    print(cnt)
