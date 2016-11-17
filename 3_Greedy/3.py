# -*- coding: utf-8 -*-
# @Date    : 2016/10/22
# @Author  : hrwhisper
from math import sqrt


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        return self.x < other.x


def min_radar(points, d):
    if not points: return 0
    if len(list(filter(lambda point: point.y > d, points))) > 0: return -1  # have no answer
    points.sort()
    px = points[0].x + sqrt(d * d - points[0].y * points[0].y)
    ans = 1
    for i in range(1,len(points)):
        if (px - points[i].x)**2 + points[i].y * points[i].y <= d*d: continue
        cx = points[i].x + sqrt(d * d - points[i].y * points[i].y)
        if cx < px:
            px = cx
            continue
        px = cx
        ans += 1
    return ans


if __name__ == '__main__':
    test_case = [
        ([Point(1, 2), Point(-3, 1), Point(2, 1)],2),
        ([Point(0, 2)],2)
    ]
    for points,d in test_case:
        print(min_radar(points,d))



