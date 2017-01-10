# -*- coding: utf-8 -*-
# @Date    : 2016/9/23
# @Author  : hrwhisper

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def createTree(node):
    if not node or node[0] == '#': return None
    root, q = TreeNode(node[0]), []
    q.append(root)
    cur, n = q.pop(0), len(node)

    for i in range(1, n):
        if node[i] == '#':
            if not i & 1:
                cur = q.pop(0)
            continue
        t = TreeNode(node[i])
        q.append(t)
        if i & 1:  # left son
            cur.left = t
        else:
            cur.right = t
            cur = q.pop(0)
    return root


def printTree(root):
    q, ans = [], []
    q.append(root)
    while q:
        cur = q.pop(0)
        if cur:
            q.append(cur.left)
            q.append(cur.right)
            ans.append(cur.val)
        else:
            ans.append('#')
    print(ans)


def search_local_minimum(root):
    while root:
        if not root.left: return root
        if root.val > root.left.val:
            root = root.left
        elif root.val > root.right.val:
            root = root.right
        else:
            return root


def test(test_cnt=10000, array_num=2**5 - 1, L=1, R=1251):
    import random
    for i in range(test_cnt):
        vis = set()
        root = []
        for _ in range(array_num):
            t = random.randint(L, R)
            while t in vis:
                t = random.randint(L, R)
            vis.add(t)
            root.append(t)
        root = createTree(root)
        root = search_local_minimum(root)
        if root.left and (root.val > root.left.val or root.val > root.right.val):
            printTree(root)


if __name__ == '__main__':
    test()
    # root = createTree([1])
    # print(search_local_minimum(root).val)
    # printTree(root)
