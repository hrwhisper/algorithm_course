# -*- coding: utf-8 -*-
# @Date    : 2016/10/18
# @Author  : hrwhisper
import heapq
import collections


class TreeNode(object):
    def __init__(self, val, cnt, left=None, right=None):
        self.cnt = cnt
        self.val = val
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.cnt < other.cnt


def create_huffman_tree(txt):
    q = [TreeNode(c, cnt) for c, cnt in collections.Counter(txt).items()]
    heapq.heapify(q)
    while len(q) > 1:
        a, b = heapq.heappop(q), heapq.heappop(q)
        heapq.heappush(q, TreeNode('', a.cnt + b.cnt, a, b))
    return q.pop()


def get_huffman_tree(cur, root, code):
    if not root.left and not root.right:  # the leaf node
        code[root.val] = cur
        return

    if root.left:  get_huffman_tree(cur + '0', root.left, code)
    if root.right: get_huffman_tree(cur + '1', root.right, code)


def decode(txt, r_haffman_code, decode_save_path, last_byte=0):
    txt = ''.join(['0' * (8 - len(bin(ord(c))[2:])) + bin(ord(c))[2:] for c in txt])
    if last_byte:
        txt = txt[:-8] + txt[-last_byte:]
    n = len(txt)
    cur, decode_txt = '', ''
    for i in range(n):
        cur += txt[i]
        if cur in r_haffman_code:
            decode_txt += r_haffman_code[cur]
            cur = ''

    with open(decode_save_path, 'w') as f:
        f.write(decode_txt)


def encode(txt, huffman_code, compress_save_path):
    with open(compress_save_path, 'wb') as f:
        txt = ''.join([huffman_code[c] for c in txt])
        last_byte = len(txt) % 8
        txt = ''.join(chr(int(txt[i:i + 8], 2)) for i in range(0, len(txt), 8))
        f.write(bytes(txt, "utf-8"))
        return last_byte


if __name__ == '__main__':
    file_paths = ['./Aesop_Fables.txt', './graph.txt']
    for cur_file_path in file_paths:
        compress_save_path = cur_file_path + '_compressed'
        decode_save_path = cur_file_path + '_compressed_decode'

        with open(cur_file_path) as f:
            txt = f.read()

        root = create_huffman_tree(txt)
        huffman_code = {}
        get_huffman_tree('', root, huffman_code)
        r_haffman_code = {code: c for c, code in huffman_code.items()}
        last_byte = encode(txt, huffman_code, compress_save_path)

        with open(compress_save_path, 'rb') as f:
            txt = f.read().decode('utf-8')

        decode(txt, r_haffman_code, decode_save_path, last_byte)

        with open(decode_save_path) as fd, \
                open(cur_file_path) as f, open(compress_save_path) as fp:
            t = f.read()
            print('{}: compression ratio: {:.2f}, decode file equals original file is'
                .format(f.name,len(fp.read()) * 1.0 / len(t)), t == fd.read())
