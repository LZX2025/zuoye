# -*- coding: gbk -*-

import random

class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class Tree:
    def __init__(self):
        self.root = None

    # 递归============================================================================
    def search_re(self, key):    # 查找
        p = self.root
        return self._search_re(p, key)

    def _search_re(self, p, key):
        if p is None:
            return None
        if p.key == key:
            return p
        elif key < p.key:
            return self._search_re(p.left, key)
        else:
            return self._search_re(p.right, key)


    def insert_re(self, key):    # 插入
        self.root =  self._insert_re(self.root, key)

    def _insert_re(self, p, key):
        if p is None:
            return TreeNode(key)
        elif p.key < key:
            return self._insert_re(p.left, key)
        elif p.key > key:
            return self._insert_re(p.right, key)
        else:
            return p

    def delete_re(self, key):
        self.root = self._delete_re(self.root, key)

    def _delete_re(self, node, key):
        if node is None:
            return node

        if key < node.key:
            node.left = self._delete_re(node.left, key)
        elif key > node.key:
            node.right = self._delete_re(node.right, key)
        else:
            # 节点有一个子节点或没有子节点
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # 节点有两个子节点：获取右子树的最小节点
            min_node = self.find_min(node.right)
            node.key = min_node.key
            node.right = self._delete_re(node.right, min_node.key)

        return node

    def find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current


    def inorder_re(self):    #中序（升序）
        result = []
        self._inorder_re(self.root, result)
        return result

    def _inorder_re(self, node, result):
        if node:
            self._inorder_re(node.left, result)
            result.append(node.key)
            self._inorder_re(node.right, result)

    def preorder_re(self):    #前序
        result = []
        self._preorder_re(self.root, result)
        return result

    def _preorder_re(self, node, result):
        if node:
            result.append(node.key)
            self._preorder_re(node.left, result)
            self._preorder_re(node.right, result)

    def postorder_re(self):    #后序
        result = []
        self._postorder_re(self.root, result)
        return result

    def _postorder_re(self, node, result):
        if node:
            self._postorder_re(node.left, result)
            self._postorder_re(node.right, result)
            result.append(node.key)



    #非递归=========================================================================

    def search_it(self, key):
        node = self.root
        while node is not None and node.key != key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node

    def insert_it(self, key):
        new_node = TreeNode(key)
        if self.root is None:
            self.root = new_node
            return

        parent = None
        current = self.root
        while current is not None:
            parent = current
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                return  # 键已存在，不插入

        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

    def inorder_it(self):    # 中序
        result = []
        stack = []
        current = self.root

        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.key)
            current = current.right

        return result

    def preorder_it(self):    # 前序
        if not self.root:
            return []

        result = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            result.append(node.key)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return result

    def postorder_it(self):    #后序
        if not self.root:
            return []

        result = []
        stack = [self.root]
        prev = None

        while stack:
            current = stack[-1]
            if not prev or prev.left == current or prev.right == current:
                if current.left:
                    stack.append(current.left)
                elif current.right:
                    stack.append(current.right)
            elif current.left == prev:
                if current.right:
                    stack.append(current.right)
            else:
                result.append(current.key)
                stack.pop()
            prev = current

        return result

    def level_order(self):    # 层序
        if not self.root:
            return []

        result = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            result.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

def main():
    tree = Tree()
    r_list = random.sample(range(100), 10)
    for key in r_list:
        tree.insert_it(key)

    print("递归中序遍历:", tree.inorder_re())
    print("非递归中序遍历:", tree.inorder_it())
    print("\n递归前序遍历:", tree.preorder_re())
    print("非递归前序遍历:", tree.preorder_it())
    print("\n递归后序遍历:", tree.postorder_re())
    print("非递归后序遍历:", tree.postorder_it())
    print("\n层序遍历:", tree.level_order())

    find_key = int(input("search key"))
    print("非递归")
    if tree.search_it(find_key):
        print("find")
    else:
        print("not find")

    print("递归")
    if tree.search_re(find_key):
        print("find")
    else:
        print("not find")

    delete_key = int(input("delete key"))
    if tree.search_it(delete_key):
        tree.delete_re(delete_key)
        print("delete")

    print("层序遍历:",tree.level_order())

if __name__ == '__main__':
    main()
