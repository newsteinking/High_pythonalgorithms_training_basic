Tree - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

94. Binary Tree inorder traversal
----------------------------------------

.. code-block:: python

    Given a binary tree, return the inorder traversal of its nodes' values.


    For example:
    Given binary tree [1,null,2,3],

       1
        \
         2
        /
       3



    return [1,3,2].


    Note: Recursive solution is trivial, could you do it iteratively?


    =================================================================
    class Solution(object):
      def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res, stack = [], [(1, root)]
        while stack:
          p = stack.pop()
          if not p[1]: continue
          stack.extend([(1, p[1].right), (0, p[1]), (1, p[1].left)]) if p[0] != 0 else res.append(p[1].val)
        return res


    =================================================================
    class Solution(object):
        # iteratively
        def inorderTraversal(self, root):
            tree_stack = []
            inorder_tra = []
            while root or tree_stack:
                # Go along the left child
                if root:
                    tree_stack.append(root)
                    root = root.left
                # Meet a left, go back to the parent node
                else:
                    node = tree_stack.pop()
                    inorder_tra.append(node.val)
                    root = node.right

            return inorder_tra


    class Solution_2(object):
        # recursively
        def inorderTraversal(self, root):
            inorder_tra = []
            self.helper(root, inorder_tra)
            return inorder_tra

        def helper(self, root, inorder_tra):
            if root:
                self.helper(root.left, inorder_tra)
                inorder_tra.append(root.val)
                self.helper(root.right, inorder_tra)

    """
    []
    [1]
    [1,2,3,null,null,4,null,null,5]
    """


95. Unique Binary Search Trees 2
----------------------------------

.. code-block:: python

    Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1...n.


    For example,
    Given n = 3, your program should return all 5 unique BST's shown below.


       1         3     3      2      1
        \       /     /      / \      \
         3     2     1      1   3      2
        /     /       \                 \
       2     1         2                 3

    =================================================================
    class Solution(object):
      def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """

        def clone(root, offset):
          if root:
            newRoot = TreeNode(root.val + offset)
            left = clone(root.left, offset)
            right = clone(root.right, offset)
            newRoot.left = left
            newRoot.right = right
            return newRoot

        if not n:
          return []
        dp = [[]] * (n + 1)
        dp[0] = [None]
        for i in range(1, n + 1):
          dp[i] = []
          for j in range(1, i + 1):
            for left in dp[j - 1]:
              for right in dp[i - j]:
                root = TreeNode(j)
                root.left = left
                root.right = clone(right, j)
                dp[i].append(root)
        return dp[-1]


    =================================================================
    class Solution(object):
        def generateTrees(self, n):
            """
            :type n: int
            :rtype: List[TreeNode]
            """
            if not n:
                return [[]]
            roots_lsit = self.root_list(1, n)
            return roots_lsit

        # Get all the roots of the BST's that store values start...end
        def root_list(self, start, end):
            # Null Tree when start > end
            if start > end:
                return []
            # Tree has just a root when start==end
            if start == end:
                return [TreeNode(start)]

            roots = []
            for i in range(start, end + 1):
                # Get all the possible roots and it's left, right childs
                left_childs = self.root_list(start, i-1)
                right_childs = self.root_list(i+1, end)
                # Have no left childs
                if not left_childs and right_childs:
                    for child in right_childs:
                        root_node = TreeNode(i)
                        root_node.right = child
                        root_node.left = None
                        roots.append(root_node)
                # Have no right childs
                elif not right_childs and left_childs:
                    for child in left_childs:
                        root_node = TreeNode(i)
                        root_node.left = child
                        root_node.right = None
                        roots.append(root_node)
                # Have both left childs and right childs
                else:
                    for l_child in left_childs:
                        for r_child in right_childs:
                            root_node = TreeNode(i)
                            root_node.left = l_child
                            root_node.right = r_child
                            roots.append(root_node)

            return roots

    """
    0
    1
    2
    3
    7
    """


96. Unique Binary Search Trees
----------------------------------

.. code-block:: python

    Given n, how many structurally unique BST's (binary search trees) that store values 1...n?


    For example,
    Given n = 3, there are a total of 5 unique BST's.


       1         3     3      2      1
        \       /     /      / \      \
         3     2     1      1   3      2
        /     /       \                 \
       2     1         2                 3



    =================================================================
    class Solution(object):
      def _numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
          for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
        return dp[-1]

      def numTrees(self, n):
        ans = 1
        for i in range(1, n + 1):
          ans = ans * (n + i) / i
        return ans / (n + 1)


    =================================================================
    class Solution(object):
        def numTrees(self, n):
            if not n:
                return 1
            dp = [0 for i in range(n + 1)]
            dp[0] = dp[1] = 1
            for i in range(2, n + 1):
                dp[i] += dp[i - 1] * 2
                # Get the symmetric left and right childs
                for j in range(1, (i - 2) / 2 + 1):
                    dp[i] += dp[j] * dp[i - 1 - j] * 2

                # Get the mid one whitout symmetry.
                if (i - 2) % 2 != 0:
                    mid_once = (i - 1) / 2
                    dp[i] += dp[mid_once] * dp[i - 1 - mid_once]

            return dp[n]

    """
    0
    1
    3
    15
    """


99. Recover Binary Search Tree
--------------------------------

.. code-block:: python

    Two elements of a binary search tree (BST) are swapped by mistake.

    Recover the tree without changing its structure.


    Note:
    A solution using O(n) space is pretty straight forward. Could you devise a constant space solution?


    =================================================================
    class Solution:
      def __init__(self):
        self.n1 = None
        self.n2 = None
        self.pre = None

      def findBadNode(self, root):
        if root is None: return
        self.findBadNode(root.left)
        if self.pre is not None:
          if root.val < self.pre.val:
            if self.n1 is None:
              self.n1 = self.pre
              self.n2 = root
            else:
              self.n2 = root
        self.pre = root
        self.findBadNode(root.right)

      def recoverTree(self, root):
        self.findBadNode(root)
        if self.n1 is not None and self.n2 is not None:
          self.n1.val, self.n2.val = self.n2.val, self.n1.val


    =================================================================
    class Solution(object):
        conflict_first = None
        conflict_second = None
        pre_node = None

        def recoverTree(self, root):
            """
            :type root: TreeNode
            :rtype: void Do not return anything, modify root in-place instead.
            """
            if not root:
                return None

            self.find_conflict(root)

            self.conflict_first.val, self.conflict_second.val = (
                self.conflict_second.val, self.conflict_first.val)

        # Do the inorder traversal and when find a decreasing pair,
        # then we find one (maybe all the two) node which is swapped.
        def find_conflict(self, root):
            if root.left:
                self.find_conflict(root.left)

            if self.pre_node and root.val < self.pre_node.val:
                if not self.conflict_first:
                    self.conflict_first = self.pre_node
                self.conflict_second = root

            self.pre_node = root
            if root.right:
                self.find_conflict(root.right)
    """
    [0,1]
    [8,9,13,2,6,4,14]
    [9,4,13,2,6,8,14]
    """


100. Same Tree
--------------------

.. code-block:: python

    Given two binary trees, write a function to check if they are equal or not.


    Two binary trees are considered equal if they are structurally identical and the nodes have the same value.


    =================================================================
    class Solution(object):
      def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p or not q:
          return p == q
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    =================================================================
    class Solution(object):
        def isSameTree(self, p, q):
            """
            :type p: TreeNode
            :type q: TreeNode
            :rtype: bool
            """
            if not p and not q:
                return True
            if (not p and q) or (p and not q):
                return False

            if p.val != q.val:
                return False
            if not self.isSameTree(p.left, q.left):
                return False
            if not self.isSameTree(p.right, q.right):
                return False

            return True

    """
    []
    [1]
    [1,2,3]
    [1,2,3]
    [2,null,3,4,5]
    [2,null,3,5,4]
    """


105. Contruct binary tree from preorder and inorder traversal
----------------------------------------------------------------

.. code-block:: python

    Given preorder and inorder traversal of a tree, construct the binary tree.

    Note:
    You may assume that duplicates do not exist in the tree.


    =================================================================
    class Solution(object):
      def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        self.preindex = 0
        ind = {v: i for i, v in enumerate(inorder)}
        head = self.dc(0, len(preorder) - 1, preorder, inorder, ind)
        return head

      def dc(self, start, end, preorder, inorder, ind):
        if start <= end:
          mid = ind[preorder[self.preindex]]
          self.preindex += 1
          root = TreeNode(inorder[mid])
          root.left = self.dc(start, mid - 1, preorder, inorder, ind)
          root.right = self.dc(mid + 1, end, preorder, inorder, ind)
          return root

    =================================================================
    class Solution(object):
        def buildTree(self, preorder, inorder):
            """
            :type preorder: List[int]
            :type inorder: List[int]
            :rtype: TreeNode
            """
            preorder_l = len(preorder)
            inorder_dict = dict(zip(inorder, xrange(preorder_l)))

            if not preorder:
                return None

            return self.recursve_build(
                preorder, 0, preorder_l-1,
                inorder, 0, preorder_l-1,
                inorder_dict)

        def recursve_build(
                self, preorder, p_start, p_end,
                inorder, i_start, i_end, pos_dict):
            # Empty tree
            if p_start > p_end:
                return None
            # Leaf
            if p_start == p_end:
                return TreeNode(preorder[p_start])

            root_val = preorder[p_start]
            root = TreeNode(root_val)

            # Get the left and right part of inorder
            inorder_pos = pos_dict[root_val]
            left_i_start = i_start
            left_i_end = inorder_pos - 1
            right_i_start = inorder_pos + 1
            right_i_end = i_end

            # Get the left and right part of preorder
            p_len = left_i_end - left_i_start
            left_p_start = p_start + 1
            left_p_end = left_p_start + p_len
            right_p_start = left_p_end + 1
            right_p_end = p_end

            # Get the left and right childrens
            root.left = self.recursve_build(
                preorder, left_p_start, left_p_end,
                inorder, left_i_start, left_i_end,
                pos_dict)
            root.right = self.recursve_build(
                preorder, right_p_start, right_p_end,
                inorder, right_i_start, right_i_end,
                pos_dict)

            return root

    """
    []
    []
    [10,8,3,2,11,5,7,9]
    [3,8,2,10,5,11,7,9]
    [7,10,4,3,1,2,8,11]
    [4,10,3,1,7,11,8,2]
    """


106. Construct binary tree from inorder and preorder traversal
------------------------------------------------------------------

.. code-block:: python

    Given inorder and postorder traversal of a tree, construct the binary tree.

    Note:
    You may assume that duplicates do not exist in the tree.


    =================================================================
    class Solution(object):
      def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if inorder and postorder:
          postorder.reverse()
          self.index = 0
          d = {}
          for i in range(0, len(inorder)):
            d[inorder[i]] = i
          return self.dfs(inorder, postorder, 0, len(postorder) - 1, d)

      def dfs(self, inorder, postorder, start, end, d):
        if start <= end:
          root = TreeNode(postorder[self.index])
          mid = d[postorder[self.index]]
          self.index += 1
          root.right = self.dfs(inorder, postorder, mid + 1, end, d)
          root.left = self.dfs(inorder, postorder, start, mid - 1, d)
          return root


    =================================================================
    class Solution(object):
        def buildTree(self, inorder, postorder):
            """
            :type inorder: List[int]
            :type postorder: List[int]
            :rtype: TreeNode
            """
            if not inorder:
                return None
            inorder_l = len(inorder)
            inorder_dict = dict(zip(inorder, xrange(inorder_l)))
            return self.recursve_build(
                inorder, 0, inorder_l - 1,
                postorder, 0, inorder_l - 1,
                inorder_dict)

        def recursve_build(
                self, inorder, i_start, i_end,
                postorder, p_start, p_end, inorder_dict):
            # Empty tree
            if i_start > i_end:
                return None
            if i_start == i_end:
                return TreeNode(inorder[i_start])

            root_val = postorder[p_end]
            root = TreeNode(root_val)

            # Get the left and right part of inorder
            inorder_pos = inorder_dict[root_val]
            l_i_start = i_start
            l_i_end = inorder_pos - 1
            r_i_start = inorder_pos + 1
            r_i_end = i_end

            # Get the left and right part of postorder
            l_p_len = l_i_end - l_i_start
            l_p_start = p_start
            l_p_end = l_p_start + l_p_len
            r_p_start = l_p_end + 1
            r_p_end = p_end - 1

            # Get the left and right childrens
            root.left = self.recursve_build(
                inorder, l_i_start, l_i_end,
                postorder, l_p_start, l_p_end,
                inorder_dict)
            root.right = self.recursve_build(
                inorder, r_i_start, r_i_end,
                postorder, r_p_start, r_p_end,
                inorder_dict)
            return root
    """
    []
    []
    [10,8,3,2,11,5,7,9]
    [3,8,2,10,5,11,7,9]
    [7,10,4,3,1,2,8,11]
    [4,10,3,1,7,11,8,2]
    """


108. Convert Sorted array to binary search tree
-------------------------------------------------

.. code-block:: python

    Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

    =================================================================
    class Solution(object):
      def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if nums:
          midPos = len(nums) / 2
          mid = nums[midPos]
          root = TreeNode(mid)
          root.left = self.sortedArrayToBST(nums[:midPos])
          root.right = self.sortedArrayToBST(nums[midPos + 1:])
          return root

    =================================================================
    class Solution(object):
        def sortedArrayToBST(self, nums):
            """
            :type nums: List[int]
            :rtype: TreeNode
            """
            if not nums:
                return None
            return self.get_root(nums)

        def get_root(self, nums):
            if not nums:
                return None
            nums_l = len(nums)
            if nums_l == 1:
                return TreeNode(nums[0])

            # Find the root of the current balanced BST,
            # which is conconverted by the current nums.
            """
            height = 1                  # The height of the balanced BST
            while nums_l > 2 ** height - 1:
                height += 1

            half_child_leaves = 2 ** (height-1) / 2
            full_level_nodes = 2 ** (height-1) - 1
            left_child_nodes = nums_l - full_level_nodes - half_child_leaves
            left_child_nodes = 0 if left_child_nodes < 0 else left_child_nodes
            root_index = full_level_nodes / 2 + left_child_nodes
            """
            root_index = nums_l / 2
            root = TreeNode(nums[root_index])
            root.left = self.get_root(nums[:root_index])
            root.right = self.get_root(nums[root_index+1:])
            return root

    """
    []
    [1,3,5,7]
    [1,3,5,7,9,11]
    [1,3,5,7,9,11,13]
    """


109. Convert sorted list to binary search tree
------------------------------------------------

.. code-block:: python

    Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

    =================================================================

    class Solution(object):
      def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if head:
          pre = None
          slow = fast = head
          while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
          root = TreeNode(slow.val)
          if pre:
            pre.next = None
            root.left = self.sortedListToBST(head)
          root.right = self.sortedListToBST(slow.next)
          return root


    =================================================================
    class Solution(object):
        def sortedListToBST(self, head):
            """
            :type head: ListNode
            :rtype: TreeNode
            """
            nums = []
            while head:
                nums.append(head.val)
                head = head.next
            if not nums:
                return None
            return self.get_root(nums)

        def get_root(self, nums):
            if not nums:
                return None
            nums_l = len(nums)
            if nums_l == 1:
                return TreeNode(nums[0])

            # Find the root of the current balanced BST,
            # which is conconverted by the current nums.
            root_index = nums_l / 2
            root = TreeNode(nums[root_index])
            root.left = self.get_root(nums[:root_index])
            root.right = self.get_root(nums[root_index+1:])
            return root

    """
    []
    [1,3,5,7]
    [1,3,5,7,9,11]
    [1,3,5,7,9,11,13]
    """



110. Balanced binary tree
-------------------------------

.. code-block:: python

    Given a binary tree, determine if it is height-balanced.



    For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

    =================================================================
    class Solution(object):
      def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def dfs(p):
          if not p:
            return 0

          left = dfs(p.left)
          right = dfs(p.right)
          if left == -1 or right == -1:
            return -1
          if abs(left - right) > 1:
            return -1
          return 1 + max(left, right)

        if dfs(root) == -1:
          return False
        return True


    =================================================================
    class Solution(object):
        def isBalanced(self, root):
            return self.depth_sub(root) != -1

        # When get depth of subtree, we check if it's balanced at the same time.
        # if subtree of one node is not balanced, then it's height is -1
        def depth_sub(self, root):
            if not root:
                return 0

            left = self.depth_sub(root.left)
            right = self.depth_sub(root.right)

            if abs(left - right) > 1 or left == -1 or right == -1:
                return -1

            return 1 + max(left, right)

    """
    class Solution(object):
        def isBalanced(self, root):
            if not root:
                return True

            if abs(self.depth(root.left) - self.depth(root.right)) > 1:
                return False
            if not self.isBalanced(root.left) or not self.isBalanced(root.right):
                return False
            return True

        # Get the tree's height
        def depth(self, root):
            if not root:
                return 0

            node_list = [root]
            depth_count = 1
            # Breadth-first Search
            while node_list:
                node_scan = node_list[:]
                node_list = []
                for node in node_scan:
                    l_child = node.left
                    r_child = node.right
                    if l_child:
                        node_list.append(l_child)
                    if r_child:
                        node_list.append(r_child)
                if node_list:
                    depth_count += 1

            return depth_count

        # Get the tree's height: recursion
        def depth_two(self, root):
            if not root:
                return 0
            if root.left or root.right:
                return 1 + max(self.depth(root.left), self.depth(root.right))
            else:
                return 1
    """
    """
    []
    [1]
    [1,2,null,3]
    [1,2,3,4,null,6,7,5,8]
    [1,2,2,3,null,null,3,4,null,null,4]
    """


111. Minimum depth of binary tree
------------------------------------

.. code-block:: python

    Given a binary tree, find its minimum depth.

    The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

    =================================================================
    class Solution(object):
      def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
          return 0
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if not left and not right:
          return 1
        elif not left:
          return right + 1
        elif not right:
          return left + 1
        else:
          return min(left, right) + 1


    =================================================================

    class Solution(object):
        def minDepth(self, root):
            """
            :type root: TreeNode
            :rtype: int
            """
            if not root:
                return 0

            if root.left and root.right:
                return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
            if not root.left:
                return 1 + self.minDepth(root.right)
            if not root.right:
                return 1 + self.minDepth(root.left)
            else:
                return 1

    """
    []
    [1]
    [1,2,null,3]
    [1,2,3,4,null,6,7,5,8]
    [1,2,2,3,null,null,3,4,null,null,4]
    """


112. Path sum
--------------------

.. code-block:: python

    Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.


    For example:
    Given the below binary tree and sum = 22,

                  5
                 / \
                4   8
               /   / \
              11  13  4
             /  \      \
            7    2      1



    return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.

    =================================================================
    from collections import deque


    class Solution(object):
      def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root:
          queue = deque([(root, root.val)])
          while queue:
            p, s = queue.popleft()
            left, right = p.left, p.right
            if left:
              queue.append((p.left, s + p.left.val))
            if right:
              queue.append((p.right, s + p.right.val))
            if not left and not right and s == sum:
              return True
          return False
        return False


    =================================================================
    class Solution(object):
        def hasPathSum(self, root, sum):
            if not root:
                return False

            root_val = root.val
            if root.left and self.hasPathSum(root.left, sum-root_val):
                return True
            if root.right and self.hasPathSum(root.right, sum-root_val):
                return True
            if not root.left and not root.right and sum == root.val:
                return True
            return False

    """
    []
    0
    [1,2,3,4,null,6,7,5,8]
    15
    [1,2,2,3,null,null,3,4,null,null,4]
    9
    """


113. Path sum 2
--------------------

.. code-block:: python

    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.


    For example:
    Given the below binary tree and sum = 22,

                  5
                 / \
                4   8
               /   / \
              11  13  4
             /  \    / \
            7    2  5   1



    return

    [
       [5,4,11,2],
       [5,8,4,5]
    ]



    =================================================================
    class Solution(object):
      def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """

        def dfs(root, s, path, res):
          if root:
            path.append(root.val)
            s -= root.val
            left = dfs(root.left, s, path, res)
            right = dfs(root.right, s, path, res)
            if not left and not right and s == 0:
              res.append(path + [])
            path.pop()
            return True

        res = []
        dfs(root, sum, [], res)
        return res


    =================================================================
    class Solution(object):
        def pathSum(self, root, sum):
            if not root:
                return []
            paths_list = []
            if not root.left and not root.right:
                if root.val == sum:
                    paths_list.append([root.val])
                return paths_list

            if root.left:
                l_paths = self.pathSum(root.left, sum-root.val)
                # There are paths along root.left
                if l_paths:
                    for path in l_paths:
                        one_path = [root.val]
                        one_path.extend(path)
                        paths_list.append(one_path)

            if root.right:
                r_paths = self.pathSum(root.right, sum-root.val)
                # There are paths along root.right
                if r_paths:
                    for path in r_paths:
                        one_path = [root.val]
                        one_path.extend(path)
                        paths_list.append(one_path)
            return paths_list


    # Pythonic way.  So short and beautiful!
    class Solution_2(object):
        def pathSum(self, root, sum):
            if not root:
                return []
            if not root.left and not root.right and sum == root.val:
                return [[root.val]]
            tmp = (self.pathSum(root.left, sum-root.val) +
                   self.pathSum(root.right, sum-root.val))
            return [[root.val] + i for i in tmp]


    """
    []
    0
    [1,2,3,4,null,6,7,5,8]
    15
    [1,2,2,3,3,3,3]
    6
    """


114. Flatten binary tree to linked list
-------------------------------------------

.. code-block:: python

    Given a binary tree, flatten it to a linked list in-place.



    For example,
    Given

             1
            / \
           2   5
          / \   \
         3   4   6



    The flattened tree should look like:

       1
        \
         2
          \
           3
            \
             4
              \
               5
                \
                 6


    click to show hints.

    Hints:
    If you notice carefully in the flattened tree, each node's right child points to the next node of a pre-order traversal.


    =================================================================
    class Solution(object):
      def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """

        def dfs(root):
          if not root:
            return root

          left = dfs(root.left)
          right = dfs(root.right)

          if not left and not right:
            return root

          if right is None:
            root.right = root.left
            root.left = None
            return left

          if not left:
            return right

          tmp = root.right
          root.right = root.left
          root.left = None
          left.right = tmp
          return right

        dfs(root)


    =================================================================
    class Solution(object):
        def flatten(self, root):
            """
            :type root: TreeNode
            :rtype: void Do not return anything, modify root in-place instead.
            """
            if not root:
                return None

            self.get_list(root)

        # Flatten the tree to a linked list in-place, and return it's tail.
        def get_list(self, root):
            left_child = root.left
            right_child = root.right

            # Leaf node: do nothing, and the linked list has just one node.
            if not left_child and not right_child:
                return root

            # Have left child node, move it to the next node in the linked list.
            # Flatten the left subtree and then get the tail
            # of the flattened subtree's linked list. Make the right child go after
            # the tail, and flatten the right subtree at last.
            if left_child:
                root.left = None
                root.right = left_child
                left_tail_node = self.get_list(left_child)

                if right_child:
                    left_tail_node.right = right_child
                    return self.get_list(right_child)
                else:
                    return left_tail_node
            # No left child node, just flatten the right node.
            else:
                return self.get_list(right_child)

    """
    []
    [1,2,3,null,null,4,5]
    [1,2,5,3,4,null,6]
    """


116. Populating next right pointers in each node
--------------------

.. code-block:: python

    Given a binary tree

        struct TreeLinkNode {
          TreeLinkNode *left;
          TreeLinkNode *right;
          TreeLinkNode *next;
        }



    Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

    Initially, all next pointers are set to NULL.


    Note:

    You may only use constant extra space.
    You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).




    For example,
    Given the following perfect binary tree,

             1
           /  \
          2    3
         / \  / \
        4  5  6  7



    After calling your function, the tree should look like:

             1 -> NULL
           /  \
          2 -> 3 -> NULL
         / \  / \
        4->5->6->7 -> NULL



    =================================================================
    class Solution:
      # @param root, a tree link node
      # @return nothing
      def connect(self, root):
        if root and root.left and root.right:
          root.left.next = root.right
          root.right.next = root.next and root.next.left
          return self.connect(root.left) or self.connect(root.right)

    =================================================================
    class Solution(object):
        def connect(self, root):
            if not root:
                return None

            cur_head = root
            next_head = None

            # Breadth-first Scan
            while cur_head:
                if cur_head.left:
                    # Get the next level's head
                    if not next_head:
                        next_head = cur_head.left
                    cur_head.left.next = cur_head.right
                if cur_head.right and cur_head.next:
                    cur_head.right.next = cur_head.next.left

                cur_head = cur_head.next

                # Go to next level.
                if not cur_head:
                    cur_head = next_head
                    next_head = None

    """ Readable implementation
    class Solution(object):
        def connect(self, root):

            # For all the non-empty nodes:
            #     node.left.next = node.right
            #     node.right.next = node.next.left(if node.next not none)

            if not root:
                return None
            if root.left:
                root.left.next = root.right
            if root.next and root.right:
                root.right.next = root.next.left

            self.connect(root.left)
            self.connect(root.right)
    """

    """
    [0]
    [1,2,3]
    [0,1,2,3,4,5,6]
    """


117. populating next right pointers in each node 2
-----------------------------------------------------

.. code-block:: python

    Follow up for problem "Populating Next Right Pointers in Each Node".
    What if the given tree could be any binary tree? Would your previous solution still work?

    Note:
    You may only use constant extra space.


    For example,
    Given the following binary tree,

             1
           /  \
          2    3
         / \    \
        4   5    7



    After calling your function, the tree should look like:

             1 -> NULL
           /  \
          2 -> 3 -> NULL
         / \    \
        4-> 5 -> 7 -> NULL



    =================================================================
    class Solution:
      # @param root, a tree link node
      # @return nothing
      def connect(self, root):
        p = root
        pre = None
        head = None
        while p:
          if p.left:
            if pre:
              pre.next = p.left
            pre = p.left
          if p.right:
            if pre:
              pre.next = p.right
            pre = p.right
          if not head:
            head = p.left or p.right
          if p.next:
            p = p.next
          else:
            p = head
            head = None
            pre = None


    =================================================================
    class Solution(object):
        def connect(self, root):
            if not root:
                return None

            cur_head = root
            next_head = None
            # Breadth-first Scan
            while cur_head:
                # Get the next node cur_head's child point to.
                next_node = cur_head.next
                while next_node:
                    if next_node.left:
                        next_node = next_node.left
                        break
                    if next_node.right:
                        next_node = next_node.right
                        break
                    next_node = next_node.next

                if cur_head.left:
                    if not next_head:
                        next_head = cur_head.left
                    if cur_head.right:
                        cur_head.left.next = cur_head.right
                    else:
                        cur_head.left.next = next_node
                if cur_head.right:
                    if not next_head:
                        next_head = cur_head.right
                    cur_head.right.next = next_node
                cur_head = cur_head.next

                # Go to next level.
                if not cur_head:
                    cur_head = next_head
                    next_head = None

    """ Readable implementation
    class Solution(object):
        def connect(self, root):
            # For all the non-empty nodes:
            #     node.left.next = right(or next_node)
            #     node.right.next = next_node, (or right)
            if not root:
                return None

            next_node = root.next
            while next_node:
                if next_node.left:
                    next_node = next_node.left
                    break
                if next_node.right:
                    next_node = next_node.right
                    break
                next_node = next_node.next

            if root.left:
                if root.right:
                    root.left.next = root.right
                else:
                    root.left.next = next_node

            if root.right:
                root.right.next = next_node

            # Get root.right done firstly because when we compute root.left,
            # we may use the node's next relationship in connect(root.right).
            self.connect(root.right)
            self.connect(root.left)
    """

    """
    [0]
    [1,2,3,4,5,null,7]
    """


124. Binary tree maximum path sum
-----------------------------------

.. code-block:: python

    Given a binary tree, find the maximum path sum.


    For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.


    For example:
    Given the below binary tree,

           1
          / \
         2   3



    Return 6.


    =================================================================
    class Solution(object):
      def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def dfs(root):
          if not root: return (0, float("-inf"))
          (l, lm), (r, rm) = map(dfs, [root.left, root.right])
          return (max(root.val, root.val + max(l, r)), max(lm, rm, root.val + max(l, r), root.val, root.val + l + r))

        return dfs(root)[1]


    =================================================================
    class Solution(object):

        def maxPathSum(self, root):
            result = []
            self.max_path(root, result)
            return result[0]

        """
        Return value root_path_sum: the max sum of the path
        which go from any sub-nodes to the current root.
        So we can extend the path go through the current root's parent.
        At the same time, we update the final max_path_sum
        with the root_path_sum and a path go through root's left and right
        """
        def max_path(self, root, max_path_sum):
            if not root:
                return 0

            left_path_sum = self.max_path(root.left, max_path_sum)
            right_path_sum = self.max_path(root.right, max_path_sum)

            # Get the max sum of path that fomr sub-nodes to current root
            max_path = max(left_path_sum, right_path_sum)
            root_path_sum = max(max_path+root.val, root.val)

            # update the max path sum
            gothrough_root_path = left_path_sum + right_path_sum + root.val
            if not max_path_sum:
                max_path_sum.append(max(root_path_sum, gothrough_root_path))
            else:
                max_path_sum[0] = max(root_path_sum, gothrough_root_path,
                                      max_path_sum[0])
            return root_path_sum

    """
    []
    [-3]
    [1,2,3]
    [1,2,3,4,5,-5,-4]
    [-6,-2,3,4,5,-5,-4]
    """


144. Binary tree preorder traversal
---------------------------------------

.. code-block:: python

    Given a binary tree, return the preorder traversal of its nodes' values.


    For example:
    Given binary tree {1,#,2,3},

       1
        \
         2
        /
       3



    return [1,2,3].


    Note: Recursive solution is trivial, could you do it iteratively?

    =================================================================
    class Solution(object):
      def preorderTraversal(self, root):
        res, stack = [], [(1, root)]
        while stack:
          p = stack.pop()
          if not p[1]: continue
          stack.extend([(1, p[1].right), (1, p[1].left), (0, p[1])]) if p[0] != 0 else res.append(p[1].val)
        return res


    =================================================================
    class Solution(object):
        # Preorder Traversal
        def preorderTraversal(self, root):
            if not root:
                return []
            result = []
            node_stack = []
            while root or node_stack:
                if root:
                    node_stack.append(root)
                    result.append(root.val)
                    root = root.left
                else:
                    node = node_stack.pop()
                    root = node.right
            return result

    """
    []
    [1, null, 2, 3]
    [1, null, 2, 3, null, 4, 5]
    """


145. Binary Tree postorder traversal
----------------------------------------

.. code-block:: python

    Given a binary tree, return the postorder traversal of its nodes' values.


    For example:
    Given binary tree {1,#,2,3},

       1
        \
         2
        /
       3



    return [3,2,1].


    Note: Recursive solution is trivial, could you do it iteratively?

    =================================================================
    class Solution(object):
      def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res, stack = [], [(1, root)]
        while stack:
          p = stack.pop()
          if not p[1]:
            continue
          if p[0] == 0:
            res.append(p[1].val)
          else:
            stack.extend([(0, p[1]), (1, p[1].right), (1, p[1].left)])
        return res

    =================================================================
    class Solution(object):
        # Postorder Traversal
        def postorderTraversal(self, root):
            if not root:
                return []
            result = []
            node_stack = []
            while root or node_stack:
                if root:
                    node_stack.append(root)
                    result.append(root.val)
                    root = root.right
                else:
                    node = node_stack.pop()
                    root = node.left
            return result[::-1]

    """
    []
    [1, null, 2, 3]
    [1, null, 2, 3, null, 4, 5]
    """


173. Binary Search tree iterator
------------------------------------

.. code-block:: python

    Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

    Calling next() will return the next smallest number in the BST.

    Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.

    Credits:Special thanks to @ts for adding this problem and creating all test cases.

    =================================================================
    class BSTIterator(object):
      def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.p = None
        self.stack = []
        if root:
          self.stack.append((1, root))

      def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0

      def next(self):
        """
        :rtype: int
        """
        stack = self.stack
        while stack:
          p = stack.pop()
          if not p[1]:
            continue
          if p[0] == 0:
            return p[1].val
          else:
            l = []
            if p[1].right:
              l.append((1, p[1].right))
            l.append((0, p[1]))
            if p[1].left:
              l.append((1, p[1].left))
            stack.extend(l)

    # Your BSTIterator will be called like this:
    # i, v = BSTIterator(root), []
    # while i.hasNext(): v.append(i.next())


    =================================================================
    class BSTIterator(object):
        def __init__(self, root):
            self.root = root
            self.node_stack = []
            self.cur_node = root

        def hasNext(self):
            if self.cur_node or self.node_stack:
                return True
            else:
                return False

        def next(self):
            # inorder traversal
            while self.cur_node:
                self.node_stack.append(self.cur_node)
                self.cur_node = self.cur_node.left

            top = self.node_stack.pop()
            self.cur_node = top.right
            return top.val

    # Your BSTIterator will be called like this:
    # i, v = BSTIterator(root), []
    # while i.hasNext(): v.append(i.next())

    """
    []
    [1]
    [10,8,16,2,9,15,17]
    """


208. Implement trie prefix
---------------------------------

.. code-block:: python

    Implement a trie with insert, search, and startsWith methods.



    Note:
    You may assume that all inputs are consist of lowercase letters a-z.


    =================================================================
    class TrieNode(object):
      def __init__(self):
        """
        Initialize your data structure here.
        """
        self.children = [None] * 26
        self.isWord = False
        self.word = ""


    class Trie(object):

      def __init__(self):
        self.root = TrieNode()

      def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        p = self.root
        for c in word:
          cVal = ord(c) - ord("a")
          if p.children[cVal]:
            p = p.children[cVal]
          else:
            newNode = TrieNode()
            p.children[cVal] = newNode
            p = newNode

        p.isWord = True
        p.word = word

      def helper(self, word):
        p = self.root
        for c in word:
          cVal = ord(c) - ord("a")
          if p.children[cVal]:
            p = p.children[cVal]
          else:
            return None
        return p

      def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.helper(word)
        if p and p.isWord:
          return True
        return False

      def startsWith(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        if self.helper(prefix):
          return True
        return False

    # Your Trie object will be instantiated and called as such:
    # trie = Trie()
    # trie.insert("somestring")
    # trie.search("key")


    =================================================================
    class TrieNode(object):
        def __init__(self):
            self.children = {}
            self.is_word = False


    class Trie(object):
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            # Inserts a word into the trie.
            cur_node = self.root
            for ch in word:
                if ch not in cur_node.children:
                    cur_node.children[ch] = TrieNode()
                cur_node = cur_node.children[ch]
            cur_node.is_word = True

        def search(self, word):
            # Returns if the word is in the trie.
            cur_node = self.root
            for ch in word:
                if ch not in cur_node.children:
                    return False
                cur_node = cur_node.children[ch]
            return cur_node.is_word

        def startsWith(self, prefix):
            # Returns if there is any word in the trie
            # that starts with the given prefix.
            cur_node = self.root
            for ch in prefix:
                if ch not in cur_node.children:
                    return False
                cur_node = cur_node.children[ch]
            return True

    """
    if __name__ == '__main__':
        trie = Trie()
        trie.insert("app")
        trie.insert("apple")
        trie.insert("beer")
        trie.insert("add")
        trie.insert("jam")
        trie.insert("rental")
        print trie.search("apps")
        print trie.search("app")
        print trie.search("ad")
    """


211. Add and Search word data Structure design
------------------------------------------------

.. code-block:: python

    Design a data structure that supports the following two operations:


    void addWord(word)
    bool search(word)



    search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.


    For example:

    addWord("bad")
    addWord("dad")
    addWord("mad")
    search("pad") -> false
    search("bad") -> true
    search(".ad") -> true
    search("b..") -> true



    Note:
    You may assume that all words are consist of lowercase letters a-z.


    click to show hint.

    You should be familiar with how a Trie works. If not, please work on this problem: Implement Trie (Prefix Tree) first.


    =================================================================
    class TrieNode:
      def __init__(self):
        self.neighbours = {}
        self.isWord = False


    class Trie:
      def __init__(self):
        self.root = TrieNode()

      def addWord(self, word):
        root = self.root
        for i in range(0, len(word)):
          c = word[i]
          if c in root.neighbours:
            root = root.neighbours[c]
          else:
            newnode = TrieNode()
            root.neighbours[c] = newnode
            root = root.neighbours[c]
        root.isWord = True


    class WordDictionary:
      def __init__(self):
        self.trie = Trie()
        self.cache = set([])

      def addWord(self, word):
        self.trie.addWord(word)
        self.cache.add(word)

      def search(self, word):
        if word in self.cache:
          return True

        def dfsHelper(root, word, index):
          if not root:
            return False

          if len(word) == index:
            if root.isWord:
              return True
            return False

          if word[index] != ".":
            if dfsHelper(root.neighbours.get(word[index], None), word, index + 1):
              return True
          else:
            for nbr in root.neighbours:
              if dfsHelper(root.neighbours[nbr], word, index + 1):
                return True
          return False

        return dfsHelper(self.trie.root, word, 0)


    =================================================================
    import collections


    class WordDictionary(object):
        # One faster, easy understand way
        # Refer to:
        # https://leetcode.com/discuss/69963/python-168ms-beat-100%25-solution
        def __init__(self):
            self.words_dict = collections.defaultdict(list)

        def addWord(self, word):
            if word:
                self.words_dict[len(word)].append(word)

        def search(self, word):
            """
            Returns if the word is in the data structure. A word could
            contain the dot character '.' to represent any one letter.
            """
            if not word:
                return False
            for w in self.words_dict[len(word)]:
                is_match = True
                for i, ch in enumerate(word):
                    if ch != "." and ch != w[i]:
                        is_match = False
                        break
                if is_match:
                    return True
            return False


    class TrieNode():
        # Refer to: 208. Implement Trie
        def __init__(self):
            self.is_word = False
            self.childrens = {}


    class WordDictionary_Trie(object):
        def __init__(self):
            self.root = TrieNode()

        def addWord(self, word):
            """
            Adds a word into the data structure.
            """
            cur_node = self.root
            for ch in word:
                if ch not in cur_node.childrens:
                    cur_node.childrens[ch] = TrieNode()
                cur_node = cur_node.childrens[ch]
            cur_node.is_word = True

        def search(self, word):
            """
            Returns if the word is in the data structure. A word could
            contain the dot character '.' to represent any one letter.
            """
            return self._dfs_searh(word, self.root)

        # Depth First Search the trie tree.
        def _dfs_searh(self, word, cur_node):
            if not word and cur_node.is_word:
                return True
            word_len = len(word)
            for i in range(word_len):
                ch = word[i]
                if ch == ".":
                    for child_ch in cur_node.childrens:
                        if self._dfs_searh(word[i+1:],
                                           cur_node.childrens[child_ch]):
                            return True
                    return False
                else:
                    if ch not in cur_node.childrens:
                        return False
                    else:
                        cur_node = cur_node.childrens[ch]
            if cur_node.is_word:
                return True
            else:
                return False

    """
    if __name__ == '__main__':
        wordDictionary = WordDictionary()
        wordDictionary.addWord("bad")
        wordDictionary.addWord("dad")
        wordDictionary.addWord("mad")
        print wordDictionary.search("xad")
        print wordDictionary.search(".a")
        print wordDictionary.search(".ad")
        print wordDictionary.search("b.")
        print wordDictionary.search(".")
    """


226. Invert Binary tree
-------------------------------

.. code-block:: python

    Invert a binary tree.
         4
       /   \
      2     7
     / \   / \
    1   3 6   9

    to
         4
       /   \
      7     2
     / \   / \
    9   6 3   1

    Trivia:
    This problem was inspired by this original tweet by Max Howell:
    Google: 90% of our engineers use the software you wrote (Homebrew), but you can�셳 invert a binary tree on a whiteboard so fuck off.

    =================================================================
    class Solution(object):
      def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
          return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


    =================================================================
    class Solution(object):
        def invertTree(self, root):
            if not root:
                return None
            root.left, root.right = root.right, root.left
            self.invertTree(root.left)
            self.invertTree(root.right)
            return root

    """
    []
    [1,2,3,4,5,6]
    """


235. Lowest common ancestor of a binary search
------------------------------------------------

.. code-block:: python

    Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.



    According to the definition of LCA on Wikipedia: �쏷he lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).��



            _______6______
           /              \
        ___2__          ___8__
       /      \        /      \
       0      _4       7       9
             /  \
             3   5



    For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

    =================================================================
    class Solution(object):
      def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        a, b = sorted([p.val, q.val])
        while not a <= root.val <= b:
          if a > root.val:
            root = root.right
          else:
            root = root.left
        return root

    =================================================================
    class Solution(object):
        # Easy to understand
        def lowestCommonAncestor(self, root, p, q):
            min_val = min(p.val, q.val)
            max_val = max(p.val, q.val)
            while root:
                value = root.val
                if min_val <= value <= max_val:
                    return root
                elif max_val < value:
                    root = root.left
                else:
                    root = root.right
            return None


    class Solution_2(object):
        """
        One elegant code, some puzzling but short code. according to:
        https://leetcode.com/discuss/44959/3-lines-with-o-1-space-1-liners-alternatives
        Just walk down from the whole tree's root as long as
        both p and q are in the same subtree
        """
        def lowestCommonAncestor(self, root, p, q):
            while (root.val - p.val) * (root.val - q.val) > 0:
                root = (root.left, root.right)[p.val > root.val]
            return root




236. Lowest common ancestor of a binary tree
--------------------

.. code-block:: python

    Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.



    According to the definition of LCA on Wikipedia: �쏷he lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).��



            _______3______
           /              \
        ___5__          ___1__
       /      \        /      \
       6      _2       0       8
             /  \
             7   4



    For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

    =================================================================
    class Solution(object):
      def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
          return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
          return root

        if root == p or root == q:
          return root

        if left:
          return left
        if right:
          return right
        return None




    =================================================================
    class Solution(object):
        """
        Recursive method: DFS.
        If the current (sub)tree contains both p and q, then the function result is their LCA.
        If only one of them is in that subtree, then the result is that one of them.
        If neither are in that subtree, the result is null/None/nil.

        More version can be found here:
        https://discuss.leetcode.com/topic/18561/4-lines-c-java-python-ruby
        """
        def lowestCommonAncestor(self, root, p, q):
            if not root or root == p or root == q:
                return root
            left = self.lowestCommonAncestor(root.left, p, q)
            right = self.lowestCommonAncestor(root.right, p, q)
            # if p and q are on both sides
            if left and right:
                return root
            else:
                return left or right


    class Solution_2(object):
        """
        Iterative method: BFS(DFS is ok too).  According to:
        https://leetcode.com/discuss/64764/java-python-iterative-solution
        """
        def lowestCommonAncestor(self, root, p, q):
            node_stack = [root]
            parent_record = {root: None}

            # Build the relationship from child to parent
            while p not in parent_record or q not in parent_record:
                node = node_stack.pop()
                if node.left:
                    node_stack.append(node.left)
                    parent_record[node.left] = node
                if node.right:
                    node_stack.append(node.right)
                    parent_record[node.right] = node

            # Trace brack from one node, record the path.
            # Then trace from the other.
            ancestors = set()
            while p:
                ancestors.add(p)
                p = parent_record[p]

            while q not in ancestors:
                q = parent_record[q]
            return q



297. Serialize and deserialize binary tree
------------------------------------------------

.. code-block:: python

    Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

    Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.


    For example, you may serialize the following tree

        1
       / \
      2   3
         / \
        4   5

    as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.



    Note: Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.


    Credits:Special thanks to @Louis1992 for adding this problem and creating all test cases.

    =================================================================
    from collections import deque


    class Codec:

      def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        ret = []
        queue = deque([root])
        while queue:
          top = queue.popleft()
          if not top:
            ret.append("None")
            continue
          else:
            ret.append(str(top.val))
          queue.append(top.left)
          queue.append(top.right)
        return ",".join(ret)

      def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        left = lambda n: 2 * n + 1
        right = lambda n: 2 * n + 2
        data = data.split(",")
        if data[0] == "None":
          return None
        root = TreeNode(int(data[0]))
        queue = deque([root])
        i = 0
        while queue and i < len(data):
          top = queue.popleft()
          i += 1
          left = right = None
          if i < len(data) and data[i] != "None":
            left = TreeNode(int(data[i]))
            queue.append(left)
          i += 1
          if i < len(data) and data[i] != "None":
            right = TreeNode(int(data[i]))
            queue.append(right)

          top.left = left
          top.right = right

        return root

    # Your Codec object will be instantiated and called as such:
    # codec = Codec()
    # codec.deserialize(codec.serialize(root))


    =================================================================
    # The leetcode way
    class Codec:
        def serialize(self, root):
            data = []
            node_queue = [root]
            start = 0
            while start < len(node_queue):
                node = node_queue[start]
                start += 1
                if node:
                    data.append(str(node.val))
                    node_queue.append(node.left)
                    node_queue.append(node.right)
                else:
                    data.append("null")
            # Remove the tail null node.
            while data and data[-1] == "null":
                del data[-1]
            return ",".join(data)

        def deserialize(self, data):
            if not data:
                return None

            # Get all the nodes.
            data_list = data.split(",")
            length = len(data_list)
            node_list = [0] * length
            for i in range(length):
                if data_list[i] == "null":
                    node_list[i] = None
                else:
                    node_list[i] = TreeNode(int(data_list[i]))

            # Build the tree.
            offset = 1
            cur_pos = 0
            while offset < length:
                if node_list[cur_pos]:
                    node_list[cur_pos].left = node_list[offset]
                    offset += 1
                    if offset < length:
                        node_list[cur_pos].right = node_list[offset]
                        offset += 1
                    else:
                        break
                else:
                    pass
                cur_pos += 1

            return node_list[0]


    class Codec_2:
        # Refer to: Recursive preorder, Python and C++, O(n)
        # https://leetcode.com/discuss/66147/recursive-preorder-python-and-c-o-n
        def serialize(self, root):
            def helper(node):
                if node:
                    vals.append(str(node.val))
                    helper(node.left)
                    helper(node.right)
                else:
                    vals.append('#')

            vals = []
            helper(root)
            return ' '.join(vals)

        def deserialize(self, data):
            def helper():
                val = next(vals)
                if val == '#':
                    return None
                node = TreeNode(int(val))
                node.left = helper()
                node.right = helper()
                return node

            vals = iter(data.split())
            return helper()

    # Your Codec object will be instantiated and called as such:
    # codec = Codec()
    # codec.deserialize(codec.serialize(root))(codec.deserialize("1,null,3,4,5"))

    """
    []
    [1,2,null,3,4]
    [1,2,3,null,4,null,5,null,6,7]
    """




331. Verify preorder serialization of a binary tree
------------------------------------------------------

.. code-block:: python

    One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as #.


         _9_
        /   \
       3     2
      / \   / \
     4   1  #  6
    / \ / \   / \
    # # # #   # #


    For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # represents a null node.


    Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.

    Each comma separated value in the string must be either an integer or a character '#' representing null pointer.

    You may assume that the input format is always valid, for example it could never contain two consecutive commas such as "1,,3".

    Example 1:
    "9,3,4,#,#,1,#,#,2,#,6,#,#"
    Return true
    Example 2:
    "1,#"
    Return false
    Example 3:
    "9,#,#,1"
    Return false

    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        p = preorder.split(",")
        if len(p) == 1:
          if p[0] == "#":
            return True
          return False
        stack = [p[0]]
        for c in p[1:]:
          if len(stack) == 1 and stack[0] == "#":
            return False
          stack.append(c)
          while len(stack) > 2 and stack[-1] + stack[-2] == "##":
            stack.pop()
            stack.pop()
            stack.pop()
            stack.append("#")
        if len(stack) == 1 and stack[0] == "#":
          return True
        return False




    =================================================================
    class Solution(object):
        def isValidSerialization(self, preorder):
            """
            When there are two consecutive "#" characters on top of the stack,
            pop both of them and replace the top element on the remain stack with "#".
            """
            preorder = preorder.split(",")
            stack = []
            for val in preorder:
                stack.append(val)
                while self.twoConsecutive(stack):
                    stack = stack[:-3]
                    stack.append("#")

            return stack == ["#"]

        def twoConsecutive(self, stack):
            if len(stack) < 3:
                return False
            return stack[-1] == stack[-2] == "#" and stack[-3] != "#"


    class Solution_2(object):
        """
        Refer to:
        https://leetcode.com/discuss/83824/7-lines-easy-java-solution
        In a binary tree, if we consider null as leaves, then
            1. all non-null node provides 2 outdegree and 1 indegree(except root).
            2. all null node provides 0 outdegree and 1 indegree.

        Record diff = outdegree - indegree. When the next node comes:
        Decrease diff by 1, because the node provides an indegree.
        If the node is not null, increase diff by 2, because it provides two out degrees.

        diff should never be negative and diff will be zero when finished.
        """
        def isValidSerialization(self, preorder):
            preorder = preorder.split(",")
            diff = 1
            for val in preorder:
                diff -= 1
                if diff < 0:
                    return False
                if val != "#":
                    diff += 2
            return diff == 0

    """
    ""
    "#,#"
    "1,#"
    "1,#,#"
    "#,#,#"
    "1,#,#,#,#"
    "9,#,#,1"
    "9,3,4,#,#,1,#,#,2,#,6,#,#"
    """



