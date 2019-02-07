Recursion - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

77. Combinations
--------------------

.. code-block:: python

    Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.


    For example,
    If n = 4 and k = 2, a solution is:



    [
      [2,4],
      [3,4],
      [2,3],
      [1,2],
      [1,3],
      [1,4],
    ]


    =================================================================
    class Solution(object):
      def combine(self, n, k):
        if k == 1:
          return [[i] for i in range(1, n + 1)]
        elif k == n:
          return [[i for i in range(1, n + 1)]]
        else:
          rs = []
          rs += self.combine(n - 1, k)
          part = self.combine(n - 1, k - 1)
          for ls in part:
            ls.append(n)
          rs += part
          return rs


    =================================================================
    class Solution(object):
        def combine(self, n, k):
            """
            :type n: int
            :type k: int
            :rtype: List[List[int]]
            """
            if k > n or not n:
                return []

            combine_list = self.combine_k(1, n, k)
            return combine_list

        def combine_k(self, start, n, k):
            combine_k = []
            # k == 1, just return the list[start, end]
            if k == 1:
                for i in range(start, n+1):
                    combine_k.append([i])
                return combine_k

            # k > 1, return every i combines all the k-1 th combine in [i+1, n]
            for i in range(start, n+2-k):
                combine_k_1 = self.combine_k(i+1, n, k-1)
                for combine_1 in combine_k_1:
                    combine = [i]
                    combine.extend(combine_1)
                    combine_k.append(combine)

            return combine_k

    """
    5
    2
    2
    3
    6
    6
    """



89. Gray Code
--------------------

.. code-block:: python

    The gray code is a binary numeral system where two successive values differ in only one bit.

    Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.

    For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:

    00 - 0
    01 - 1
    11 - 3
    10 - 2


    Note:
    For a given n, a gray code sequence is not uniquely defined.

    For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.

    For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.

    =================================================================
    class Solution(object):
      def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if n < 1:
          return [0]
        ans = [0] * (2 ** n)
        ans[1] = 1
        mask = 0x01
        i = 1
        while i < n:
          mask <<= 1
          for j in range(0, 2 ** i):
            root = (2 ** i)
            ans[root + j] = ans[root - j - 1] | mask
          i += 1
        return ans


    =================================================================
    class Solution(object):
        def grayCode(self, n):
            """
            :type n: int
            :rtype: List[int]
            """
            if not n:
                return [0]

            if n == 1:
                return [0, 1]

            # Consume n's sequence is: 0..0, 0..1, ..., 1..0
            # When comes to n+1, it's sequence is simple as followers:
            # 0{0..0, 0..1, ..., 1..0}, 1{1..0, ..., 0..1, 0..0}
            # Then second part of past line is just a reverse of n's sequence.
            high_digit = 2 ** (n-1)
            gray_code_list = self.grayCode(n-1)
            for num in gray_code_list[::-1]:
                gray_code_list.append(high_digit + num)

            return gray_code_list

    """
    0
    2
    3
    4
    """



101. Symmetric Tree
--------------------

.. code-block:: python

    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).


    For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

        1
       / \
      2   2
     / \ / \
    3  4 4  3



    But the following [1,2,2,null,3,null,3]  is not:

        1
       / \
      2   2
       \   \
       3    3




    Note:
    Bonus points if you could solve it both recursively and iteratively.


    =================================================================
    class Solution(object):
      def isSymmetric(self, node):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def helper(root, mirror):
          if not root and not mirror:
            return True
          if root and mirror and root.val == mirror.val:
            return helper(root.left, mirror.right) and helper(root.right, mirror.left)
          return False

        return helper(node, node)

    =================================================================
    class Solution(object):
        def isSymmetric(self, root):
            return self.helper(root, root)

        # If two nodes are symmetric
        def helper(self, lNode, rNode):
            if not lNode or not rNode:
                return lNode == rNode
            if lNode.val != rNode.val:
                return False
            return (self.helper(lNode.left, rNode.right) and
                    self.helper(lNode.right, rNode.left))

    """
    []
    [1]
    [1,2,3]
    [1,2,2,3,4,4,3]
    [1,2,2,null,3,null,3]
    """
