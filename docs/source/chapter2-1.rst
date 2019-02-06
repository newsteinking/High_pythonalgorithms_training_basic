Backtracking - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode



22. generate-parentheses
--------------------------------------

.. code-block:: python

    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.



    For example, given n = 3, a solution set is:


    [
      "((()))",
      "(()())",
      "(())()",
      "()(())",
      "()()()"
    ]


    class Solution(object):
      def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        def dfs(left, path, res, n):
          if len(path) == 2 * n:
            if left == 0:
              res.append("".join(path))
            return

          if left < n:
            path.append("(")
            dfs(left + 1, path, res, n)
            path.pop()
          if left > 0:
            path.append(")")
            dfs(left - 1, path, res, n)
            path.pop()

        res = []
        dfs(0, [], res, n)
        return res

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        # node = [leftchild, rightchild, count"(", count")", current str]
        if not n:
            return [""]
        solution = []
        root = [None, None, 1, 0, "("]
        self.generate_child_tree(root, n, solution)
        return solution


    def generate_child_tree(self, node, n, solution):
        # the node is the leave and the str is what we want
        if node[2] == node[3] == n:
            node[0] = None
            node[1] = None
            solution.append(node[4])

        # the node have both left and right child
        elif node[2] > node[3] and node[2] < n:
            left_child = [None, None, node[2] + 1, node[3], node[4] + "("]
            right_child = [None, None, node[2], node[3] + 1, node[4] + ")"]
            node[0] = left_child
            node[1] = right_child
            self.generate_child_tree(left_child, n, solution)
            self.generate_child_tree(right_child, n, solution)

        # the node have only left child
        elif node[2] == node[3] and node[2] < n:
            left_child = [None, None, node[2] + 1, node[3], node[4] + "("]
            node[0] = left_child
            self.generate_child_tree(left_child, n, solution)

        # the node have only left child
        else:
            right_child = [None, None, node[2], node[3] + 1, node[4] + ")"]
            node[1] = right_child
            self.generate_child_tree(right_child, n, solution)

    """
    0
    1
    3
    5
    """


131. Palindrome-partitioning
--------------------------------------

.. code-block:: python






131. Palindrome-partitioning
--------------------------------------

.. code-block:: python

    Given a string s, partition s such that every substring of the partition is a palindrome.


    Return all possible palindrome partitioning of s.


    For example, given s = "aab",

    Return

    [
      ["aa","b"],
      ["a","a","b"]
    ]

    class Solution(object):
      def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        pal = [[False for i in range(0, len(s))] for j in range(0, len(s))]
        ans = [[[]]] + [[] for _ in range(len(s))]

        for i in range(0, len(s)):
          for j in range(0, i + 1):
            if (s[j] == s[i]) and ((j + 1 > i - 1) or (pal[j + 1][i - 1])):
              pal[j][i] = True
              for res in ans[j]:
                a = res + [s[j:i + 1]]
                ans[i + 1].append(a)
        return ans[-1]

    class Solution(object):
        def partition(self, s):
            if not s:
                return []
            self.result = []
            self.end = len(s)
            self.str = s

            self.is_palindrome = [[False for i in range(self.end)]
                                  for j in range(self.end)]

            for i in range(self.end-1, -1, -1):
                for j in range(self.end):
                    if i > j:
                        pass
                    elif j-i < 2 and s[i] == s[j]:
                        self.is_palindrome[i][j] = True
                    elif self.is_palindrome[i+1][j-1] and s[i] == s[j]:
                        self.is_palindrome[i][j] = True
                    else:
                        self.is_palindrome[i][j] = False

            self.palindrome_partition(0, [])
            return self.result

        def palindrome_partition(self, start, sub_strs):
            if start == self.end:
                # It's confused the following sentence doesn't work.
                # self.result.append(sub_strs)
                self.result.append(sub_strs[:])
                return

            for i in range(start, self.end):
                if self.is_palindrome[start][i]:
                    sub_strs.append(self.str[start:i+1])
                    self.palindrome_partition(i+1, sub_strs)
                    sub_strs.pop()      # Backtracking here


    if __name__ == "__main__":
        sol = Solution()
        print sol.partition("aab")
        print sol.partition("aabb")
        print sol.partition("aabaa")
        print sol.partition("acbca")
        print sol.partition("acbbca")


131. Palindrome-partitioning
--------------------------------------

.. code-block:: python



131. Palindrome-partitioning
--------------------------------------

.. code-block:: python




131. Palindrome-partitioning
--------------------------------------

.. code-block:: python



131. Palindrome-partitioning
--------------------------------------

.. code-block:: python



131. Palindrome-partitioning
--------------------------------------

.. code-block:: python



131. Palindrome-partitioning
--------------------------------------

.. code-block:: python


131. Palindrome-partitioning
--------------------------------------

.. code-block:: python



131. Palindrome-partitioning
--------------------------------------

.. code-block:: python


131. Palindrome-partitioning
--------------------------------------

.. code-block:: python