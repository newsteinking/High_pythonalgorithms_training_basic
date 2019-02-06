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

    =================================================================

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
    =================================================================
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


39. Combination Sum
--------------------------------------

.. code-block:: python

    Given a set of candidate numbers (C) (without duplicates) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.


    The same repeated number may be chosen from C unlimited number of times.


    Note:

    All numbers (including target) will be positive integers.
    The solution set must not contain duplicate combinations.




    For example, given candidate set [2, 3, 6, 7] and target 7,
    A solution set is:

    [
      [7],
      [2, 2, 3]
    ]
    =================================================================
    class Solution(object):
      def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(candidates, start, target, path, res):
          if target == 0:
            return res.append(path + [])

          for i in range(start, len(candidates)):
            if target - candidates[i] >= 0:
              path.append(candidates[i])
              dfs(candidates, i, target - candidates[i], path, res)
              path.pop()

        res = []
        dfs(candidates, 0, target, [], res)
        return res
    ===================================================================
    class Solution(object):
        """ Classic backtracking problem.

        One key point: for one specified number,
        just scan itself and numbers larger than it to avoid duplicate combinations.
        Besides, the current path need to be reset after dfs call in general.
        Here we can just use `path + [num]` to avoid modifying path, so no need to reset.
        Refer to:
        https://discuss.leetcode.com/topic/23142/python-dfs-solution
        """
        def combinationSum(self, candidates, target):
            if not candidates:
                return []

            ans = []
            candidates.sort()
            self.dfs_search(candidates, 0, target, [], ans)
            return ans

        def dfs_search(self, candidates, start, target, path, ans):
            if target == 0:
                ans.append(path)
            else:
                for i in xrange(start, len(candidates)):
                    # Cannot find the suitable sets, just return.
                    num = candidates[i]
                    if num > target:
                        return
                    self.dfs_search(candidates, i, target - num, path + [num], ans)

    """
    []
    2
    [2, 3, 6, 7]
    7
    [1, 2, 3, 4]
    10
    """


40. Combination Sum 2
--------------------------------------

.. code-block:: python

    Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.


    Each number in C may only be used once in the combination.

    Note:

    All numbers (including target) will be positive integers.
    The solution set must not contain duplicate combinations.




    For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8,
    A solution set is:

    [
      [1, 7],
      [1, 2, 5],
      [2, 6],
      [1, 1, 6]
    ]
    ===================================================================
    class Solution(object):
      def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(nums, target, start, visited, path, res):
          if target == 0:
            res.append(path + [])
            return

          for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
              continue
            if target - nums[i] < 0:
              return 0
            if i not in visited:
              visited.add(i)
              path.append(nums[i])
              dfs(nums, target - nums[i], i + 1, visited, path, res)
              path.pop()
              visited.discard(i)

        candidates.sort()
        res = []
        visited = set([])
        dfs(candidates, target, 0, visited, [], res)
        return res

    ===================================================================
    class Solution(object):
        """ Classic backtracking problem.

        One key point: for one specified number,
        just scan the number larger than it to avoid duplicate combinations.
        Besides, the current path need to be reset after dfs call in general.
        Here we can just use `path + [num]` to avoid modifying path, so no need to reset.
        """

        def combinationSum2(self, candidates, target):
            if not candidates:
                return []
            candidates.sort()
            ans = []
            self.dfs_search(candidates, 0, target, [], ans)
            return ans

        def dfs_search(self, candidates, start, target, path, ans):
            if target == 0:
                ans.append(path)
            for i in xrange(start, len(candidates)):
                num = candidates[i]
                if num > target:
                    return
                # Here skip the same `adjacent` element to avoid duplicated.
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                self.dfs_search(candidates, i + 1,
                                target - num, path + [num], ans)

    """
    []
    1
    [2, 5, 1, 4, 9]
    11
    [10, 1, 2, 7, 6, 1, 5]
    8
    """







46. Permutations
--------------------------------------

.. code-block:: python

    Given a collection of distinct numbers, return all possible permutations.



    For example,
    [1,2,3] have the following permutations:

    [
      [1,2,3],
      [1,3,2],
      [2,1,3],
      [2,3,1],
      [3,1,2],
      [3,2,1]
    ]

    ===================================================================
    class Solution(object):
      def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        visited = set([])

        def dfs(nums, path, res, visited):
          if len(path) == len(nums):
            res.append(path + [])
            return

          for i in range(0, len(nums)):
            # if i > 0 and nums[i - 1] == nums[i]:
            #     continue
            if i not in visited:
              visited.add(i)
              path.append(nums[i])
              dfs(nums, path, res, visited)
              path.pop()
              visited.discard(i)

        dfs(nums, [], res, visited)
        return res


    ===================================================================
    class Solution(object):
        # Easy to understand: recursively.
        def permute(self, nums):
            ans = []
            self.dfs(nums, [], ans)
            return ans

        def dfs(self, nums, path, ans):
            if not nums:
                ans.append(path)
            for i, n in enumerate(nums):
                self.dfs(nums[:i] + nums[i + 1:], path + [n], ans)


    class Solution_2(object):
        # Pythonic way.  recursively.
        # According to: https://leetcode.com/discuss/42550/one-liners-in-python
        def permute(self, nums):
            return [[n] + p
                    for i, n in enumerate(nums)
                    for p in self.permute(nums[:i] + nums[i + 1:])] or [[]]

    """
    []
    [1]
    [1,2,3]
    """

47. Permutation 2
--------------------------------------

.. code-block:: python

    Given a collection of numbers that might contain duplicates, return all possible unique permutations.



    For example,
    [1,1,2] have the following unique permutations:

    [
      [1,1,2],
      [1,2,1],
      [2,1,1]
    ]

    ===================================================================
    class Solution(object):
      def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()

        def dfs(nums, res, path, visited):
          if len(path) == len(nums):
            res.append(path + [])
            return

          for i in range(len(nums)):
            if i in visited:
              continue
            if i > 0 and nums[i] == nums[i - 1] and i - 1 not in visited:
              continue
            visited |= {i}
            path.append(nums[i])
            dfs(nums, res, path, visited)
            path.pop()
            visited -= {i}

        dfs(nums, res, [], set())
        return res


    ===================================================================
    class Solution(object):
        # Easy to understand: recursively.
        # Just like get permute for distinct numbers.
        def permuteUnique(self, nums):
            ans = []
            nums.sort()
            self.dfs(nums, 0, ans)
            return ans

        def dfs(self, num, begin, ans):
            if begin == len(num) - 1:
                ans.append(num)
                return

            for i in range(begin, len(num)):
                if i != begin and num[i] == num[begin]:
                    continue
                num[i], num[begin] = num[begin], num[i]
                # num[:], get a new copy.  Just like pass by value
                self.dfs(num[:], begin + 1, ans)


    class Solution_2(object):
        '''
        1. sort nums in ascending order, add it to res;
        2. generate the next permutation of nums, and add it to res;
        3. repeat 2 until the next permutation of nums.
        '''
        def permuteUnique(self, nums):
            nums.sort()
            ans = []
            ans.append(nums[:])
            while self.nextPermutation(nums):
                ans.append(nums[:])

            return ans

        def nextPermutation(self, nums):
            length = len(nums)
            index = length - 1

            while index >= 1:
                if nums[index] > nums[index - 1]:
                    for i in range(length - 1, index - 1, -1):
                        if nums[i] > nums[index - 1]:
                            nums[i], nums[index - 1] = nums[index - 1], nums[i]
                            nums[index:] = sorted(nums[index:])
                            return True
                else:
                    index -= 1

            # Nums is in descending order, just reverse it.
            return False


    """
    []
    [1]
    [1,2,3]
    [2,2,3,3]
    """



51. NQueens
--------------------------------------

.. code-block:: python

    The n-queens puzzle is the problem of placing n queens on an n횞n chessboard such that no two queens attack each other.



    Given an integer n, return all distinct solutions to the n-queens puzzle.

    Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.

    For example,
    There exist two distinct solutions to the 4-queens puzzle:

    [
     [".Q..",  // Solution 1
      "...Q",
      "Q...",
      "..Q."],

     ["..Q.",  // Solution 2
      "Q...",
      "...Q",
      ".Q.."]
    ]

    ===================================================================
    class Solution(object):
      def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        ans = []

        def dfs(path, n, ans):
          if len(path) == n:
            ans.append(drawChess(path))
            return

          for i in range(n):
            if i not in path and isValidQueen(path, i):
              path.append(i)
              dfs(path, n, ans)
              path.pop()

        def isValidQueen(path, k):
          for i in range(len(path)):
            if abs(k - path[i]) == abs(len(path) - i):
              return False
          return True

        def drawChess(path):
          ret = []
          chess = [["."] * len(path) for _ in range(len(path))]
          for i in range(0, len(path)):
            chess[i][path[i]] = "Q"
          for chs in chess:
            ret.append("".join(chs))
          return ret

        dfs([], n, ans)
        return ans


    ===================================================================
    class Solution(object):
        allNQueens = []

        def solveNQueens(self, n):
            self.allNQueens = []
            self.cols = [True] * n
            self.left_right = [True] * (2 * n - 1)
            self.right_left = [True] * (2 * n - 1)
            queueMatrix = [["."] * n for row in range(n)]
            self.solve(0, queueMatrix, n)

            return self.allNQueens

        def solve(self, row, matrix, n):
            """
            Refer to:
            https://discuss.leetcode.com/topic/13617/accepted-4ms-c-solution-use-backtracking-and-bitmask-easy-understand
            The number of columns is n, the number of 45° diagonals is 2 * n - 1,
            the number of 135° diagonals is also 2 * n - 1.
            When reach [row, col], the column No. is col,
            the 45° diagonal No. is row + col and the 135° diagonal No. is n - 1 + col - row.

            | | |                / / /             \ \ \
            O O O               O O O               O O O
            | | |              / / / /             \ \ \ \
            O O O               O O O               O O O
            | | |              / / / /             \ \ \ \
            O O O               O O O               O O O
            | | |              / / /                 \ \ \
            3 columns        5 45° diagonals     5 135° diagonals    (when n is 3)
            """

            # Get one Queen Square
            if row == n:
                result = ["".join(r) for r in matrix]
                self.allNQueens.append(result)
                return

            for col in range(n):
                if self.cols[col] and self.left_right[row + n - 1 - col] and self.right_left[row + col]:
                    matrix[row][col] = "Q"
                    self.cols[col] = self.left_right[row + n - 1 - col] = self.right_left[row + col] = False
                    # Solve the child question
                    self.solve(row + 1, matrix, n)
                    # Backtracking here.
                    matrix[row][col] = "."
                    self.cols[col] = self.left_right[
                        row + n - 1 - col] = self.right_left[row + col] = True

    """
    1
    5
    8
    """


52. N-Queens 2
--------------------------------------

.. code-block:: python

    Follow up for N-Queens problem.

    Now, instead outputting board configurations, return the total number of distinct solutions.

    ===================================================================
    class Solution(object):
      def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """

        def dfs(path, n):
          if len(path) == n:
            return 1
          res = 0
          for i in range(n):
            if i not in path and isValidQueen(path, i):
              path.append(i)
              res += dfs(path, n)
              path.pop()
          return res

        def isValidQueen(path, k):
          for i in range(len(path)):
            if abs(k - path[i]) == abs(len(path) - i):
              return False
          return True

        return dfs([], n)

    ===================================================================
    class Solution(object):
        countNQueens = 0

        def totalNQueens(self, n):
            self.countNQueens = 0
            cols_used = [-1 for i in range(n)]
            self.solveNQueens(0, cols_used, n)
            return self.countNQueens

        def solveNQueens(self, row, cols_used, n):
            for col in range(n):
                if self.isValid(row, col, cols_used, n):
                    if row == n - 1:
                        self.countNQueens += 1
                        return

                    cols_used[row] = col
                    self.solveNQueens(row + 1, cols_used, n)
                    cols_used[row] = -1

        def isValid(self, row, col, cols_used, n):
            """ Can check isvalid with using hash, implemented by c++.

            Refer to:
            https://discuss.leetcode.com/topic/13617/accepted-4ms-c-solution-use-backtracking-and-bitmask-easy-understand
            The number of columns is n, the number of 45° diagonals is 2 * n - 1,
            the number of 135° diagonals is also 2 * n - 1.
            When reach [row, col], the column No. is col,
            the 45° diagonal No. is row + col and the 135° diagonal No. is n - 1 + col - row.

            | | |                / / /             \ \ \
            O O O               O O O               O O O
            | | |              / / / /             \ \ \ \
            O O O               O O O               O O O
            | | |              / / / /             \ \ \ \
            O O O               O O O               O O O
            | | |              / / /                 \ \ \
            3 columns        5 45° diagonals     5 135° diagonals    (when n is 3)
            """
            for i in range(row):
                # Check for the according col above the current row.
                if cols_used[i] == col:
                    return False

                # Check from left-top to right-bottom
                if cols_used[i] == col - row + i:
                    return False

                # Check from right-top to left-bottom
                if cols_used[i] == col + row - i:
                    return False
            return True

    """
    1
    5
    8
    """


79. Word Search
--------------------------------------

.. code-block:: python

    Given a 2D board and a word, find if the word exists in the grid.


    The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.



    For example,
    Given board =

    [
      ['A','B','C','E'],
      ['S','F','C','S'],
      ['A','D','E','E']
    ]


    word = "ABCCED", -> returns true,
    word = "SEE", -> returns true,
    word = "ABCB", -> returns false.


    ===================================================================
    class Solution:
      # @param board, a list of lists of 1 length string
      # @param word, a string
      # @return a boolean
      def exist(self, board, word):
        # write your code here
        if word == "":
          return True
        if len(board) == 0:
          return False
        visited = [[0] * len(board[0]) for i in range(0, len(board))]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def dfs(i, j, board, visited, word, index):
          if word[index] != board[i][j]:
            return False
          if len(word) - 1 == index:
            return True
          for direction in directions:
            ni, nj = i + direction[0], j + direction[1]
            if ni >= 0 and ni < len(board) and nj >= 0 and nj < len(board[0]):
              if visited[ni][nj] == 0:
                visited[ni][nj] = 1
                if dfs(ni, nj, board, visited, word, index + 1):
                  return True
                visited[ni][nj] = 0
          return False

        for i in range(0, len(board)):
          for j in range(0, len(board[0])):
            visited[i][j] = 1
            if dfs(i, j, board, visited, word, 0):
              return True
            visited[i][j] = 0
        return False


    ===================================================================
    class Solution(object):
        def exist(self, board, word):
            if not board and word:
                return False
            if not word:
                return True

            m_rows = len(board)
            n_cols = len(board[0])
            for row in range(m_rows):
                for col in range(n_cols):
                    if board[row][col] == word[0]:
                        board[row][col] = "*"
                        if (self.exist_adjacent(
                                [row, col],
                                word[1:],
                                board)):
                            return True
                        # Backtracking here
                        board[row][col] = word[0]
            return False

        def exist_adjacent(self, cur_pos, next_str, board):
            # Find all the characters in word.
            if not next_str:
                return True

            adj_pos = self.adj_pos_lists(cur_pos, board)
            # No adjancent position can be used.
            if not adj_pos:
                return False

            # For every adjacent position, find out whether it contains
            # the first character in the word or not.
            # If matches, then resursively check the other characters in word.
            for pos in adj_pos:
                row = pos[0]
                col = pos[1]
                if board[row][col] == next_str[0]:
                    board[row][col] = "*"
                    if (self.exist_adjacent(
                            [row, col],
                            next_str[1:],
                            board)):
                        return True
                    # Backtracking here
                    board[row][col] = next_str[0]

            return False

        # Find the adjacent position around cur_pos
        def adj_pos_lists(self, cur_pos, board):
            m_rows = len(board)
            n_cols = len(board[0])
            row = cur_pos[0]
            col = cur_pos[1]
            adj_list = []
            if row - 1 >= 0:
                adj_list.append([row - 1, col])
            if row + 1 < m_rows:
                adj_list.append([row + 1, col])
            if col - 1 >= 0:
                adj_list.append([row, col - 1])
            if col + 1 < n_cols:
                adj_list.append([row, col + 1])
            return adj_list

    """
    []
    ""
    []
    "as"
    ["abce","sfcs", "adee"]
    "abcced"
    ["abce","sfcs", "adee"]
    "abcb"
    ["ABCE","SFES","ADEE"]
    "ABCESEEEFSAD"
    """



90. Subsets 2
--------------------------------------

.. code-block:: python


    Given a collection of integers that might contain duplicates, nums, return all possible subsets.

    Note: The solution set must not contain duplicate subsets.


    For example,
    If nums = [1,2,2], a solution is:



    [
      [2],
      [1],
      [1,2,2],
      [2,2],
      [1,2],
      []
    ]

    ===================================================================
    class Solution(object):
      def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def dfs(start, nums, path, res, visited):
          res.append(path + [])

          for i in range(start, len(nums)):
            if start != i and nums[i] == nums[i - 1]:
              continue
            if i not in visited:
              visited[i] = 1
              path.append(nums[i])
              dfs(i + 1, nums, path, res, visited)
              path.pop()
              del visited[i]

        nums.sort()
        res = []
        visited = {}
        dfs(0, nums, [], res, visited)
        return res

    ===================================================================
    class Solution(object):
        def subsetsWithDup(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """

            if not nums:
                return []

            nums.sort()
            nums_len = len(nums)

            # Keep the subsets without duplicate subsets
            subsets = [[nums[0]]]
            # Keep the previous subsets which contains previous nums.
            pre_subset = [[nums[0]]]

            for i in range(1, nums_len):
                # Combine current num with the previous subsets,
                # Then update the previous subsets
                if nums[i] == nums[i-1]:
                    for j in range(len(pre_subset)):
                        one_set = pre_subset[j][:]
                        one_set.append(nums[i])
                        subsets.append(one_set)
                        pre_subset[j] = one_set

                # Combine current num with all the subsets before.
                # Then update the previous subsets
                else:
                    pre_subset = []
                    for j in range(len(subsets)):
                        one_set = subsets[j][:]
                        one_set.append(nums[i])
                        subsets.append(one_set)
                        pre_subset.append(one_set)
                    pre_subset.append([nums[i]])
                    subsets.append([nums[i]])

            subsets.append([])
            return subsets

    """
    []
    [1,2]
    [1,2,2]
    [1,2,2,3,3,4,5]
    """



93. Restore IP Addresses
--------------------------------------

.. code-block:: python

    Given a string containing only digits, restore it by returning all possible valid IP address combinations.


    For example:
    Given "25525511135",


    return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
    ===================================================================
    class Solution(object):
      def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        ans = []
        n = len(s)

        def isValid(num):
          if len(num) == 1:
            return True
          if len(num) > 1 and num[0] != "0" and int(num) <= 255:
            return True
          return False

        for i in range(0, min(3, n - 3)):
          a = s[:i + 1]
          if not isValid(a):
            break
          for j in range(i + 1, min(i + 4, n - 2)):
            b = s[i + 1:j + 1]
            if not isValid(b):
              break
            for k in range(j + 1, min(j + 4, n - 1)):
              c = s[j + 1:k + 1]
              d = s[k + 1:]
              if not isValid(c):
                break
              if not isValid(d):
                continue
              ans.append("{}.{}.{}.{}".format(a, b, c, d))
        return ans


    ===================================================================
    class Solution(object):
        def restoreIpAddresses(self, s):
            """
            :type s: str
            :rtype: List[str]
            """
            address_block_list = self.restoreAddress(s, 1)
            address_list = []
            for address in address_block_list:
                if len(address) == 4:
                    address_list.append(".".join(address))
            return address_list

        def restoreAddress(self, s, count):
            address_block = []
            # No address field
            if not s:
                return address_block

            # We have get the fourth address fields
            if count == 4:
                if s[0] != "0" and len(s) <= 3 and int(s) <= 255:
                    address_block.append([s])
                if s == "0":
                    address_block.append([s])
                return address_block

            # Current field is '0'
            if s[0] == "0":
                address_1 = self.restoreAddress(s[1:], count + 1)
                for block in address_1:
                    cur_address = ['0']
                    cur_address.extend(block)
                    if len(cur_address) == 5 - count:
                        address_block.append(cur_address)
                return address_block

            # Current address field is made by i numbers.
            for i in range(1, 4):
                if len(s) < i or int(s[:i]) > 255:
                    continue
                address_1 = self.restoreAddress(s[i:], count + 1)
                for block in address_1:
                    cur_address = [s[:i]]
                    cur_address.extend(block)
                    if len(cur_address) == 5 - count:
                        address_block.append(cur_address)
            return address_block

    """
    "25525511135"
    "0000"
    "0100100"
    "11"
    """



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
    =================================================================
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
    =================================================================
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



216. Combination Sum 3
--------------------------------------

.. code-block:: python

    Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.



     Example 1:
    Input:  k = 3,  n = 7
    Output:

    [[1,2,4]]


     Example 2:
    Input:  k = 3,  n = 9
    Output:

    [[1,2,6], [1,3,5], [2,3,4]]



    Credits:Special thanks to @mithmatt for adding this problem and creating all test cases.
    =================================================================
    class Solution(object):
      def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """

        def dfs(k, start, path, subsum, res, visited):
          if len(path) == k and subsum == 0:
            res.append(path + [])
            return
          if len(path) >= k or subsum <= 0:
            return

          for i in range(start, 10):
            if visited[i] == 0:
              visited[i] = 1
              path.append(i)
              dfs(k, i + 1, path, subsum - i, res, visited)
              visited[i] = 0
              path.pop()

        visited = [0] * 10
        res = []
        dfs(k, 1, [], n, res, visited)
        return res


    =================================================================
    class Solution(object):
        def combinationSum3(self, k, n):
            self.combination = []
            self._combination_sum(k, n, [])
            return self.combination

        def _combination_sum(self, k, n, nums):
            if not k:
                if sum(nums) == n:
                    # self.combination.append(nums)
                    # Warning: nums[:] get a new list.
                    # If not, we will get self.combination = [[], [], ...] finally.
                    self.combination.append(nums[:])
                else:
                    return

            # Get the new num from start
            start = 1
            if nums:
                start = nums[-1] + 1
            for i in range(start, 10):
                cur_sum = sum(nums) + i
                if cur_sum <= n:
                    nums.append(i)
                    self._combination_sum(k - 1, n, nums)
                    del nums[-1]    # Backtracking
                else:
                    break

    """
    0
    3
    3
    7
    9
    45
    """
