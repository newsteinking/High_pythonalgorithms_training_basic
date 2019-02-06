Depth First Search - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

98. validate binary search tree
-------------------------------------

.. code-block:: python

    Given a binary tree, determine if it is a valid binary search tree (BST).



    Assume a BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.



    Example 1:

        2
       / \
      1   3

    Binary tree [2,1,3], return true.


    Example 2:

        1
       / \
      2   3

    Binary tree [1,2,3], return false.

    =================================================================
    class Solution(object):
      def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        prev = -float("inf")
        stack = [(1, root)]
        while stack:
          p = stack.pop()
          if not p[1]:
            continue
          if p[0] == 0:
            if p[1].val <= prev:
              return False
            prev = p[1].val
          else:
            stack.append((1, p[1].right))
            stack.append((0, p[1]))
            stack.append((1, p[1].left))
        return True


    =================================================================
    class Solution(object):
        def isValidBST(self, root):
            """
            :type root: TreeNode
            :rtype: bool
            """
            # When do inorder traversal, the val growth bigger.
            node_stack = []
            max_val = "init"
            while root or node_stack:
                if not root:
                    if not node_stack:
                        return True
                    node = node_stack.pop()
                    if max_val == "init" or node.val > max_val:
                        max_val = node.val
                    else:
                        return False
                    root = node.right
                else:
                    node_stack.append(root)
                    root = root.left
            return True

    """
    []
    [1]
    [1,null,2,3]
    [10,5,15,null,null,6,20]
    """


126. word ladder 2
--------------------

.. code-block:: python

    Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:


    Only one letter can be changed at a time
    Each transformed word must exist in the word list. Note that beginWord is not a transformed word.



    For example,


    Given:
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]


    Return

      [
        ["hit","hot","dot","dog","cog"],
        ["hit","hot","lot","log","cog"]
      ]




    Note:

    Return an empty list if there is no such transformation sequence.
    All words have the same length.
    All words contain only lowercase alphabetic characters.
    You may assume no duplicates in the word list.
    You may assume beginWord and endWord are non-empty and are not the same.




    UPDATE (2017/1/20):
    The wordList parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.

    =================================================================
    from collections import deque


    class Solution(object):
      def findLadders(self, beginWord, endWord, wordlist):
        """
        :type beginWord: str
        :type endWord: str
        :type wordlist: Set[str]
        :rtype: List[List[int]]
        """

        def getNbrs(src, dest, wordList):
          res = []
          for c in string.ascii_lowercase:
            for i in range(0, len(src)):
              newWord = src[:i] + c + src[i + 1:]
              if newWord == src:
                continue
              if newWord in wordList or newWord == dest:
                yield newWord

        def bfs(beginWord, endWord, wordList):
          distance = {beginWord: 0}
          queue = deque([beginWord])
          length = 0
          while queue:
            length += 1
            for k in range(0, len(queue)):
              top = queue.popleft()
              for nbr in getNbrs(top, endWord, wordList):
                if nbr not in distance:
                  distance[nbr] = distance[top] + 1
                  queue.append(nbr)
          return distance

        def dfs(beginWord, endWord, wordList, path, res, distance):
          if beginWord == endWord:
            res.append(path + [])
            return

          for nbr in getNbrs(beginWord, endWord, wordList):
            if distance.get(nbr, -2) + 1 == distance[beginWord]:
              path.append(nbr)
              dfs(nbr, endWord, wordList, path, res, distance)
              path.pop()

        res = []
        distance = bfs(endWord, beginWord, wordlist)
        dfs(beginWord, endWord, wordlist, [beginWord], res, distance)
        return res


    =================================================================
    class Solution(object):
        def findLadders(self, beginWord, endWord, wordlist):
            if beginWord == endWord:
                return [[beginWord]]
            cur_level = [beginWord]
            next_level = []
            visited_word = {}
            visited_word[beginWord] = 1

            # BFS: find whether there are shortest transformation sequence(s)
            find_shortest = False
            self.pre_word_list = {}
            while cur_level:
                if find_shortest:
                    break
                for cur_word in cur_level:
                    cur_len = len(cur_word)
                    # Get the next level
                    # When I put "abc...xyz" in the out loop, it just exceeded.
                    for i in range(cur_len):
                        pre_word = cur_word[:i]
                        post_word = cur_word[i+1:]
                        for j in "abcdefghijklmnopqrstuvwxyz":
                            next_word = pre_word + j + post_word
                            # Just find one shorttest transformation sequence
                            if next_word == endWord:
                                find_shortest = True
                            else:
                                pass
                            if (next_word not in visited_word and
                                    next_word in wordlist or next_word == endWord):
                                if next_word not in next_level:
                                    next_level.append(next_word)
                                else:
                                    pass

                                if next_word not in self.pre_word_list:
                                    self.pre_word_list[next_word] = [cur_word]
                                else:
                                    self.pre_word_list[next_word].append(cur_word)
                            else:
                                pass
                for w in next_level:
                    visited_word[w] = 1
                # Scan the next level then
                cur_level = next_level
                next_level = []
            if find_shortest:
                self.results = []
                self.dfs_sequences(beginWord, endWord, [endWord])
                return self.results
            else:
                return []

        """
        Build the path according to the pre_word_list
        """
        def dfs_sequences(self, beginWord, endWord, path):
            if beginWord == endWord:
                self.results.append(list(path[-1::-1]))
            elif endWord in self.pre_word_list:
                for pre_w in self.pre_word_list[endWord]:
                    path.append(pre_w)
                    self.dfs_sequences(beginWord, pre_w, path)
                    path.pop()
            else:
                pass
            return

    """
    if __name__ == '__main__':
        sol = Solution()

        print sol.findLadders("hit", "hhh", ["hot", "dot", "dog", "lot", "log"])
        print sol.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log"])
        print sol.findLadders(
            "hit", "cog",
            ["hot", "dot", "dog", "lot", "log", "hog"])

        print sol.findLadders(
            "cet", "ism",
            ['cot', 'con', 'ion', 'inn', 'ins', 'its', 'ito', 'ibo', 'ibm', 'get',
             'gee', 'gte', 'ate', 'ats', 'its', 'ito', 'ibo', 'ibm'])
    """


129. sum root to leaf numbers
------------------------------------

.. code-block:: python

    Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
    An example is the root-to-leaf path 1->2->3 which represents the number 123.

    Find the total sum of all root-to-leaf numbers.

    For example,

        1
       / \
      2   3



    The root-to-leaf path 1->2 represents the number 12.
    The root-to-leaf path 1->3 represents the number 13.


    Return the sum = 12 + 13 = 25.

    =================================================================
    class Solution(object):
      def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.sum = 0

        def dfs(root, pathsum):
          if root:
            pathsum += root.val
            left = dfs(root.left, pathsum * 10)
            right = dfs(root.right, pathsum * 10)
            if not left and not right:
              self.sum += pathsum
            return True

        dfs(root, 0)
        return self.sum


    =================================================================
    class Solution(object):
        """
        Depth First Search
        """
        def sumNumbers(self, root):
            node_stack = []
            path_sum = 0
            # Keep the path number from root to the current node.
            cur_node_num = 0

            while root or node_stack:
                if root:
                    cur_node_num = cur_node_num * 10 + root.val
                    node_stack.append([root, cur_node_num])
                    root = root.left

                else:
                    if node_stack:
                        pop_record = node_stack.pop()
                        root = pop_record[0].right
                        cur_node_num = pop_record[1]
                        # Meet a leaf node
                        if not pop_record[0].left and not root:
                            path_sum += cur_node_num

                    else:
                        break
            return path_sum

    """
    []
    [1,2,3]
    [1,null,2,3,null,null,4,5,6]
    """


140. word break 2
--------------------

.. code-block:: python

    Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. You may assume the dictionary does not contain duplicate words.



    Return all such possible sentences.



    For example, given
    s = "catsanddog",
    dict = ["cat", "cats", "and", "sand", "dog"].



    A solution is ["cats and dog", "cat sand dog"].



    UPDATE (2017/1/4):
    The wordDict parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.

    =================================================================
    class Solution(object):
      def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        res = []
        if not self.checkWordBreak(s, wordDict):
          return res
        queue = [(0, "")]
        slen = len(s)
        lenList = [l for l in set(map(len, wordDict))]
        while queue:
          tmpqueue = []
          for q in queue:
            start, path = q
            for l in lenList:
              if start + l <= slen and s[start:start + l] in wordDict:
                newnode = (start + l, path + " " + s[start:start + l] if path else s[start:start + l])
                tmpqueue.append(newnode)
                if start + l == slen:
                  res.append(newnode[1])
          queue, tmpqueue = tmpqueue, []
        return res

      def checkWordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        queue = [0]
        slen = len(s)
        lenList = [l for l in set(map(len, wordDict))]
        visited = [0 for _ in range(0, slen + 1)]
        while queue:
          tmpqueue = []
          for start in queue:
            for l in lenList:
              if s[start:start + l] in wordDict:
                if start + l == slen:
                  return True
                if visited[start + l] == 0:
                  tmpqueue.append(start + l)
                  visited[start + l] = 1
          queue, tmpqueue = tmpqueue, []
        return False


    =================================================================
    class Solution(object):
        """
        Dynamic Programming
        dp[i]: if s[i:] can be broken to wordDict. then:
        dp[i-1] = s[i:i+k] in wordDict and dp[i+k+1], for all the possible k.
        """
        def wordBreak(self, s, wordDict):
            if not s:
                return [""]

            self.s_len = len(s)
            self.result = []
            self.str = s
            self.words = wordDict

            dp = [False for i in range(self.s_len + 1)]
            dp[-1] = True

            for i in range(self.s_len - 1, -1, -1):
                k = 0
                while k + i < self.s_len:
                    cur_fisrt_word = self.str[i:i+k+1]
                    if cur_fisrt_word in self.words and dp[i + k + 1]:
                        dp[i] = True
                        break

                    k += 1

            self.word_break(0, [], dp)
            return self.result

        # Depth First Search
        def word_break(self, start, word_list, dp):
            if start == self.s_len:
                self.result.append(" ".join(word_list))
                return

            k = 0
            while start+k < self.s_len:
                cur_word = self.str[start:start+k+1]
                if cur_word in self.words and dp[start+k+1]:
                    word_list.append(cur_word)
                    self.word_break(start+k+1, word_list, dp)
                    word_list.pop()
                k += 1
    """
    "a"
    []
    ""
    []
    "catsanddog"
    ["cat","cats","and","sand","dog"]
    "leetcode"
    ["leet", "code", "lee", "t"]
    """


200. number of islands
-----------------------------

.. code-block:: python

    Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

    Example 1:
    11110110101100000000
    Answer: 1
    Example 2:
    11000110000010000011
    Answer: 3

    Credits:Special thanks to @mithmatt for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        visited = set()
        ans = 0

        def dfs(grid, i, j, visited):
          if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == "0" or (i, j) in visited:
            return False
          visited |= {(i, j)}
          for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            newi, newj = i + di, j + dj
            dfs(grid, newi, newj, visited)
          return True

        for i in range(0, len(grid)):
          for j in range(0, len(grid[0])):
            if dfs(grid, i, j, visited):
              ans += 1
        return ans

    =================================================================
    class Solution(object):
        def numIslands(self, grid):
            if not grid:
                return 0
            island_counts = 0
            self.m_rows = len(grid)
            self.n_cols = len(grid[0])
            self.grid = grid

            island_counts = 0
            for i in range(self.m_rows):
                for j in range(self.n_cols):
                    if grid[i][j] == "1":
                        island_counts += 1
                        self.merge_surround(i, j)

            return island_counts

        # Depth First Search
        # Merge all the adjacent islands to one island.
        def merge_surround(self, i, j):
            if self.grid[i][j] == "1" or self.grid[i][j] == "#":
                if i+1 < self.m_rows and self.grid[i+1][j] == "1":
                    self.grid[i+1][j] = "#"
                    self.merge_surround(i+1, j)
                if j+1 < self.n_cols and self.grid[i][j+1] == "1":
                    self.grid[i][j+1] = "#"
                    self.merge_surround(i, j+1)
                if i-1 >= 0 and self.grid[i-1][j] == "1":
                    self.grid[i-1][j] = "#"
                    self.merge_surround(i-1, j)
                if j-1 >= 0 and self.grid[i][j-1] == "1":
                    self.grid[i][j-1] = "#"
                    self.merge_surround(i, j-1)
            return

    """
    ["1"]
    ["111","010","111"]
    ["111", "100", "101", "111"]
    ["11110", "11010", "11000", "00000"]
    ["11000", "11000", "00100", "00011"]
    """


257. Binary tree paths
-------------------------------

.. code-block:: python

    Given a binary tree, return all root-to-leaf paths.


    For example, given the following binary tree:



       1
     /   \
    2     3
     \
      5



    All root-to-leaf paths are:
    ["1->2->5", "1->3"]


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution:
      # @param {TreeNode} root
      # @return {string[]}
      def binaryTreePaths(self, root):
        def helper(root, path, res):
          if root:
            path.append(str(root.val))
            left = helper(root.left, path, res)
            right = helper(root.right, path, res)
            if not left and not right:
              res.append("->".join(path))
            path.pop()
            return True

        res = []
        helper(root, [], res)
        return res


    =================================================================
    class Solution(object):
        def binaryTreePaths(self, root):
            if not root:
                return []
            node_stack = [[root, str(root.val)]]
            path_str = []
            while node_stack:
                node, path = node_stack.pop()
                if node.left:
                    new_path = path + "->" + str(node.left.val)
                    node_stack.append([node.left, new_path])
                if node.right:
                    new_path = path + "->" + str(node.right.val)
                    node_stack.append([node.right, new_path])
                if not node.left and not node.right:
                    path_str.append(path)
            return path_str

    """
    []
    [1]
    [1,2,3,4,null,null,null,5]
    """


282. expression add operators
------------------------------------

.. code-block:: python

    Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.


    Examples:
    "123", 6 -> ["1+2+3", "1*2*3"]
    "232", 8 -> ["2*3+2", "2+3*2"]
    "105", 5 -> ["1*0+5","10-5"]
    "00", 0 -> ["0+0", "0-0", "0*0"]
    "3456237490", 9191 -> []


    Credits:Special thanks to @davidtan1890 for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def addOperators(self, num, target):
        res, self.target = [], target
        for i in range(1, len(num) + 1):
          if i == 1 or (i > 1 and num[0] != "0"):  # prevent "00*" as a number
            self.dfs(num[i:], num[:i], int(num[:i]), int(num[:i]), res)  # this step put first number in the string
        return res

      def dfs(self, num, temp, cur, last, res):
        if not num:
          if cur == self.target:
            res.append(temp)
          return
        for i in range(1, len(num) + 1):
          val = num[:i]
          if i == 1 or (i > 1 and num[0] != "0"):  # prevent "00*" as a number
            self.dfs(num[i:], temp + "+" + val, cur + int(val), int(val), res)
            self.dfs(num[i:], temp + "-" + val, cur - int(val), -int(val), res)
            self.dfs(num[i:], temp + "*" + val, cur - last + last * int(val), last * int(val), res)


    =================================================================
    class Solution(object):
        def addOperators(self, num, target):
            """ Once you can understand the solution space tree, you just get it.

            Refer to:
            https://discuss.leetcode.com/topic/24523/java-standard-backtrace-ac-solutoin-short-and-clear
            """
            ans = []
            self.dfs_search(ans, "", num, target, 0, 0, 0)
            return ans

        def dfs_search(self, ans, path, num, target, pos, pre_num, value):
            """  Put binary operator in pos, and then calculate the new value.

            @pre_num: when process *, we need to know the previous number.
            """
            if pos == len(num):
                if value == target:
                    ans.append(path)
                return

            for i in range(pos + 1, len(num) + 1):
                cur_str, cur_n = num[pos: i], int(num[pos: i])
                # Digit can not begin with 0 (01, 00, 02 are not valid), except 0 itself.
                if i > pos + 1 and num[pos] == '0':
                    break
                if pos == 0:
                    self.dfs_search(ans, path + cur_str, num, target, i, cur_n, cur_n)
                # All three different binary operators: +, -, *
                else:
                    self.dfs_search(ans, path + "+" + cur_str, num,
                                    target, i, cur_n, value + cur_n)
                    self.dfs_search(ans, path + "-" + cur_str, num,
                                    target, i, -cur_n, value - cur_n)
                    self.dfs_search(ans, path + "*" + cur_str, num,
                                    target, i, pre_num * cur_n, value - pre_num + pre_num * cur_n)

    """
    "000"
    0
    "123"
    6
    "232"
    8
    "1005"
    5
    "3456237490"
    9191
    """


301. remove invalid parentheses
----------------------------------

.. code-block:: python

    Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

    Note: The input string may contain letters other than the parentheses ( and ).



    Examples:

    "()())()" -> ["()()()", "(())()"]
    "(a)())()" -> ["(a)()()", "(a())()"]
    ")(" -> [""]



    Credits:Special thanks to @hpplayer for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        def isValid(s):
          stack = []
          for c in s:
            if c == "(":
              stack.append("(")
            elif c == ")":
              stack.append(")")
              if len(stack) >= 2 and stack[-2] + stack[-1] == "()":
                stack.pop()
                stack.pop()
          return len(stack)

        def dfs(s, res, cache, length):
          if s in cache:
            return

          if len(s) == length:
            if isValid(s) == 0:
              res.append(s)
              return

          for i in range(0, len(s)):
            if s[i] == "(" or s[i] == ")" and len(s) - 1 >= length:
              dfs(s[:i] + s[i + 1:], res, cache, length)
              cache.add(s[:i] + s[i + 1:])

        prepLeft = ""
        for i in range(0, len(s)):
          if s[i] == "(":
            prepLeft += s[i:]
            break
          elif s[i] != ")":
            prepLeft += s[i]

        prepRight = ""
        for i in reversed(range(0, len(prepLeft))):
          if prepLeft[i] == ")":
            prepRight += prepLeft[:i + 1][::-1]
            break
          elif prepLeft[i] != "(":
            prepRight += prepLeft[i]

        s = prepRight[::-1]
        length = len(s) - isValid(s)
        res = []
        cache = set()
        dfs(s, res, cache, length)
        return res


    =================================================================
    class Solution(object):
        """
        Violent search: Generate all possible states by removing one ( or ),
        check if they are valid.  It is something like Breadth First Search.
        Easy to understand but slower.
        """
        def removeInvalidParentheses(self, s):
            unvalid_str = {s}
            # unvalid_str = set(s)  Wrong initinal way.

            # Every time we go into the iteration,
            # We delete one more parentheses then check all the possible situation.
            while True:
                valid = filter(self.isvalid, unvalid_str)
                if valid:
                    return valid
                else:
                    new_set = set()
                    for str in unvalid_str:
                        for i in range(len(str)):
                            new_set.add(str[:i] + str[i+1:])
                    unvalid_str = new_set

        def isvalid(self, str):
            count = 0
            for c in str:
                if c == "(":
                    count += 1
                elif c == ")":
                    count -= 1
                    if count < 0:
                        return False
                else:
                    pass

            return count == 0


    class Solution_2(object):
        """
        Depth First Search with backtrack.
        Generate new strings by removing parenthesis,
        and calculate the total number of mismatched parentheses.
            1. If the mismatched parentheses increased, then go back.
            2. Otherwise, remove parentheses until have no mismatched parentheses.
        """
        def removeInvalidParentheses(self, s):
            self.visited = {s}   # self.visited = set([s])
            return self.dfsRemove(s)

        def dfsRemove(self, s):
            count = self.mismatchedCount(s)
            if count == 0:
                return [s]

            result = []
            for i in range(len(s)):
                if s[i] not in "()":
                    continue
                # Remove one ( or )
                new = s[:i] + s[i+1:]
                if new not in self.visited and self.mismatchedCount(new) < count:
                    self.visited.add(new)
                    result.extend(self.dfsRemove(new))
            return result

        def mismatchedCount(self, s):
            """
            Get the total number of mismatched parentheses in string s.
            Actually it's the minimum number of parentheses we need to remove.
            """
            mis_l, mis_r = 0, 0
            for ch in s:
                if ch == "(":
                    mis_l += 1
                elif ch == ")":
                    mis_l -= 1
                    mis_r += (mis_l < 0)
                    mis_l = max(mis_l, 0)
                else:
                    pass
            return mis_l + mis_r


    class Solution_3(object):
        """
        The fastest one.  According to:
        https://leetcode.com/discuss/82300/44ms-python-solution
        The main point is:
            1. Scan from left to right, make sure count["("]>=count[")"].
            2. Then scan from right to left, make sure count["("]<=count[")"].
        """
        def removeInvalidParentheses(self, s):
            possibles = {s}
            count = {"(": 0, ")": 0}
            removed = 0

            # Scan s from left to right to remove mismatched ).
            for i, ch in enumerate(s):
                # Remove pre or current ) to make count["("] >= count[")"]
                if ch == ")" and count["("] == count[")"]:
                    new_possible = set()
                    while possibles:
                        j = 0
                        str = possibles.pop()
                        while j + removed <= i:
                            if str[j] == ")":
                                new_possible.add(str[:j] + str[j+1:])
                            j += 1
                    possibles = new_possible
                    removed += 1
                elif ch in count:
                    count[ch] += 1
                else:
                    pass

            # Scan possibles from right to left to remove mismatched (.
            count = {"(": 0, ")": 0}
            possible_len = len(s) - removed
            pos = len(s)
            for i in range(possible_len-1, -1, -1):
                # !!! Attention: all mismatched ( appear after mismatched ).
                pos -= 1
                ch = s[pos]
                # Remove post or current ( to make count["("] <= count[")"]
                if ch == "(" and count["("] == count[")"]:
                    new_possible = set()
                    while possibles:
                        str = possibles.pop()
                        j = i
                        while j < len(str):
                            if str[j] == "(":
                                new_possible.add(str[:j] + str[j+1:])
                            j += 1
                    possibles = new_possible
                elif ch in count:
                    count[ch] += 1
                else:
                    pass

            return list(possibles)

    """
    ""
    ")("
    ")))"
    "((("
    ")()("
    "())))"
    "()())()"
    "(a)())()"
    """


306. additive number
-------------------------

.. code-block:: python

    Additive number is a string whose digits can form additive sequence.

    A valid additive sequence should contain at least three numbers. Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two.


    For example:
    "112358" is an additive number because the digits can form an additive sequence: 1, 1, 2, 3, 5, 8.
    1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
    "199100199" is also an additive number, the additive sequence is: 1, 99, 100, 199.
    1 + 99 = 100, 99 + 100 = 199



    Note: Numbers in the additive sequence cannot have leading zeros, so sequence 1, 2, 03 or 1, 02, 3 is invalid.


    Given a string containing only digits '0'-'9', write a function to determine if it's an additive number.


    Follow up:
    How would you handle overflow for very large input integers?


    Credits:Special thanks to @jeantimex for adding this problem and creating all test cases.
    
    =================================================================
    class Solution(object):
      def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """
        n = len(num)
        for x in range(0, n / 2):
          if x > 0 and num[0] == "0":
            break
          for y in range(x + 1, n):
            if y - x > 1 and num[x + 1] == "0":
              break
            i, j, k = 0, x, y
            while k < n:
              a = int(num[i:j + 1])
              b = int(num[j + 1:k + 1])
              add = str(int(a + b))
              if not num.startswith(add, k + 1):
                break
              if len(add) + 1 + k == len(num):
                return True
              i = j + 1
              j = k
              k = k + len(add)
        return False


    =================================================================
    class Solution(object):
        # According to:
        # https://leetcode.com/discuss/70089/python-solution
        # The key point is choose first two number then recursively check.
        # DFS: recursice implement.
        def isAdditiveNumber(self, num):
            length = len(num)
            for i in range(1, length/2+1):
                for j in range(1, (length-i)/2 + 1):
                    first, second, others = num[:i], num[i:i+j], num[i+j:]
                    if self.isValid(first, second, others):
                        return True
            return False

        def isValid(self, first, second, others):
            # Numbers in the additive sequence cannot have leading zeros,
            if ((len(first) > 1 and first[0] == "0") or
                    (len(second) > 1 and second[0] == "0")):
                return False
            sum_str = str(int(first) + int(second))
            if sum_str == others:
                return True
            elif others.startswith(sum_str):
                return self.isValid(second, sum_str, others[len(sum_str):])
            else:
                return False


    class Solution_2(object):
        # DFS: iterative implement.
        def isAdditiveNumber(self, num):
            length = len(num)
            for i in range(1, length/2+1):
                for j in range(1, (length-i)/2 + 1):
                    first, second, others = num[:i], num[i:i+j], num[i+j:]
                    if ((len(first) > 1 and first[0] == "0") or
                            (len(second) > 1 and second[0] == "0")):
                        continue

                    while others:
                        sum_str = str(int(first) + int(second))
                        if sum_str == others:
                            return True
                        elif others.startswith(sum_str):
                            first, second, others = (
                                second, sum_str, others[len(sum_str):])
                        else:
                            break

            return False

    """
    "1123"
    "1203"
    "112324"
    "112334"
    "112358"
    """
