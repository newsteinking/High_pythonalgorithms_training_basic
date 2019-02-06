Breadth First Search - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

102. Binary Tree level order traveral
-------------------------------------------

.. code-block:: python

    Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).


    For example:
    Given binary tree [3,9,20,null,null,15,7],

        3
       / \
      9  20
        /  \
       15   7



    return its level order traversal as:

    [
      [3],
      [9,20],
      [15,7]
    ]


    =================================================================
    from collections import deque


    class Solution(object):
      def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
          return []
        ans = [[root.val]]
        queue = deque([root])
        while queue:
          levelans = []
          for _ in range(0, len(queue)):
            root = queue.popleft()
            if root.left:
              levelans.append(root.left.val)
              queue.append(root.left)
            if root.right:
              levelans.append(root.right.val)
              queue.append(root.right)
          if levelans:
            ans.append(levelans)
        return ans


    =================================================================
    class Solution(object):
        def levelOrder(self, root):
            if not root:
                return []

            cur_level, ans = [root], []

            # Breadth-first Search, Pythonic way.
            while cur_level:
                ans.append([node.val for node in cur_level])
                cur_level = [kid for n in cur_level
                             for kid in (n.left, n.right) if kid]

            return ans

    """
    []
    [1]
    [1,2,3]
    [3,9,20,null,null,15,7]
    """

103. Binary tree zigzag level order traversal
--------------------------------------------------

.. code-block:: python

    Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).


    For example:
    Given binary tree [3,9,20,null,null,15,7],

        3
       / \
      9  20
        /  \
       15   7



    return its zigzag level order traversal as:

    [
      [3],
      [20,9],
      [15,7]
    ]

    =================================================================
    from collections import deque


    class Solution(object):
      def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        stack = deque([root])
        ans = []
        odd = True
        while stack:
          level = []
          for k in range(0, len(stack)):
            top = stack.popleft()
            if top is None:
              continue
            level.append(top.val)
            stack.append(top.left)
            stack.append(top.right)
          if level:
            if odd:
              ans.append(level)
            else:
              ans.append(level[::-1])
          odd = not odd
        return ans

    =================================================================
    class Solution(object):
        def zigzagLevelOrder(self, root):
            if not root:
                return []

            left2right = 1
            # 1. scan the level from left to right. -1 reverse.
            ans, stack, temp = [], [root], []
            while stack:
                temp = [node.val for node in stack]
                stack = [child for node in stack
                         for child in (node.left, node.right) if child]

                ans += [temp[::left2right]]         # Pythonic way
                left2right *= -1

            return ans

    """
    []
    [1]
    [1,2,3]
    [0,1,2,3,4,5,6,null,null,7,null,8,9,null,10]
    """


104. Maximum depth of binary tree
-------------------------------------------

.. code-block:: python

    Given a binary tree, find its maximum depth.

    The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

    =================================================================
    class Solution(object):
      def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
          return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    =================================================================
    class Solution(object):
        def maxDepth(self, root):
            """
            :type root: TreeNode
            :rtype: int
            """
            if not root:
                return 0

            node_list = [root]
            depth_count = 1
            # Breadth-first Search
            while node_list:
                # node_scan: all the nodes in one level.
                # Traverse node_scan and get all the nodes of next level,
                # Then update node_list
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
    """
    []
    [1]
    [1,2,3]
    [0,1,2,3,4,5,6,null,null,7,null,8,9,null,10]
    """


107. Binary tree level order traveral 2
-------------------------------------------

.. code-block:: python

    Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).


    For example:
    Given binary tree [3,9,20,null,null,15,7],

        3
       / \
      9  20
        /  \
       15   7



    return its bottom-up level order traversal as:

    [
      [15,7],
      [9,20],
      [3]
    ]

    =================================================================
    from collections import deque


    class Solution(object):
      def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
          return []
        ans = [[root.val]]
        queue = deque([root])
        while queue:
          levelans = []
          for _ in range(0, len(queue)):
            root = queue.popleft()
            if root.left:
              levelans.append(root.left.val)
              queue.append(root.left)
            if root.right:
              levelans.append(root.right.val)
              queue.append(root.right)
          if levelans:
            ans.append(levelans)
        return ans[::-1]

    =================================================================
    class Solution(object):
        def levelOrderBottom(self, root):
            """
            :type root: TreeNode
            :rtype: List[List[int]]
            """
            if not root:
                return []

            node_list = [root]
            level_traversal = [[root.val]]

            # Breadth-first Search
            while node_list:
                # node_scan: all the nodes in one level.
                # Traverse node_scan and get all the nodes of next level,
                # Then update node_list, and the solution level_traversal
                node_scan = node_list[:]
                node_list = []
                node_level = []
                for node in node_scan:
                    l_child = node.left
                    r_child = node.right
                    if l_child:
                        node_level.append(l_child.val)
                        node_list.append(l_child)
                    if r_child:
                        node_level.append(r_child.val)
                        node_list.append(r_child)
                if node_level:
                    level_traversal.insert(0, node_level)

            return level_traversal

    """
    []
    [1]
    [1,2,3]
    [3,9,20,null,null,15,7]
    """



126. word ladder 2
-------------------------------------------

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



127. word ladder
-------------------------------------------

.. code-block:: python

    Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:


    Only one letter can be changed at a time.
    Each transformed word must exist in the word list. Note that beginWord is not a transformed word.



    For example,


    Given:
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]


    As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
    return its length 5.



    Note:

    Return 0 if there is no such transformation sequence.
    All words have the same length.
    All words contain only lowercase alphabetic characters.
    You may assume no duplicates in the word list.
    You may assume beginWord and endWord are non-empty and are not the same.




    UPDATE (2017/1/20):
    The wordList parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.

    =================================================================
    import string
    from collections import deque


    class Solution(object):
      def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: Set[str]
        :rtype: int
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

        queue = deque([beginWord])
        length = 0
        while queue:
          length += 1
          for k in range(0, len(queue)):
            top = queue.popleft()
            for nbr in getNbrs(top, endWord, wordList):
              wordList.remove(nbr)
              if nbr == endWord:
                return length + 1
              queue.append(nbr)
        return 0

    =================================================================
    class Solution(object):
        def ladderLength(self, beginWord, endWord, wordList):
            """
            Breadth First Search
            When build the adjacency tree, skip the visited word
            """
            if beginWord == endWord:
                return 1
            cur_level = [beginWord]
            next_level = []
            visited_word = {}
            visited_word[beginWord] = 1
            length = 0
            while cur_level:
                length += 1
                for cur_word in cur_level:
                    cur_len = len(cur_word)
                    # Get the next level
                    # When I put "abc...xyz" in the out loop, it just exceeded.
                    for i in range(cur_len):
                        pre_word = cur_word[:i]
                        post_word = cur_word[i+1:]
                        for j in "abcdefghijklmnopqrstuvwxyz":
                            next_word = pre_word + j + post_word
                            # Find the endWord
                            if next_word == endWord:
                                return length + 1
                            elif (next_word not in visited_word and
                                    next_word in wordList):
                                visited_word[next_word] = 1
                                next_level.append(next_word)
                            else:
                                pass

                # Scan the next level then
                cur_level = next_level
                next_level = []
            return 0

        """ disapproved, when wordList growth bigger, it may be called too many times
        def is_one_distance(self, word_1, word_2):
            # alert(len(word_1) == len(word_2))
            word_l = len(word_1)
            one_distance = False
            for i in range(word_l):
                if word_1[i] != word_2[i]:
                    if not one_distance:
                        one_distance = True
                    else:
                        return False

            return one_distance
        """
    """
    if __name__ == '__main__':
        sol = Solution()
        print sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"])
        print sol.ladderLength("hit", "cog", ["hot", "dot", "doh", "lot", "loh"])
        print sol.ladderLength(
            "hit", "cog",
            ["hot", "dot", "dog", "lot", "log", "hig", "hog"])

        print sol.ladderLength(
            "cet", "ism",
            ['cot', 'con', 'ion', 'inn', 'ins', 'its', 'ito', 'ibo', 'ibm', 'get',
             'gee', 'gte', 'ate', 'ats', 'its', 'ito', 'ibo', 'ibm'])
    """



130. Surrounded regions
-------------------------------------------

.. code-block:: python

    Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

    A region is captured by flipping all 'O's into 'X's in that surrounded region.



    For example,

    X X X X
    X O O X
    X X O X
    X O X X




    After running your function, the board should be:

    X X X X
    X X X X
    X X X X
    X O X X
    =================================================================
    class UnionFind():
      def __init__(self, m, n):
        self.dad = [i for i in range(0, m * n)]
        self.rank = [0 for i in range(0, m * n)]
        self.m = m
        self.n = n

      def find(self, x):
        dad = self.dad
        if dad[x] != x:
          dad[x] = self.find(dad[x])
        return dad[x]

      def union(self, xy):
        dad = self.dad
        rank = self.rank
        x, y = map(self.find, xy)
        if x == y:
          return False
        if rank[x] > rank[y]:
          dad[y] = x
        else:
          dad[x] = y
          if rank[x] == rank[y]:
            rank[y] += 1
        return True


    class Solution:
      # @param {list[list[str]]} board a 2D board containing 'X' and 'O'
      # @return nothing
      def solve(self, board):
        # Write your code here
        if len(board) == 0:
          return
        regions = set([])
        n, m = len(board), len(board[0])
        uf = UnionFind(len(board[0]), len(board))
        directions = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
        for i in range(0, len(board)):
          for j in range(0, len(board[0])):
            if board[i][j] == 'X':
              continue
            for d in ["d", "r"]:
              di, dj = directions[d]
              newi, newj = i + di, j + dj
              if newi >= 0 and newi < len(board) and newj >= 0 and newj < len(board[0]):
                if board[newi][newj] == "O":
                  uf.union((newi * m + newj, i * m + j))

        for i in range(0, len(board)):
          for j in [0, len(board[0]) - 1]:
            if board[i][j] == "O":
              regions.add(uf.find(i * m + j))

        for j in range(0, len(board[0])):
          for i in [0, len(board) - 1]:
            if board[i][j] == "O":
              regions.add(uf.find(i * m + j))

        for i in range(0, len(board)):
          for j in range(0, len(board[0])):
            if board[i][j] == "O" and uf.find(i * m + j) not in regions:
              board[i][j] = "X"

    =================================================================
    class Solution(object):
        def solve(self, board):
            """
            :type board: List[List[str]]
            :rtype: void Do not return anything, modify board in-place instead.
            """
            if not board:
                return
            m_rows = len(board)
            n_cols = len(board[0])
            if m_rows <= 2 or n_cols <= 2:
                return

            for row in range(m_rows):
                board[row] = list(board[row])

            # Search from the first and last row
            for i in [0, m_rows-1]:
                for j in range(n_cols):
                    if board[i][j] == "O":
                        self.breadth_first_search(i, j, board)

            # Search from the first and last column
            for j in [0, n_cols-1]:
                for i in range(m_rows):
                    if board[i][j] == "O":
                        self.breadth_first_search(i, j, board)

            # Make all the O surrounded by X to be X
            for i in range(m_rows):
                for j in range(n_cols):
                    if board[i][j] == "O":
                        board[i][j] = "X"
                    if board[i][j] == "#":
                        board[i][j] = "O"
                board[i] = "".join(board[i])

        """
        Mark all the Os can be arrived from outside row(column) to be '#'
        And return one O node's adjacent O nodes
        """
        def set_adjacency(self, row, col, board):
            board[row][col] = "#"
            adjacency_node = []
            m_rows = len(board)
            n_cols = len(board[0])
            if row - 1 >= 0 and board[row-1][col] == "O":
                board[row-1][col] = "#"
                adjacency_node.append([row-1, col])
            if row + 1 < m_rows and board[row+1][col] == "O":
                board[row+1][col] = "#"
                adjacency_node.append([row+1, col])
            if col - 1 >= 0 and board[row][col-1] == "O":
                board[row][col-1] = "#"
                adjacency_node.append([row, col-1])
            if col + 1 < n_cols and board[row][col+1] == "O":
                board[row][col+1] = "#"
                adjacency_node.append([row, col+1])
            return adjacency_node

        # Do BFS to every out border O ndoe.
        def breadth_first_search(self, row, col, board):
            adjacency_nodes = self.set_adjacency(row, col, board)
            adjacency_record = []
            while adjacency_nodes:
                for node in adjacency_nodes:
                    adjacency_record.extend(
                        self.set_adjacency(node[0], node[1], board))
                adjacency_nodes = adjacency_record
                adjacency_record = []
    """
    []
    ["XXX", "XOX", "XXX"]
    ["OOX", "XOX", "OXX"]
    ["XXXX", "XOOX", "XXOX", "XOXX"]
    """



199. Binary tree right side view
-------------------------------------------

.. code-block:: python

    Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.


    For example:
    Given the following binary tree,

       1            <---
     /   \
    2     3         <---
     \     \
      5     4       <---



    You should return [1, 3, 4].


    Credits:Special thanks to @amrsaqr for adding this problem and creating all test cases.
    =================================================================
    from collections import deque


    class Solution(object):
      def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """

        def dfs(root, h):
          if root:
            if h == len(ans):
              ans.append(root.val)
            dfs(root.right, h + 1)
            dfs(root.left, h + 1)

        ans = []
        dfs(root, 0)
        return ans

    =================================================================
    class Solution(object):
        # Breadth First Search
        def rightSideView(self, root):
            if not root:
                return []
            cur_level = [root]
            next_level = []
            result = []
            while cur_level:
                for node in cur_level:
                    if node.left:
                        next_level.append(node.left)
                    if node.right:
                        next_level.append(node.right)
                result.append(cur_level[-1].val)
                cur_level = next_level
                next_level = []
            return result

    """
    []
    [1,2,3]
    [1,2,3,null,4,null,5]
    """



310. Minimum height trees
-------------------------------------------

.. code-block:: python

    For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs).
    Given such a graph, write a function to find all the MHTs and return a list of their root labels.



    Format
    The graph contains n nodes which are labeled from 0 to n - 1.
    You will be given the number n and a list of undirected edges (each edge is a pair of labels).


    You can assume that no duplicate edges will appear in edges. Since all edges are
        undirected, [0, 1] is the same as [1, 0] and thus will not appear together in
        edges.


        Example 1:


        Given n = 4, edges = [[1, 0], [1, 2], [1, 3]]



            0
            |
            1
           / \
          2   3


        return  [1]



        Example 2:


        Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]


         0  1  2
          \ | /
            3
            |
            4
            |
            5


        return  [3, 4]



        Note:


        (1) According to the definition
        of tree on Wikipedia: �쏿 tree is an undirected graph in which any two vertices are connected by
        exactly one path. In other words, any connected graph without simple cycles is a tree.��


        (2) The height of a rooted tree is the number of edges on the longest downward path between the root and a
        leaf.


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    from collections import deque


    class Solution(object):
      def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if len(edges) == 0:
          if n > 0:
            return [0]
          return []

        def bfs(graph, start):
          queue = deque([(start, None)])
          level = 0
          maxLevel = -1
          farthest = None
          while queue:
            level += 1
            for i in range(0, len(queue)):
              label, parent = queue.popleft()
              for child in graph.get(label, []):
                if child != parent:
                  queue.append((child, label))
                  if level > maxLevel:
                    maxLevel = level
                    farthest = child
          return farthest

        def dfs(graph, start, end, visited, path, res):
          if start == end:
            res.append(path + [])
            return True
          visited[start] = 1
          for child in graph.get(start, []):
            if visited[child] == 0:
              path.append(child)
              if dfs(graph, child, end, visited, path, res):
                return True
              path.pop()

        graph = {}
        for edge in edges:
          graph[edge[0]] = graph.get(edge[0], []) + [edge[1]]
          graph[edge[1]] = graph.get(edge[1], []) + [edge[0]]

        start = bfs(graph, edges[0][0])
        end = bfs(graph, start)
        res, visited = [], [0 for i in range(0, n)]
        dfs(graph, start, end, visited, [start], res)
        if not res:
          return []
        path = res[0]
        if len(path) % 2 == 0:
          return [path[len(path) / 2 - 1], path[len(path) / 2]]
        else:
          return [path[len(path) / 2]]

    =================================================================
    class Solution(object):
        """
        The basic idea is
        "keep deleting leaves layer-by-layer, until reach the root."

        Specifically, first find all the leaves, then remove them.
        After removing, some nodes will become new leaves. So we can
        continue remove them. Eventually, there is only 1 or 2 nodes
        left. If there is only one node left, it is the root. If there
        are 2 nodes, either of them could be a possible root.
        """
        def findMinHeightTrees(self, n, edges):
            if n == 1:
                return [0]

            adj = [[] for i in xrange(n)]
            for i, j in edges:
                adj[i].append(j)
                adj[j].append(i)

            leaves = []
            for i in xrange(n):
                if len(adj[i]) == 1:
                    leaves.append(i)

            while n > 2:
                n -= len(leaves)
                new_leaves = []
                for node in leaves:
                    adj_node = adj[node][0]
                    adj[adj_node].remove(node)
                    if len(adj[adj_node]) == 1:
                        new_leaves.append(adj_node)
                leaves = new_leaves

            return leaves

    """
    1
    []
    2
    [0,1]
    4
    [[1,0],[1,2],[1,3]]
    """



322. coin change
-------------------------------------------

.. code-block:: python

    You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.



    Example 1:
    coins = [1, 2, 5], amount = 11
    return 3 (11 = 5 + 5 + 1)



    Example 2:
    coins = [2], amount = 3
    return -1.



    Note:
    You may assume that you have an infinite number of each kind of coin.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
          for coin in coins:
            if i - coin >= 0:
              dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float("inf") else -1

    =================================================================
    class Solution(object):
        def coinChange(self, coins, amount):
            """
            Very classic dynamic programming problem, like 0-1 Knapsack problem.
            dp[i] is the fewest number of coins making up amount i,
            then for every coin in coins, dp[i] = min(dp[i - coin] + 1).
            """
            dp = [amount + 1] * (amount + 1)
            dp[0] = 0
            for i in xrange(amount + 1):
                for coin in coins:
                    if coin <= i:
                        dp[i] = min(dp[i], dp[i - coin] + 1)
            return -1 if dp[amount] > amount else dp[amount]


    class Solution_2(object):
        def coinChange(self, coins, amount):
            # BFS Way.  Scan the possible tree level by level. More Faster!
            if amount == 0:
                return 0
            amounts = [False] * (amount + 1)
            coins_sum = [0]
            count = 0

            # upper bound on number of coins (+1 to represent the impossible case)
            coins.sort(reverse=True)
            upperBound = amount / coins[-1] + 1

            # Use upperBound to pruning.
            while coins_sum and count < upperBound:
                new_coins_sum = []
                count += 1
                for s in coins_sum:
                    for coin in coins:
                        new_sum = s + coin
                        if new_sum == amount:
                            return count
                        elif new_sum > amount:
                            continue
                        elif not amounts[new_sum]:
                            amounts[new_sum] = True
                            new_coins_sum.append(new_sum)
                        else:
                            pass
                coins_sum = new_coins_sum
            return -1

    """
    [1, 2, 5]
    11
    [1]
    0
    [2]
    3
    """


