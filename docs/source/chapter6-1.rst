Combination - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode


30. substring with concatenaton of all words
-------------------------------------------------

.. code-block:: python

    You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.



    For example, given:
    s: "barfoothefoobarman"
    words: ["foo", "bar"]



    You should return the indices: [0,9].
    (order does not matter).

    =================================================================
    from collections import deque


    class Solution(object):
      def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        if len(words) > len(s):
          return []
        d = {}
        t = {}
        ans = []
        deq = deque([])
        wl = len(words[0])
        fullscore = 0
        for word in words:
          d[word] = d.get(word, 0) + 1
          fullscore += 1

        for i in range(0, len(s)):
          head = start = i
          t.clear()
          score = 0

          while start + wl <= len(s) and s[start:start + wl] in d:
            cword = s[start:start + wl]
            t[cword] = t.get(cword, 0) + 1
            if t[cword] <= d[cword]:
              score += 1
            else:
              break
            start += wl

          if score == fullscore:
            ans.append(head)

        return ans


    =================================================================
    class Solution(object):
        """ Easy to think, but is very slow.

        Use an unordered_map<string, int> counts to record the expected times of each word and
        another unordered_map<string, int> seen to record the times we have seen
        """

        def findSubstring(self, s, words):
            if not s or not words:
                return []

            word_cnt = {}
            for w in words:
                word_cnt[w] = word_cnt.get(w, 0) + 1

            s_len, word_l = len(s), len(words[0])
            concatenation_l = len(words) * word_l
            ans = []
            for i in range(s_len - concatenation_l + 1):
                candidate_map = {}
                j = 0
                while j < len(words):
                    w = s[i + j * word_l: i + (j + 1) * word_l]
                    if w not in word_cnt:
                        break
                    candidate_map[w] = candidate_map.get(w, 0) + 1
                    if candidate_map.get(w, 0) > word_cnt[w]:
                        break
                    j += 1

                if j == len(words):
                    ans.append(i)

            return ans


    class Solution_2(object):
        """ Use hashmap and two point.

        Travel all the words combinations to maintain a slicing window.
        There are wl(word len) times travel, each time n/wl words:
        mostly 2 times travel for each word:
            one left side of the window, the other right side of the window
        So, time complexity O(wl * 2 * N/wl) = O(2N)
        Refer to:
        https://discuss.leetcode.com/topic/6617/an-o-n-solution-with-detailed-explanation
        """
        def findSubstring(self, s, words):
            if not s or not words:
                return []

            word_cnt = {}
            for w in words:
                word_cnt[w] = word_cnt.get(w, 0) + 1

            s_len, w_len, cnt = len(s), len(words[0]), len(words)
            i = 0
            ans = []
            while i < w_len:
                left, count = i, 0
                candidate_cnt = {}
                for j in range(i, s_len, w_len):
                    cur_str = s[j: j + w_len]
                    if cur_str in word_cnt:
                        candidate_cnt[cur_str] = candidate_cnt.get(cur_str, 0) + 1
                        count += 1
                        if candidate_cnt[cur_str] <= word_cnt[cur_str]:
                            pass
                        else:
                            # A more word, advance the window left side possiablly
                            while candidate_cnt[cur_str] > word_cnt[cur_str]:
                                left_str = s[left: left + w_len]
                                candidate_cnt[left_str] -= 1
                                left += w_len
                                count -= 1

                        # come to a result
                        if count == cnt:
                            ans.append(left)
                            candidate_cnt[s[left:left + w_len]] -= 1
                            count -= 1
                            left += w_len
                    # not a valid word, clear the window.
                    else:
                        candidate_cnt = {}
                        left = j + w_len
                        count = 0
                i += 1
            return ans


    class Solution_Fail(object):
        """ Pythonic way, easy to think, but Time Limit Exceeded.

        Use two hash-map.
        """
        def findSubstring(self, s, words):
            if not s or not words:
                return []
            import collections
            word_cnt = collections.Counter(words)
            s_len, word_l = len(s), len(words[0])
            concatenation_l = len(words) * word_l
            ans = []
            for i in range(s_len - concatenation_l + 1):
                candidate_str = s[i:i + concatenation_l]
                split_str = [candidate_str[j:j + word_l]
                             for j in range(0, concatenation_l, word_l)]
                candidate_cnt = collections.Counter(split_str)
                if not (word_cnt - candidate_cnt):
                    ans.append(i)
            return ans

    """
    ""
    []
    "barfoothefoobarman"
    ["foo", "bar"]
    "barfoofoobarthefoobarman"
    ["bar","foo","the"]
    """


37. sudoku solver
--------------------

.. code-block:: python

    Write a program to solve a Sudoku puzzle by filling the empty cells.

    Empty cells are indicated by the character '.'.

    You may assume that there will be only one unique solution.



    A sudoku puzzle...




    ...and its solution numbers marked in red.

    =================================================================
    class Solution(object):
      def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        cacheBox = [[0] * len(board) for _ in range(len(board))]
        cacheRow = [[0] * len(board) for _ in range(len(board))]
        cacheCol = [[0] * len(board) for _ in range(len(board))]

        def helper(board, i, j, cacheRow, cacheCol, cacheBox):
          if board[i][j] == ".":
            for k in range(1, 10):
              if i < 0 or i >= len(board) or j < 0 or j >= len(board):
                continue
              ib = (i / 3) * 3 + j / 3
              if cacheRow[i][k - 1] == 1 or cacheCol[j][k - 1] == 1 or cacheBox[ib][k - 1] == 1:
                continue

              cacheRow[i][k - 1] = cacheCol[j][k - 1] = cacheBox[ib][k - 1] = 1
              board[i][j] = str(k)
              if i == j == len(board) - 1:
                return True
              if i + 1 < len(board):
                if helper(board, i + 1, j, cacheRow, cacheCol, cacheBox):
                  return True
              elif j + 1 < len(board):
                if helper(board, 0, j + 1, cacheRow, cacheCol, cacheBox):
                  return True
              board[i][j] = "."
              cacheRow[i][k - 1] = cacheCol[j][k - 1] = cacheBox[ib][k - 1] = 0
          else:
            if i == j == len(board) - 1:
              return True
            if i + 1 < len(board):
              if helper(board, i + 1, j, cacheRow, cacheCol, cacheBox):
                return True
            elif j + 1 < len(board):
              if helper(board, 0, j + 1, cacheRow, cacheCol, cacheBox):
                return True
          return False

        for i in range(len(board)):
          for j in range(len(board)):
            if board[i][j] != ".":
              ib = (i / 3) * 3 + j / 3
              k = int(board[i][j]) - 1
              cacheRow[i][k] = cacheCol[j][k] = cacheBox[ib][k] = 1
        print
        helper(board, 0, 0, cacheRow, cacheCol, cacheBox)


    =================================================================
    class Solution(object):
        nums_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        def solveSudoku(self, board):
            """ Hash and Backtracking.

            :type board: List[List[str]]
            :rtype: void Do not return anything, modify board in-place instead.
            """
            # Pay Attention, can not define two-degree array as: [[0]*9]*9
            self.rows_hash, self.cols_hash = [[0] * 9 for i in range(9)], [[0] * 9 for i in range(9)]
            self.panel_hash = [[0] * 9 for i in range(9)]

            # Add all existing number to hash.
            for i in xrange(9):
                for j in xrange(9):
                    if board[i][j] != ".":
                        self.try_num(int(board[i][j]) - 1, i, j)

            self.dfs_search(0, board)

        def dfs_search(self, cur, board):
            if cur == 81:
                return True
            r, c = cur / 9, cur % 9

            # The existing number must be valid, because we are promised that
            # there will be only one unique solution.
            if board[r][c] != ".":
                return self.dfs_search(cur + 1, board)

            else:
                for n in self.nums_list:
                    if self.try_num(n - 1, r, c):
                        board[r][c] = str(n)
                        if self.dfs_search(cur + 1, board):
                            return True
                        # Remember to bacrtrack here.
                        board[r][c] = "."
                        self.backtrack(n - 1, r, c)
                return False

        def try_num(self, num, row, col):
            panel_pos = row / 3 * 3 + col / 3
            if (self.rows_hash[row][num] or self.cols_hash[col][num] or
                    self.panel_hash[panel_pos][num]):
                return False
            else:
                self.rows_hash[row][num] = 1
                self.cols_hash[col][num] = 1
                self.panel_hash[panel_pos][num] = 1
                return True

        def backtrack(self, num, row, col):
            panel_pos = row / 3 * 3 + col / 3
            self.rows_hash[row][num] = 0
            self.cols_hash[col][num] = 0
            self.panel_hash[panel_pos][num] = 0

    """
    ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
    ["53..7....","6..195...",".98....6.","8...6...3","4..8.3..1","7...2...6",".6....28.","...419..5","....8..79"]
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

146. lru cache
--------------------

.. code-block:: python

    Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.


    get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
    put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.


    Follow up:
    Could you do both operations in O(1) time complexity?

    Example:

    LRUCache cache = new LRUCache( 2 /* capacity */ );

    cache.put(1, 1);
    cache.put(2, 2);
    cache.get(1);       // returns 1
    cache.put(3, 3);    // evicts key 2
    cache.get(2);       // returns -1 (not found)
    cache.put(4, 4);    // evicts key 1
    cache.get(1);       // returns -1 (not found)
    cache.get(3);       // returns 3
    cache.get(4);       // returns 4

    =================================================================
    class List(object):
      @staticmethod
      def delete(elem):
        elem.prev.next = elem.next
        elem.next.prev = elem.prev
        return elem

      @staticmethod
      def move(elem, newPrev, newNext):
        elem.prev = newPrev
        elem.next = newNext
        newPrev.next = elem
        newNext.prev = elem

      @staticmethod
      def append(head, elem):
        List.move(elem, head.prev, head)

      @staticmethod
      def isEmpty(head):
        return head.next == head.prev == head

      @staticmethod
      def initHead(head):
        head.prev = head.next = head


    class Node(object):
      def __init__(self, key, value, head):
        self.key = key
        self.value = value
        self.head = head
        self.prev = self.next = None

      def hit(self):
        List.delete(self)
        List.append(self.head, self)


    class LRUCache(object):
      def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.d = {}
        self.cap = capacity
        self.head = Node(-1, -1, None)
        List.initHead(self.head)

      def get(self, key):
        """
        :rtype: int
        """
        if key not in self.d:
          return -1
        self.d[key].hit()
        return self.d[key].value

      def set(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: nothing
        """
        if self.cap == 0:
          return

        if key in self.d:
          self.d[key].hit()
          self.d[key].value = value
        else:
          if len(self.d) >= self.cap:
            oldNode = List.delete(self.head.next)
            del self.d[oldNode.key]

          newNode = Node(key, value, self.head)
          List.append(self.head, newNode)
          self.d[key] = newNode


    =================================================================
    class LRUCache(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}
            self.doubleLinkedList = DoubleLinkedList()
            self.len = 0

        def get(self, key):
            # Get operator will also update the linked list(Don't forget)
            if key in self.cache:
                node = self.cache[key]
                self.doubleLinkedList.delete(node)
                self.doubleLinkedList.append(node)
                return self.cache[key].value
            else:
                return -1

        def set(self, key, value):
            # update the (key,value) pair in both hash and linked list.
            if key in self.cache:
                node = self.cache[key]
                self.doubleLinkedList.delete(node)
                new_node = Node(key, value)
                self.doubleLinkedList.append(new_node)
                self.cache[key] = new_node

            else:
                node = Node(key, value)
                # Add the new node to cache
                if self.len < self.capacity:
                    self.doubleLinkedList.append(node)
                    self.cache[key] = node
                    self.len += 1
                # Remove the head of linked list and append the new node
                else:
                    replaced_node = self.doubleLinkedList.del_head()
                    del self.cache[replaced_node.key]
                    self.doubleLinkedList.append(node)
                    self.cache[key] = node


    class Node:
        def __init__(self, key=None, value=None, next_node=None, pre_node=None):
            self.key = key
            self.value = value
            self.next = next_node
            self.pre = pre_node


    # Double linked list
    class DoubleLinkedList:
        def __init__(self):
            self.head = None
            self.tail = None

        def append(self, node):
            if not self.head:
                self.head = node
                self.tail = self.head
            else:
                self.tail.next = node
                node.pre = self.tail
                self.tail = node

        def delete(self, node):
            if self.head == self.tail:
                self.head, self.tail = None, None
            elif node == self.head:
                node.next.pre = None
                self.head = node.next
            elif node == self.tail:
                node.pre.next = None
                self.tail = node.pre
            else:
                node.pre.next = node.next
                node.next.pre = node.pre

        def del_head(self):
            del_head = self.head
            self.delete(self.head)
            return del_head

    """
    if __name__ == '__main__':
        ca = LRUCache(2)
        ca.set(2, 1)
        print "AA", ca.get(2)
        ca.set(2, 2)
        print "BB",  ca.get(2)
        ca.set(3, 3)
        print "CC", ca.get(3)
        # what if: print "CC", ca.get(2)
        ca.set(4, 1)
        print "CC", ca.get(2)
    """


300. longest increasing subsequence
--------------------------------------

.. code-block:: python

    Given an unsorted array of integers, find the length of longest increasing subsequence.


    For example,
    Given [10, 9, 2, 5, 3, 7, 101, 18],
    The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.


    Your algorithm should run in O(n2) complexity.


    Follow up: Could you improve it to O(n log n) time complexity?

    Credits:Special thanks to @pbrother for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tail = []
        for num in nums:
          idx = bisect.bisect_left(tail, num)
          if idx == len(tail):
            tail.append(num)
          else:
            tail[idx] = num
        return len(tail)


    =================================================================
    class Solution(object):
        """
        Clear explanation is here:
        https://leetcode.com/discuss/67687/c-o-nlogn-solution-with-explainations-4ms
        https://leetcode.com/discuss/67643/java-python-binary-search-o-nlogn-time-with-explanation

        The key to the solution is: build a ladder for numbers: dp.
        dp[i]: the smallest num of all increasing subsequences with length i+1.
        When a new number x comes, compare it with the number in each level:
            1. If x is larger than all levels, append it, increase the size by 1
            2. If dp[i-1] < x <= dp[i], update dp[i] with x.

        For example, say we have nums = [4,5,6,3],
        then all the available increasing subsequences are:

        len = 1: [4], [5], [6], [3]   => dp[0] = 3
        len = 2: [4, 5], [5, 6]       => dp[1] = 5
        len = 3: [4, 5, 6]            => dp[2] = 6
        """
        def lengthOfLIS(self, nums):
            dp = [0] * len(nums)
            size = 0
            for n in nums:
                # Binary search here.
                left, right = 0, size
                while left < right:
                    mid = (left + right) / 2
                    if dp[mid] < n:
                        left = mid + 1
                    else:
                        right = mid
                # Append the next number
                dp[right] = n
                # Update size
                if right == size:
                    size += 1

            return size

    """
    []
    [3]
    [1,1,1,1]
    [10,9,2,5,3,7,101,18]
    """



324. wiggle sort 2
--------------------

.. code-block:: python


    Given an unsorted array nums, reorder it such that
    nums[0] < nums[1] > nums[2] < nums[3]....



    Example:
    (1) Given nums = [1, 5, 1, 1, 6, 4], one possible answer is [1, 4, 1, 5, 1, 6].
    (2) Given nums = [1, 3, 2, 2, 3, 1], one possible answer is [2, 3, 1, 3, 1, 2].



    Note:
    You may assume all input has valid answer.



    Follow Up:
    Can you do it in O(n) time and/or in-place with O(1) extra space?


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    import random


    class Solution(object):
      def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 2:
          nums.sort()
          return
        numscopy = nums + []
        mid = self.quickselect(0, len(nums) - 1, nums, len(nums) / 2 - 1)
        ans = [mid] * len(nums)
        if len(nums) % 2 == 0:
          l = len(nums) - 2
          r = 1
          for i in range(0, len(nums)):
            if nums[i] < mid:
              ans[l] = nums[i]
              l -= 2
            elif nums[i] > mid:
              ans[r] = nums[i]
              r += 2
        else:
          l = 0
          r = len(nums) - 2
          for i in range(0, len(nums)):
            if nums[i] < mid:
              ans[l] = nums[i]
              l += 2
            elif nums[i] > mid:
              ans[r] = nums[i]
              r -= 2
        for i in range(0, len(nums)):
          nums[i] = ans[i]

      def quickselect(self, start, end, A, k):
        if start == end:
          return A[start]

        mid = self.partition(start, end, A)

        if mid == k:
          return A[k]
        elif mid > k:
          return self.quickselect(start, mid - 1, A, k)
        else:
          return self.quickselect(mid + 1, end, A, k)

      def partition(self, start, end, A):
        left, right = start, end
        pivot = A[left]
        while left < right:
          while left < right and A[right] <= pivot:
            right -= 1
          A[left] = A[right]
          while left < right and A[left] >= pivot:
            left += 1
          A[right] = A[left]
        A[left] = pivot
        return left


    =================================================================
    class Solution(object):
        def wiggleSort(self, nums):
            """ Sort needed.
            Sort the array(small to big), and cut into two parts:
                For even size, left half size==right half size,
                For odd size,  left half size==right half size+1.
                (smaller part there may be one more number.)

            Then put the smaller half of the numbers on the even indexes,
            and the larger half on the odd indexes.
            Here iterate from the back of two halves,
            so that the duplicates between two parts can be split apart.

            Clear solutionm, explanation and proof can be found here:
            https://leetcode.com/discuss/76965/3-lines-python-with-explanation-proof
            """
            nums.sort()
            # half = len(nums[::2]) or half = (len(nums) + 1) // 2
            # nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]
            half = len(nums[::2]) - 1
            nums[::2], nums[1::2] = nums[half::-1], nums[:half:-1]


    class Solution_2(object):
        def wiggleSort(self, nums):
            """ O(n)-time O(1)-space solution, no sort here.

            Find the kth smallest element, where k is the half the size (if size is even)
            or half the size+1 (if size is odd).

            Then do a three-way-partition, so that they can be split in two parts.
            Number in left parts <= those in right parts and the duplicates are around median.

            Then put the smaller half of the numbers on the even indexes,
            and the larger half on the odd indexes.
            Here iterate from the back of two halves,
            so that the duplicates between two parts can be split apart.

            According to:
            https://leetcode.com/discuss/77133/o-n-o-1-after-median-virtual-indexing
            https://discuss.leetcode.com/topic/38189/clear-java-o-n-avg-time-o-n-space-solution-using-3-way-partition
            """
            mid = len(nums[::2])
            mid_val = self.findKthLargest(nums, mid)
            self.three_way_partition(nums, mid_val)

            nums[::2], nums[1::2] = nums[mid - 1::-1], nums[:mid - 1:-1]

        def three_way_partition(self, nums, mid_val):
            """ Dutch national flag problem.

            Refer to:
            https://en.wikipedia.org/wiki/Dutch_national_flag_problem
            """
            i, j, n = 0, 0, len(nums) - 1
            while j <= n:
                if nums[j] < mid_val:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
                elif nums[j] > mid_val:
                    nums[n], nums[j] = nums[j], nums[n]
                    n -= 1
                else:
                    j += 1

        def findKthLargest(self, nums, k):
            """ Can be done in O(logn) with partition.  Here use built-in heap method.
            """
            import heapq
            return heapq.nsmallest(k, nums)[-1]

    """
    [4, 5, 5, 6]
    [1, 5, 1, 1, 6, 4]
    [1, 3, 2, 2, 3, 1]
    """


329. longest increasing path in a matrix
------------------------------------------

.. code-block:: python

    Given an integer matrix, find the length of the longest increasing path.


    From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).


    Example 1:

    nums = [
      [9,9,4],
      [6,6,8],
      [2,1,1]
    ]




    Return 4

    The longest increasing path is [1, 2, 6, 9].


    Example 2:

    nums = [
      [3,4,5],
      [3,2,6],
      [2,2,1]
    ]




    Return 4

    The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.

    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]


    class Solution(object):
      def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """

        def dfs(matrix, i, j, visited, cache):
          if (i, j) in visited:
            return visited[(i, j)]

          ret = 0
          for di, dj in directions:
            p, q = i + di, j + dj
            if p < 0 or q < 0 or p >= len(matrix) or q >= len(matrix[0]):
              continue
            if (p, q) not in cache and matrix[p][q] > matrix[i][j]:
              cache.add((p, q))
              r = dfs(matrix, p, q, visited, cache)
              ret = max(ret, r)
              cache.discard((p, q))

          visited[(i, j)] = ret + 1
          return ret + 1

        visited = {}
        cache = set()
        ans = 0
        for i in range(0, len(matrix)):
          for j in range(0, len(matrix[0])):
            cache.add((i, j))
            ans = max(ans, dfs(matrix, i, j, visited, cache))
            cache.discard((i, j))
        return ans


    =================================================================
    class Solution(object):
        def longestIncreasingPath(self, matrix):
            """
            According to:
            https://leetcode.com/discuss/81747/python-solution-memoization-dp-288ms
            1. Do DFS from every cell
            2. Compare every 4 direction and skip unmatched cells.
            3. Get matrix max from every cell's max
            4. Use matrix[x][y] <= matrix[i][j] so we don't need a visited[m][n] array
            The key is to cache the distance because it's frequently to revisit a cell
            """
            def dfs(i, j):
                if not dp[i][j]:
                    val = matrix[i][j]
                    dp[i][j] = 1 + max(
                        dfs(i - 1, j) if i and val > matrix[i - 1][j] else 0,
                        dfs(i + 1, j) if i < M - 1 and val > matrix[i + 1][j] else 0,
                        dfs(i, j - 1) if j and val > matrix[i][j - 1] else 0,
                        dfs(i, j + 1) if j < N - 1 and val > matrix[i][j + 1] else 0)
                return dp[i][j]

            if not matrix or not matrix[0]:
                return 0
            M, N = len(matrix), len(matrix[0])
            dp = [[0] * N for i in range(M)]
            return max(dfs(x, y) for x in range(M) for y in range(N))

    """
    [[]]
    [[3,4,5],[3,2,6],[2,2,1]]
    [[9,9,4],[6,6,8],[2,1,1]]
    """



355. Design twitter
--------------------

.. code-block:: python

    Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:



    postTweet(userId, tweetId): Compose a new tweet.
    getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
    follow(followerId, followeeId): Follower follows a followee.
    unfollow(followerId, followeeId): Follower unfollows a followee.



    Example:

    Twitter twitter = new Twitter();

    // User 1 posts a new tweet (id = 5).
    twitter.postTweet(1, 5);

    // User 1's news feed should return a list with 1 tweet id -> [5].
    twitter.getNewsFeed(1);

    // User 1 follows user 2.
    twitter.follow(1, 2);

    // User 2 posts a new tweet (id = 6).
    twitter.postTweet(2, 6);

    // User 1's news feed should return a list with 2 tweet ids -> [6, 5].
    // Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
    twitter.getNewsFeed(1);

    // User 1 unfollows user 2.
    twitter.unfollow(1, 2);

    // User 1's news feed should return a list with 1 tweet id -> [5],
    // since user 1 is no longer following user 2.
    twitter.getNewsFeed(1);


    =================================================================
    import heapq


    class Twitter(object):

      def __init__(self):
        """
        Initialize your data structure here.
        """
        self.ts = 0
        self.tweets = collections.defaultdict(list)
        self.friendship = collections.defaultdict(set)

      def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        tInfo = self.ts, tweetId, userId, len(self.tweets[userId])
        self.tweets[userId].append(tInfo)
        self.ts -= 1

      def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        ret = []
        heap = []
        if self.tweets[userId]:
          heapq.heappush(heap, self.tweets[userId][-1])

        for followeeId in self.friendship[userId]:
          if self.tweets[followeeId]:
            heapq.heappush(heap, self.tweets[followeeId][-1])
        cnt = 10
        while heap and cnt > 0:
          _, tid, uid, idx = heapq.heappop(heap)
          ret.append(tid)
          if idx > 0:
            heapq.heappush(heap, self.tweets[uid][idx - 1])
          cnt -= 1
        return ret

      def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId == followeeId:
          return
        self.friendship[followerId] |= {followeeId}

      def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.friendship[followerId] -= {followeeId}

    # Your Twitter object will be instantiated and called as such:
    # obj = Twitter()
    # obj.postTweet(userId,tweetId)
    # param_2 = obj.getNewsFeed(userId)
    # obj.follow(followerId,followeeId)
    # obj.unfollow(followerId,followeeId)


    =================================================================
    class Twitter(object):
        """
        Accordting to:
        https://discuss.leetcode.com/topic/47838/python-solution
        """
        def __init__(self):
            self.timer = itertools.count(step=-1)
            self.tweets = collections.defaultdict(collections.deque)
            self.followees = collections.defaultdict(set)

        def postTweet(self, userId, tweetId):
            """Compose a new tweet.
            """
            self.tweets[userId].appendleft((next(self.timer), tweetId))

        def getNewsFeed(self, userId):
            """Retrieve the 10 most recent tweet ids in the user's news feed.

            Each item in the news feed must be posted by users who the user
            followed or by the user herself.
            Tweets must be ordered from most recent to least recent.
            """
            tweets = heapq.merge(*(self.tweets[u] for u in
                                  (self.followees[userId] | {userId})))
            return [t for _, t in itertools.islice(tweets, 10)]

        def follow(self, followerId, followeeId):
            """Follower follows a followee. If the operation is invalid, it should be a no-op.
            """
            self.followees[followerId].add(followeeId)

        def unfollow(self, followerId, followeeId):
            """Follower unfollows a followee. If the operation is invalid, it should be a no-op.
            """
            self.followees[followerId].discard(followeeId)


    # Your Twitter object will be instantiated and called as such:
    # obj = Twitter()
    # obj.postTweet(userId,tweetId)
    # param_2 = obj.getNewsFeed(userId)
    # obj.follow(followerId,followeeId)
    # obj.unfollow(followerId,followeeId)

    """
    ["Twitter","postTweet","postTweet","getNewsFeed","postTweet","getNewsFeed"]
    [[],[1,5],[1,3],[3,5],[1,6],[3,5,6]]
    ["Twitter","postTweet","getNewsFeed","follow","postTweet","getNewsFeed","unfollow","getNewsFeed"]
    [[],[1,5],[1],[1,2],[2,6],[1],[1,2],[1]]
    """


