Hash table - Easy 2
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

1. Two Sum
--------------------

.. code-block:: python

    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.


    Example:

    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].


    =================================================================
    class Solution(object):
      def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}
        for i, num in enumerate(nums):
          if target - num in d:
            return [d[target - num], i]
          d[num] = i
        # no special case handling because it's assumed that it has only one solution


    =================================================================

    class Solution_2(object):
        # Hashtable
        def twoSum(self, nums, target):
            nums_dict = {}
            for index1, number1 in enumerate(nums):
                number2 = target - number1
                if number2 in nums_dict:
                    return nums_dict[number2] + 1, index1 + 1
                nums_dict[number1] = index1

    """
    [1,2]
    3
    [3,2,4]
    6
    """


3. Longest substring without repeating charater
--------------------------------------------------

.. code-block:: python

    Given a string, find the length of the longest substring without repeating characters.

    Examples:

    Given "abcabcbb", the answer is "abc", which the length is 3.

    Given "bbbbb", the answer is "b", with the length of 1.

    Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.


    =================================================================
    class Solution(object):
      def _lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = collections.defaultdict(int)
        l = ans = 0
        for i, c in enumerate(s):
          while l > 0 and d[c] > 0:
            d[s[i - l]] -= 1
            l -= 1
          d[c] += 1
          l += 1
          ans = max(ans, l)
        return ans

      def lengthOfLongestSubstring(self, s):
        d = {}
        start = 0
        ans = 0
        for i, c in enumerate(s):
          if c in d:
            start = max(start, d[c] + 1)
          d[c] = i
          ans = max(ans, i - start + 1)
        return ans



    =================================================================
    class Solution(object):
        def lengthOfLongestSubstring(self, s):
            """
            :type s: str
            :rtype: int
            """

            max_length = 0
            start = 0   # Start index of the substring without repeating characters
            end = 0     # End index of the substring without repeating characters
            char_dict = {}

            for index in range(len(s)):
                char = s[index]
                # Find out a repeating character. So reset start and end.
                if char in char_dict and start <= char_dict[char] <= end:
                    start = char_dict[char] + 1
                    end = index
                # char is not in the substring already, add it to the substring.
                else:
                    end = index
                    if end - start + 1 > max_length:
                        max_length = end - start + 1
                char_dict[char] = index

            return max_length

    """
    ""
    "bbbbb"
    "abcabcbb"
    """


36. Valid Sudoku
--------------------

.. code-block:: python


    Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.

    The Sudoku board could be partially filled, where empty cells are filled with the character '.'.



    A partially filled sudoku which is valid.


    Note:
    A valid Sudoku board (partially filled) is not necessarily solvable. Only the filled cells need to be validated.


    =================================================================
    class Solution(object):
      def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        cacheCol = [[0] * 9 for _ in range(0, 10)]
        cacheRow = [[0] * 9 for _ in range(0, 10)]
        cacheBox = [[0] * 9 for _ in range(0, 10)]

        for i in range(0, 9):
          for j in range(0, 9):
            ib = (i / 3) * 3 + j / 3
            if board[i][j] == ".":
              continue
            num = int(board[i][j]) - 1
            if cacheRow[i][num] != 0 or cacheCol[j][num] != 0 or cacheBox[ib][num] != 0:
              return False
            cacheRow[i][num] = 1
            cacheCol[j][num] = 1
            cacheBox[ib][num] = 1
        return True


    =================================================================
    class Solution(object):
        def isValidSudoku(self, board):
            # check for rows
            for row in board:
                row_hash = {}
                for c in row:
                    if c != "." and c in row_hash:
                        return False
                    row_hash[c] = 1

            # check for cols
            for i in range(9):
                col_hash = {}
                for row in board:
                    if row[i] != "." and row[i] in col_hash:
                        return False
                    col_hash[row[i]] = 1

            # check for panel
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    count = 0
                    panel_hash = {}
                    while(count < 9):
                        c = board[i + count // 3][j + count % 3]
                        count += 1
                        if c != "." and c in panel_hash:
                            return False
                        panel_hash[c] = 1

            return True

    """
    ["..4...63.",".........","5......9.","...56....","4.3.....1","...7.....","...5.....",".........","........."]
    [".87654321","2........","3........","4........","5........","6........","7........","8........","9........"]
    """



49. Group Anagrams
--------------------

.. code-block:: python

    Given an array of strings, group anagrams together.


    For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
    Return:

    [
      ["ate", "eat","tea"],
      ["nat","tan"],
      ["bat"]
    ]

    Note: All inputs will be in lower-case.


    =================================================================
    class Solution(object):
      def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        def hash(count):
          p1, p2 = 2903, 29947
          ret = 0
          for c in count:
            ret = ret * p1 + c
            p1 *= p2
          return ret

        d = {}

        for str in strs:
          count = [0] * 26
          for c in str:
            count[ord(c) - ord('a')] += 1
          key = hash(count)
          if key not in d:
            d[key] = [str]
          else:
            d[key].append(str)
        return [d[k] for k in d]



    =================================================================
    class Solution(object):
        def groupAnagrams(self, strs):
            """Hash tables: use sorted(word) as key.

            Note that list is unhashable type, so we need to change sorted
            str to tuple, which is hashable type.
            """
            d = {}
            for w in sorted(strs):
                key = tuple(sorted(w))
                d[key] = d.get(key, []) + [w]
            return d.values()

    """
    [""]
    ["aaa", "aaa", "aa", "bb"]
    ["a", "b", "c", "d"]
    """


128. Longest Consecutive Sequence
--------------------------------------

.. code-block:: python

    class Solution(object):
        def groupAnagrams(self, strs):
            """Hash tables: use sorted(word) as key.

            Note that list is unhashable type, so we need to change sorted
            str to tuple, which is hashable type.
            """
            d = {}
            for w in sorted(strs):
                key = tuple(sorted(w))
                d[key] = d.get(key, []) + [w]
            return d.values()

    """
    [""]
    ["aaa", "aaa", "aa", "bb"]
    ["a", "b", "c", "d"]
    """


    =================================================================
    class Solution(object):
      def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 0
        s = set(nums)
        for num in nums:
          if num in s:
            s.discard(num)
            cnt = 1
            right = num + 1
            left = num - 1
            while left in s:
              s.discard(left)
              cnt += 1
              left -= 1
            while right in s:
              s.discard(right)
              cnt += 1
              right += 1
            ans = max(ans, cnt)
        return ans



    =================================================================
    class Solution(object):
        def longestConsecutive(self, nums):
            """
            Build a hash to find whether a num in nums or not in O(1) time.
            """
            nums_dict = {num: False for num in nums}
            max_length = 0
            for num in nums:
                if nums_dict[num]:
                    continue

                # Find the post consecutive number
                next_num = num + 1
                while next_num in nums_dict:
                    nums_dict[next_num] = True
                    next_num += 1

                # Find the pre consecutive number
                pre_num = num - 1
                while pre_num in nums_dict:
                    nums_dict[pre_num] = True
                    pre_num -= 1

                max_length = max(next_num-pre_num-1, max_length)

            return max_length

    """
    []
    [0]
    [100, 4, 200, 1, 3, 2]
    [2147483646,-2147483647,0,2,2147483644,-2147483645,2147483645]
    """



146. LRU Cache
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
    import collections


    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            # An OrderedDict is a dictionary subclass
            # that remembers the order in which its contents are added.
            self.cache = collections.OrderedDict()

        def get(self, key):
            if key not in self.cache:
                return -1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        def set(self, key, value):
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) == self.capacity:
                self.cache.popitem(last=False)
            else:
                pass
            self.cache[key] = value

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



149. Max Points on a line
------------------------------------

.. code-block:: python

    Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.


    =================================================================
    class Solution(object):
      def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """

        def gcd(a, b):
          while b:
            a, b = b, a % b
          return a

        ans = 1
        d = {}
        points.sort(key=lambda p: (p.x, p.y))
        for i in range(0, len(points)):
          if i > 0 and (points[i].x, points[i].y) == (points[i - 1].x, points[i - 1].y):
            continue
          overlap = 1
          for j in range(i + 1, len(points)):
            x1, y1 = points[i].x, points[i].y
            x2, y2 = points[j].x, points[j].y
            ku, kd = y2 - y1, x2 - x1
            if (x1, y1) != (x2, y2):
              kg = gcd(ku, kd)
              ku /= kg
              kd /= kg
              d[(ku, kd, x1, y1)] = d.get((ku, kd, x1, y1), 0) + 1
            else:
              overlap += 1
              ans = max(ans, overlap)
            ans = max(ans, d.get((ku, kd, x1, y1), 0) + overlap)
        return min(ans, len(points))



    =================================================================
    class Solution(object):
        def maxPoints(self, points):
            if not points:
                return 0
            # Record all the duplicate points
            replicate = {}
            for point in points:
                replicate[(point.x, point.y)] = replicate.get(
                    (point.x, point.y), 0) + 1

            # Get all the different nodes
            diff_points = replicate.keys()
            diff_count = len(diff_points)
            if diff_count == 1:
                return replicate[diff_points[0]]

            maxPoints = 0
            # Get all the different slope's point numbers.
            for i in xrange(diff_count-1):
                slopes = {}
                slope = 0
                for j in range(i+1, diff_count):
                    dx = diff_points[i][0] - diff_points[j][0]
                    dy = diff_points[i][1] - diff_points[j][1]
                    if dx == 0:
                        slope = "#"
                    elif dy == 0:
                        slope = 0
                    else:
                        slope = float(dy) / dx
                    slopes[slope] = (slopes.get(slope, 0) +
                                     replicate[diff_points[j]])

                maxPoints = max(maxPoints,
                                max(slopes.values())+replicate[diff_points[i]])

            return maxPoints

    """
    []
    [[1,1]]
    [[1,1],[2,2],[1,1],[1,1],[2,2],[2,3]]
    """



187. Repeated DNA Sequences
---------------------------------

.. code-block:: python

    All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.

    Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.


    For example,

    Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",

    Return:
    ["AAAAACCCCC", "CCCCCAAAAA"].



    =================================================================
    class Solution(object):
      def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        d = {}
        ans = []
        for i in range(len(s) - 9):
          key = s[i:i + 10]
          if key in d:
            d[key] += 1
            if d[key] == 2:
              ans.append(key)
          else:
            d[key] = 1
        return ans


    =================================================================
    class Solution(object):
        def findRepeatedDnaSequences(self, s):
            str_hash = {}
            sequence = []
            len_s = len(s)
            for i in range(len_s-9):
                cur_str = s[i:i+10]
                str_hash[cur_str] = str_hash.get(cur_str, 0) + 1
                if str_hash[cur_str] == 2:
                    sequence.append(cur_str)
            return sequence


    """
    "AAA"
    "AAAAAAAAAA"
    "AAAAAAAAAAA"
    "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    """




