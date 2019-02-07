Others - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

220. Contains Duplicate 3
--------------------------------

.. code-block:: python

    Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.



    =================================================================
    import bisect


    class Solution(object):
      def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if k == 0:
          return False
        bst = []
        if k < 0 or t < 0:
          return False
        for i, num in enumerate(nums):
          idx = bisect.bisect_left(bst, num)
          if idx < len(bst) and abs(bst[idx] - num) <= t:
            return True
          if idx > 0 and abs(bst[idx - 1] - num) <= t:
            return True
          if len(bst) >= k:
            del bst[bisect.bisect_left(bst, nums[i - k])]
          bisect.insort(bst, num)
        return False


    =================================================================
    class Solution(object):
        """
        Bucket sort. Refer to:
        https://leetcode.com/discuss/48670/o-n-python-using-buckets-with-explanation-10-lines
        1. Each bucket i save one number, which satisfy val/(t+1) == i.
        2. For each number, the possible candidate can only be
        in the same bucket or the two buckets besides.
        3. Keep as many as k buckets to ensure that the difference is at most k.
        """
        def containsNearbyAlmostDuplicate(self, nums, k, t):
            if t < 0 or k < 1:
                return False
            buckets = {}
            for i, val in enumerate(nums):
                bucket_num = val / (t+1)
                # Find out if there is a satisfied candidate or not.
                for b in range(bucket_num-1, bucket_num+2):
                    if b in buckets and abs(buckets[b] - nums[i]) <= t:
                        return True
                # update the bucket.
                buckets[bucket_num] = nums[i]

                # Remove the bucket which is too far away.
                if len(buckets) > k:
                    del buckets[nums[i - k] / (t+1)]

            return False

        # Intuitively, easy to understand, but time limit exceed.
        """
        def containsNearbyAlmostDuplicate(self, nums, k, t):
            if not nums:
                return False
            len_nums = len(nums)
            for i in range(len_nums-k):
                for j in range(i+1, i+k+1):
                    if abs(nums[i] - nums[j]) <= t:
                        return True
            return False
        """

    """
    []
    3
    0
    [-1,-2,-3,-3]
    1
    0
    [1,3,5,7,1]
    3
    1
    """


229. Majority Element 2
------------------------------

.. code-block:: python

    Given an integer array of size n, find all elements that appear more than &lfloor; n/3 &rfloor; times. The algorithm should run in linear time and in O(1) space.

    =================================================================
    class Solution(object):
      def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) == 0 or nums is None:
          return []
        c1, c2 = None, None
        n1, n2 = 0, 0
        for i in range(0, len(nums)):
          if c1 == nums[i]:
            n1 += 1
          elif c2 == nums[i]:
            n2 += 1
          elif n1 == 0:
            c1 = nums[i]
            n1 += 1
          elif n2 == 0:
            c2 = nums[i]
            n2 += 1
          else:
            n1, n2 = n1 - 1, n2 - 1

        print
        c1, c2

        ret = []
        size = len(nums)
        cn1 = 0
        cn2 = 0
        for i in range(0, len(nums)):
          if nums[i] == c1:
            cn1 += 1
          elif nums[i] == c2:
            cn2 += 1

        if cn1 >= size / 3 + 1:
          ret.append(c1)
        if cn2 >= size / 3 + 1:
          ret.append(c2)
        return ret


    =================================================================
    class Solution(object):
        def majorityElement(self, nums):
            if not nums:
                return []
            candidate_1, candidate_2 = 0, 1
            count_1, count_2 = 0, 0
            for num in nums:
                if num == candidate_1:
                    count_1 += 1
                elif num == candidate_2:
                    count_2 += 1
                elif not count_1:
                    candidate_1, count_1 = num, 1
                elif not count_2:
                    candidate_2, count_2 = num, 1
                else:
                    count_1 -= 1
                    count_2 -= 1
            result = []
            for num in [candidate_1, candidate_2]:
                if nums.count(num) > len(nums) / 3:
                    result.append(num)
            return result
    """
    []
    [0,0,0]
    [1,2,2,3,3,1,1,1]
    [2,2,2,3,3,4,3,2]
    [1,1,2]
    [3,0,3,4]
    """



239. Sliding window Maximum
-----------------------------------

.. code-block:: python


    Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

    For example,
    Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.


    Window position                Max
    ---------------               -----
    [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7


    Therefore, return the max sliding window as [3,3,5,5,6,7].

    Note:
    You may assume k is always valid, ie: 1 &le; k &le; input array's size for non-empty array.

    Follow up:
    Could you solve it in linear time?

    =================================================================
    class Solution(object):
      def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if k == 0:
          return []
        ans = [0 for _ in range(len(nums) - k + 1)]
        stack = collections.deque([])
        for i in range(0, k):
          while stack and nums[stack[-1]] < nums[i]:
            stack.pop()
          stack.append(i)
        ans[0] = nums[stack[0]]
        idx = 0
        for i in range(k, len(nums)):
          idx += 1
          if stack and stack[0] == i - k:
            stack.popleft()
          while stack and nums[stack[-1]] < nums[i]:
            stack.pop()
          stack.append(i)
          ans[idx] = nums[stack[0]]

        return ans


    =================================================================
    from collections import deque


    class Solution(object):
        # Implemented in array, slower than deque
        def maxSlidingWindow(self, nums, k):
            max_num = []
            queue = []
            for i, v in enumerate(nums):
                # remove numbers out of range k
                if queue and queue[0] == i-k:
                    queue = queue[1:]
                # remove smaller numbers in k range as they are useless
                while queue and v > nums[queue[-1]]:
                    queue.pop()
                queue.append(i)
                if i+1 >= k:
                    max_num.append(nums[queue[0]])

            return max_num


    class Solution_2(object):
        # Implemented in dqueue, much faster
        def maxSlidingWindow(self, nums, k):
            max_num = []
            queue = deque()
            for i, v in enumerate(nums):
                if queue and queue[0] == i-k:
                    queue.popleft()
                while queue and v > nums[queue[-1]]:
                    queue.pop()
                queue.append(i)
                if i+1 >= k:
                    max_num.append(nums[queue[0]])

            return max_num
    """
    []
    0
    [1,3,-1,-3,5,3,6,7]
    3
    [1,3,-1,-3,5,3,6,7]
    2
    """



284. Peeking Iterator
-----------------------

.. code-block:: python

    Given an Iterator class interface with methods: next() and hasNext(), design and implement a PeekingIterator that support the peek() operation -- it essentially peek() at the element that will be returned by the next call to next().


    Here is an example. Assume that the iterator is initialized to the beginning of the list: [1, 2, 3].

    Call next() gets you 1, the first element in the list.

    Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.

    You call next() the final time and it returns 3, the last element. Calling hasNext() after that should return false.


    Follow up: How would you extend your design to be generic and work with all types, not just integer?

    Credits:Special thanks to @porker2008 for adding this problem and creating all test cases.

    =================================================================
    class PeekingIterator(object):
      def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self.nextElem = None

      def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.nextElem:
          return self.nextElem
        if self.iter.hasNext():
          self.nextElem = self.iter.next()
        return self.nextElem

      def next(self):
        """
        :rtype: int
        """
        ret = self.nextElem

        if self.nextElem:
          self.nextElem = None
          return ret

        return self.iter.next()

      def hasNext(self):
        """
        :rtype: bool
        """
        return (self.nextElem is not None) or self.iter.hasNext()

    # Your PeekingIterator object will be instantiated and called as such:
    # iter = PeekingIterator(Iterator(nums))
    # while iter.hasNext():
    #     val = iter.peek()   # Get the next element but not advance the iterator.
    #     iter.next()         # Should return the same value as [val].

    =================================================================

    class PeekingIterator(object):
        def __init__(self, iterator):
            self.iter = iterator
            self.temp = self.iter.next() if self.iter.hasNext() else None

        def peek(self):
            return self.temp

        def next(self):
            ret = self.temp
            self.temp = self.iter.next() if self.iter.hasNext() else None
            return ret

        def hasNext(self):
            return self.temp is not None
            # return not self.temp


    # Your PeekingIterator object will be instantiated and called as such:
    # iter = PeekingIterator(Iterator(nums))
    # while iter.hasNext():
    #     val = iter.peek()   # Get the next element but not advance the iterator.
    #     iter.next()         # Should return the same value as [val].



307. Range sum query mutable
---------------------------------

.. code-block:: python

    Given an integer array nums, find the sum of the elements between indices i and j (i &le; j), inclusive.

    The update(i, val) function modifies nums by updating the element at index i to val.

    Example:

    Given nums = [1, 3, 5]

    sumRange(0, 2) -> 9
    update(1, 2)
    sumRange(0, 2) -> 8



    Note:

    The array is only modifiable by the update function.
    You may assume the number of calls to update and sumRange function is distributed evenly.

    =================================================================
    # Segment tree node
    class STNode(object):
      def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None


    class SegmentedTree(object):
      def __init__(self, nums, start, end):
        self.root = self.buildTree(nums, start, end)

      def buildTree(self, nums, start, end):
        if start > end:
          return None

        if start == end:
          node = STNode(start, end)
          node.total = nums[start]
          return node

        mid = start + (end - start) / 2

        root = STNode(start, end)
        root.left = self.buildTree(nums, start, mid)
        root.right = self.buildTree(nums, mid + 1, end)
        root.total = root.left.total + root.right.total
        return root

      def updateVal(self, i, val):
        def updateVal(root, i, val):
          if root.start == root.end:
            root.total = val
            return val
          mid = root.start + (root.end - root.start) / 2
          if i <= mid:
            updateVal(root.left, i, val)
          else:
            updateVal(root.right, i, val)

          root.total = root.left.total + root.right.total
          return root.total

        return updateVal(self.root, i, val)

      def sumRange(self, i, j):
        def rangeSum(root, start, end):
          if root.start == start and root.end == end:
            return root.total

          mid = root.start + (root.end - root.start) / 2
          if j <= mid:
            return rangeSum(root.left, start, end)
          elif i >= mid + 1:
            return rangeSum(root.right, start, end)
          else:
            return rangeSum(root.left, start, mid) + rangeSum(root.right, mid + 1, end)

        return rangeSum(self.root, i, j)


    class NumArray(object):
      def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        self.stTree = SegmentedTree(nums, 0, len(nums) - 1)

      def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        return self.stTree.updateVal(i, val)

      def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.stTree.sumRange(i, j)

    # Your NumArray object will be instantiated and called as such:
    # numArray = NumArray(nums)
    # numArray.sumRange(0, 1)
    # numArray.update(1, 10)
    # numArray.sumRange(1, 2)


    =================================================================
    class NumArray(object):
        """
        1. Binary Indexed Trees.
        Here is the clear explanation about Binary Indexed Tree:
        http://blog.jobbole.com/96430/#
        """
        def __init__(self, nums):
            self.n = len(nums)
            self.nums = nums
            self.sum_tree = [0] * (self.n+1)
            for i in range(self.n):
                self._add(i+1, nums[i])

        def _add(self, i, val):
            while i <= self.n:
                self.sum_tree[i] += val
                i += (i & -i)

        # Get the sum of array nums[0:i], inclusive.
        def _sum(self, i):
            sum_val = 0
            while i > 0:
                sum_val += self.sum_tree[i]
                i -= (i & -i)
            return sum_val

        # Pay attention to the meanning of num & -num.
        # def _lowbit(self, num):
        #    return num & -num

        def update(self, i, val):
            self._add(i+1, val - self.nums[i])
            self.nums[i] = val

        def sumRange(self, i, j):
            if not self.nums:
                return 0
            # sum of elements nums[i..j], inclusive.
            return self._sum(j+1) - self._sum(i)

    """
    if __name__ == '__main__':
        numArray = NumArray([1, 3, 5, 7, 8, 10])
        print numArray.sumRange(0, 4)
        numArray.update(1, 1)
        print numArray.sumRange(1, 3)
    """




329. Longest Increasing Path in a Matrix
--------------------------------------------

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



332. Reconstruct itinerary
-------------------------------

.. code-block:: python

    Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.


    Note:

    If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
    All airports are represented by three capital letters (IATA code).
    You may assume all tickets form at least one valid itinerary.




        Example 1:
        tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
        Return ["JFK", "MUC", "LHR", "SFO", "SJC"].


        Example 2:
        tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
        Return ["JFK","ATL","JFK","SFO","ATL","SFO"].
        Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"]. But it is larger in lexical order.


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    from collections import deque


    class Solution(object):
      def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        graph = {}
        hashset = set([])
        for ticket in tickets:
          graph[ticket[0]] = graph.get(ticket[0], []) + [ticket[1]]

        maxLen = len(tickets) + 1

        for k in graph:
          graph[k] = deque(sorted(graph[k]))

        def dfs(path, graph, maxLen, start):
          if len(path) == maxLen:
            return path + []
          for k in range(0, len(graph.get(start, []))):
            nbr = graph.get(start, [])
            top = nbr.popleft()
            path.append(top)
            ret = dfs(path, graph, maxLen, top)
            if ret:
              return ret
            path.pop()
            nbr.append(top)
          return []

        return dfs(["JFK"], graph, maxLen, "JFK")


    =================================================================
    class Solution(object):
        def findItinerary(self, tickets):
            """ Eulerian path. Hierholzer Algorithm, greedy DFS with backtracking.

            Refer to: Short Ruby / Python / Java / C++
            https://discuss.leetcode.com/topic/36370/short-ruby-python-java-c
            """
            import collections
            targets = collections.defaultdict(list)
            for a, b in sorted(tickets, reverse=True):
                targets[a] += b,
            route = []

            def visit(airport):
                while targets[airport]:
                    visit(targets[airport].pop())
                route.append(airport)
            visit('JFK')

            return route[::-1]

    """
    [["JFK", "MUC"], ["JFK", "SJC"], ["SJC", "JFK"], ["MUC", "ATL"]]
    [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
    [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    [["JFK", "MUC"], ["MUC", "SJC"], ["SJC", "ATL"], ["MUC", "LHR"], ["LHR", "SJC"]]
    """



