DivideConquer - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

215. Kth Largest Element Array
-----------------------------------

.. code-block:: python


    Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

    For example,
    Given [3,2,1,5,6,4] and k = 2, return 5.


    Note:
    You may assume k is always valid, 1 &le; k &le; array's length.

    Credits:Special thanks to @mithmatt for adding this problem and creating all test cases.
    =================================================================
    import random


    class Solution(object):
      def findKthLargest(self, nums, k):
        """
        :type A: List[int]
        :type k: int
        :rtype: int
        """

        def quickselect(start, end, nums, k):
          if start == end:
            return nums[start]

          mid = partition(start, end, nums)

          if mid == k:
            return nums[mid]
          elif k > mid:
            return quickselect(mid + 1, end, nums, k)
          else:
            return quickselect(start, mid - 1, nums, k)

        def partition(start, end, nums):
          p = random.randrange(start, end + 1)
          pv = nums[p]
          nums[end], nums[p] = nums[p], nums[end]
          mid = start
          for i in range(start, end):
            if nums[i] >= pv:
              nums[i], nums[mid] = nums[mid], nums[i]
              mid += 1
          nums[mid], nums[end] = nums[end], nums[mid]
          return mid

        ret = quickselect(0, len(nums) - 1, nums, k - 1)
        return ret

      def partition(start, end, nums):
        p = random.randrange(start, end + 1)
        pv = nums[p]
        nums[end], nums[p] = nums[p], nums[end]
        mid = start
        for i in range(start, end):
          if nums[i] >= pv:
            nums[i], nums[mid] = nums[mid], nums[i]
            mid += 1
        nums[mid], nums[end] = nums[end], nums[mid]
        return mid


    =================================================================
    class Solution(object):
        # Simple way: O(nlogn)
        def findKthLargest(self, nums, k):
            return sorted(nums)[-k]


    class Solution_2(object):
        # QuickSelect, according to:
        # http://www.cs.yale.edu/homes/aspnes/pinewiki/QuickSelect.html
        # Heap implement by c++ can be found in c++ version.
        def findKthLargest(self, nums, k):
            pivot = nums[0]
            nums1, nums2 = [], []
            for num in nums:
                if num > pivot:
                    nums1.append(num)
                elif num < pivot:
                    nums2.append(num)
            if k <= len(nums1):
                return self.findKthLargest(nums1, k)
            elif k > len(nums) - len(nums2):
                return self.findKthLargest(nums2, k - (len(nums) - len(nums2)))
            else:
                return pivot

    """
    [1]
    1
    [3,2,1,5,6,4]
    2
    [1,2,1,3,9]
    2
    """



240. Search 2D Matrix 2
-----------------------------------

.. code-block:: python


    Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:



    Integers in each row are sorted in ascending from left to right.
    Integers in each column are sorted in ascending from top to bottom.




    For example,

    Consider the following matrix:


    [
      [1,   4,  7, 11, 15],
      [2,   5,  8, 12, 19],
      [3,   6,  9, 16, 22],
      [10, 13, 14, 17, 24],
      [18, 21, 23, 26, 30]
    ]


    Given target = 5, return true.
    Given target = 20, return false.
    =================================================================
    class Solution(object):
      def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        def binarySearch(nums, target):
          start, end = 0, len(nums) - 1
          while start + 1 < end:
            mid = start + (end - start) / 2
            if nums[mid] > target:
              end = mid
            elif nums[mid] < target:
              start = mid
            else:
              return True
          if nums[start] == target:
            return True
          if nums[end] == target:
            return True
          return False

        for nums in matrix:
          if binarySearch(nums, target):
            return True
        return False


    =================================================================
    class Solution(object):
        """
        O(m+n)
        Check the top-right corner.
        If it's not the target, then remove the top row or rightmost column.
        """
        def searchMatrix(self, matrix, target):
            if not matrix or len(matrix[0]) < 1:
                return False
            m, n = len(matrix), len(matrix[0])

            # We start search the matrix from top right corner
            # Initialize the current position to top right corner.
            row, col = 0, n - 1
            while row < m and col >= 0:
                if matrix[row][col] == target:
                    return True
                elif matrix[row][col] > target:
                    col -= 1
                else:
                    row += 1
            return False


    class Solution_2(object):
        # O(m+n): same as the pre solution, more efficient and pythonic.
        # According to
        # https://leetcode.com/discuss/47571/4-lines-c-6-lines-ruby-7-lines-python-1-liners
        def searchMatrix(self, matrix, target):
            if not matrix or len(matrix[0]) < 1:
                return False
            n = len(matrix[0])
            col = -1
            for row in matrix:
                while col + n > 0 and row[col] > target:
                    col -= 1
                if row[col] == target:
                    return True
            return False


    class Solution_3(object):
        # O(mn): 1 lines python. Just for fun
        def searchMatrix(self, matrix, target):
            return any(target in row for row in matrix)

    """
    [[]]
    0
    [[-5]]
    -2
    [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24]]
    12
    """



241. Different Ways To Add Parenthese
---------------------------------------------

.. code-block:: python


    Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and *.

    Example 1
    Input: "2-1-1".
    ((2-1)-1) = 0
    (2-(1-1)) = 2
    Output: [0, 2]

    Example 2
    Input: "2*3-4*5"
    (2*(3-(4*5))) = -34
    ((2*3)-(4*5)) = -14
    ((2*(3-4))*5) = -10
    (2*((3-4)*5)) = -10
    (((2*3)-4)*5) = 10
    Output: [-34, -14, -10, -10, 10]

    Credits:Special thanks to @mithmatt for adding this problem and creating all test cases.
    =================================================================
    from operator import *


    class Solution(object):
      def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        ops = {"+": add, "-": sub, "*": mul, "/": div}
        ans = []
        for i, c in enumerate(input):
          if c in ops:
            left = self.diffWaysToCompute(input[:i])
            right = self.diffWaysToCompute(input[i + 1:])
            ans.extend([ops[c](a, b) for a in left for b in right])
        return ans if ans else [int(input)]

    =================================================================
    class Solution(object):
        """
        Recursive way: easy to understand.  The key idea for this solution is:
        each operator in this string could be the last operator to be operated.
        We just iterator over all these cases.
        """

        def diffWaysToCompute(self, input):
            if input.isdigit():
                return [int(input)]

            res = []
            for i in xrange(len(input)):
                if input[i] in "+-*":
                    res_left = self.diffWaysToCompute(input[:i])
                    res_right = self.diffWaysToCompute(input[i + 1:])
                    for left in res_left:
                        for right in res_right:
                            res.append(self.computer(left, right, input[i]))
            return res

        def computer(self, m, n, op):
            if op == "+":
                return m + n
            elif op == "-":
                return m - n
            else:
                return m * n


    class Solution_2(object):
        # Use cache to avoid repeating subquestions in recursive way.
        def diffWaysToCompute(self, input):
            self.cache = {}
            return self.computerWithCache(input)

        def computerWithCache(self, input):
            if input.isdigit():
                self.cache[input] = [int(input)]
                return [int(input)]

            res = []
            for i in xrange(len(input)):
                if input[i] in "+-*":
                    left_str = input[:i]
                    res_left = (self.cache[left_str] if left_str in self.cache
                                else self.computerWithCache(input[:i]))
                    right_str = input[i + 1:]
                    res_right = (self.cache[right_str] if right_str in self.cache
                                 else self.computerWithCache(input[i + 1:]))

                    for left in res_left:
                        for right in res_right:
                            res.append(self.computer(left, right, input[i]))
            self.cache[input] = res
            return res

        def computer(self, m, n, op):
            if op == "+":
                return m + n
            elif op == "-":
                return m - n
            else:
                return m * n

    """
    "0"
    "2-1-1"
    "2*3-4*5"
    "3-6*7+8-12*1"
    """

