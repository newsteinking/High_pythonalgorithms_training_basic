Array - Easy 2
=======================================




`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

31. Next Permutation
-------------------------------

.. code-block:: python


    Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.


    If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).


    The replacement must be in-place, do not allocate extra memory.


    Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
    1,2,3 &#8594; 1,3,2
    3,2,1 &#8594; 1,2,3
    1,1,5 &#8594; 1,5,1


    class Solution(object):
        def nextPermutation(self, nums):
            length = len(nums)
            index = length - 1

            """
            Scan from the end of nums and get nums[index],
            find one pair which nums[mark] > nums[mark - 1],
            then swap the smallest number in nums[mark:] and nums[mark - 1].
            Finally sort nums[mark:] and we will slove the problem.
            """
            while index >= 1:
                if nums[index] > nums[index - 1]:
                    for i in range(length - 1, index - 1, -1):
                        if nums[i] > nums[index - 1]:
                            nums[i], nums[index - 1] = nums[index - 1], nums[i]
                            nums[index:] = sorted(nums[index:])
                            return
                else:
                    index -= 1

            # Nums is in descending order, just reverse it.
            nums.reverse()

    """
    []
    [1]
    [1,2,3]
    [3,2,1]
    [1,1,2,2,4,5,5]
    """


    class Solution(object):
      def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if nums is None or len(nums) <= 1:
          return

        pos = None
        p = len(nums) - 2
        # find the first number that is not in correct order
        while p >= 0:
          if nums[p + 1] > nums[p]:
            pos = p
            break
          p -= 1

        if pos is None:
          self.reverse(nums, 0, len(nums) - 1)
          return

        # find the min value in the rest of the array
        minPos, minV = pos + 1, nums[pos + 1]
        for i in range(pos + 1, len(nums)):
          if nums[i] <= minV and nums[i] > nums[pos]:
            minV = nums[i]
            minPos = i
        # swap the two above number and reverse the array from `pos`
        nums[pos], nums[minPos] = nums[minPos], nums[pos]
        self.reverse(nums, pos + 1, len(nums) - 1)

      def reverse(self, nums, start, end):
        while start < end:
          nums[start], nums[end] = nums[end], nums[start]
          start += 1
          end -= 1


41. First Missing Positive
-------------------------------

.. code-block:: python


    Given an unsorted integer array, find the first missing positive integer.



    For example,
    Given [1,2,0] return 3,
    and [3,4,-1,1] return 2.



    Your algorithm should run in O(n) time and uses constant space.


    class Solution(object):
      def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        while i < len(nums):
          if 0 < nums[i] <= len(nums) and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
          else:
            i += 1

        for i in range(0, len(nums)):
          if nums[i] != i + 1:
            return i + 1
        return len(nums) + 1


    class Solution(object):
        def firstMissingPositive(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            # Put all i+1 in nums[i]
            nums_len = len(nums)
            for i in range(nums_len):
                # Swap nums[i] to the appropriate position until current
                # nums[i] can't be push to the list, which is <0 or >nums_len
                # By the way, pay attention to situation as [1,1].
                while nums[i] != i + 1 and 0 < nums[i] <= nums_len:
                    index = nums[i] - 1
                    if nums[index] == nums[i]:
                        break
                    nums[i], nums[index] = nums[index], nums[i]
                    # nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]

            for i in range(nums_len):
                if nums[i] != i + 1:
                    return i + 1

            return nums_len + 1

    """
    []
    [1,2,0]
    [3,4,-1,1]
    [3,4,-1,1,2,2,0,12,3]
    """



54. Spiral Matrix
-------------------------------

.. code-block:: python

    Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.



    For example,
    Given the following matrix:


    [
     [ 1, 2, 3 ],
     [ 4, 5, 6 ],
     [ 7, 8, 9 ]
    ]


    You should return [1,2,3,6,9,8,7,4,5].


    class Solution(object):
      def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
          return []
        ans = []
        left, up, down, right = 0, 0, len(matrix) - 1, len(matrix[0]) - 1
        while left <= right and up <= down:
          for i in range(left, right + 1):
            ans += matrix[up][i],
          up += 1
          for i in range(up, down + 1):
            ans += matrix[i][right],
          right -= 1
          for i in reversed(range(left, right + 1)):
            ans += matrix[down][i],
          down -= 1
          for i in reversed(range(up, down + 1)):
            ans += matrix[i][left],
          left += 1
        return ans[:(len(matrix) * len(matrix[0]))]


    class Solution(object):
        def spiralOrder(self, matrix):
            """
            :type matrix: List[List[int]]
            :rtype: List[int]
            """
            if not matrix:
                return []

            m_row = len(matrix)
            n_col = len(matrix[0])
            min_m_n = min(m_row, n_col)

            spiral_order = []
            step = 0
            while step < (min_m_n + 1) / 2:
                horizontal_len = n_col - 1 - 2 * step
                vertical_len = m_row - 1 - 2 * step
                # print "step.._ |", step, horizontal_len, vertical_len

                # Add the current up edge to spiral order.
                if vertical_len == 0 and horizontal_len > 0:
                    horizontal_len += 1
                for i in range(horizontal_len):
                    spiral_order.append(matrix[step][i + step])

                # Add the current right edge to spiral order.
                if horizontal_len == 0 and vertical_len > 0:
                    vertical_len += 1
                for i in range(vertical_len):
                    spiral_order.append(matrix[i + step][n_col - 1 - step])

                if vertical_len > 0:
                    # Add the current down edge to spiral order.
                    for i in range(horizontal_len):
                        spiral_order.append(
                            matrix[m_row - 1 - step][n_col - 1 - step - i])

                if horizontal_len > 0:
                    # Add the current left edge to spiral order.
                    for i in range(vertical_len):
                        spiral_order.append(
                            matrix[m_row - 1 - step - i][step])

                step += 1

            # For N * N matrix, where N is an odd number.
            if vertical_len == horizontal_len == 0 and m_row == n_col:
                spiral_order.append(matrix[m_row / 2][n_col / 2])

            return spiral_order


    """
    []
    [[1]]
    [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
    [[1],[2],[3]]
    [[2,5],[8,4],[0,-1]]
    [[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]
    """







56. Merge Intervals
-------------------------------

.. code-block:: python

    Given a collection of intervals, merge all overlapping intervals.


    For example,
    Given [1,3],[2,6],[8,10],[15,18],
    return [1,6],[8,10],[15,18].


    class Solution(object):
      def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        ans = []
        for intv in sorted(intervals, key=lambda x: x.start):
          if ans and ans[-1].end >= intv.start:
            ans[-1].end = max(ans[-1].end, intv.end)
          else:
            ans.append(intv)
        return ans


    class Solution(object):
        def merge(self, intervals):
            """
            :type intervals: List[Interval]
            :rtype: List[Interval]
            """
            merged_list = []
            length = len(intervals)
            intervals.sort(key=lambda interval: interval.start)
            i = 0

            # Scan every interval and merge the overlapping intervals.
            while i < length:
                j = i + 1
                while j < length and intervals[j].start <= intervals[i].end:
                    intervals[i].start = min(intervals[i].start,
                                             intervals[j].start)
                    intervals[i].end = max(intervals[i].end,
                                           intervals[j].end)
                    j += 1

                merged_list.append(intervals[i])
                i = j

            return merged_list

    """
    []
    [[1,4],[4,5]]
    [[1,3],[2,6],[8,10],[15,18]]
    [[12,13],[1,3],[5,8],[2,6],[6,7]]
    """





57. Insert Intervals
-------------------------------

.. code-block:: python

    Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

    You may assume that the intervals were initially sorted according to their start times.


    Example 1:
    Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].



    Example 2:
    Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].



    This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].


    class Solution(object):
      def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        s, e = newInterval.start, newInterval.end
        left = filter(lambda x: x.end < newInterval.start, intervals)
        right = filter(lambda x: x.start > newInterval.end, intervals)
        if left + right != intervals:
          s = min(intervals[len(left)].start, s)
          e = max(intervals[~len(right)].end, e)
        return left + [Interval(s, e)] + right


    class Solution(object):
        def insert(self, intervals, newInterval):
            """
            :type intervals: List[Interval]
            :type newInterval: Interval
            :rtype: List[Interval]
            """
            merged_list = []
            length = len(intervals)

            # Insert the newInterval to the right position
            index = 0
            while index < length:
                if intervals[index].start >= newInterval.start:
                    intervals.insert(index, newInterval)
                    break
                index += 1
            if index == length:
                intervals.append(newInterval)

            i = 0
            length += 1
            # Scan every interval and merge the overlapping intervals.
            while i < length:
                j = i + 1
                while j < length and intervals[j].start <= intervals[i].end:
                    intervals[i].start = min(intervals[i].start,
                                             intervals[j].start)
                    intervals[i].end = max(intervals[i].end,
                                           intervals[j].end)
                    j += 1

                merged_list.append(intervals[i])
                i = j

            return merged_list

    """
    []
    [5,7]
    [[1,4],[6,8],[7,8]]
    [3,10]
    [[1,5]]
    [6,8]
    """





59. Spiral Matrix 2
-------------------------------

.. code-block:: python

    Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.


    For example,
    Given n = 3,

    You should return the following matrix:

    [
     [ 1, 2, 3 ],
     [ 8, 9, 4 ],
     [ 7, 6, 5 ]
    ]

    class Solution(object):
      def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        ans = [[0] * n for _ in range(n)]
        left, right, up, down = 0, n - 1, 0, n - 1
        k = 1
        while left <= right and up <= down:
          for i in range(left, right + 1):
            ans[up][i] = k
            k += 1
          up += 1
          for i in range(up, down + 1):
            ans[i][right] = k
            k += 1
          right -= 1
          for i in reversed(range(left, right + 1)):
            ans[down][i] = k
            k += 1
          down -= 1
          for i in reversed(range(up, down + 1)):
            ans[i][left] = k
            k += 1
          left += 1
        return ans

    class Solution(object):
        def generateMatrix(self, n):
            """
            :type n: int
            :rtype: List[List[int]]
            """
            if not n:
                return []

            matrix = [[-1 for row in range(n)] for col in range(n)]
            current_num = 1
            step = 0
            while step < n / 2:
                edge_len = n - 1 - 2 * step

                # Get number from left to right(up edge)
                for i in range(edge_len):
                    matrix[step][i + step] = current_num
                    current_num += 1

                # Get number from up to down(right edge)
                for i in range(edge_len):
                    matrix[i + step][n - 1 - step] = current_num
                    current_num += 1

                # Get number from right to left(down edge)
                for i in range(edge_len):
                    matrix[n - 1 - step][n - 1 - step - i] = current_num
                    current_num += 1

                # Get number from down to up(left edge)
                for i in range(edge_len):
                    matrix[n - 1 - step - i][step] = current_num
                    current_num += 1
                step += 1

            if n % 2 == 1:
                matrix[n/2][n/2] = current_num
            return matrix

    """
    0
    1
    3
    4
    """




73. Set Matrix Zeroes
-------------------------------

.. code-block:: python


    Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.


    click to show follow up.

    Follow up:


    Did you use extra space?
    A straight forward solution using O(mn) space is probably a bad idea.
    A simple improvement uses O(m + n) space, but still not the best solution.
    Could you devise a constant space solution?


    class Solution(object):
      def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        colZeroFlag = False
        for i in range(0, len(matrix)):
          if matrix[i][0] == 0:
            colZeroFlag = True
          for j in range(1, len(matrix[0])):
            if matrix[i][j] == 0:
              matrix[i][0] = matrix[0][j] = 0

        for i in reversed(range(0, len(matrix))):
          for j in reversed(range(1, len(matrix[0]))):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
              matrix[i][j] = 0
          if colZeroFlag:
            matrix[i][0] = 0


    class Solution(object):
        def setZeroes(self, matrix):
            """
            :type matrix: List[List[int]]
            :rtype: void Do not return anything, modify matrix in-place instead.
            """
            if not matrix:
                return []

            m = len(matrix)
            n = len(matrix[0])

            # Frist, make sure whether first row and first col is all 0.
            first_row = False
            for i in range(n):
                if matrix[0][i] == 0:
                    first_row = True
            first_col = False
            for j in range(m):
                if matrix[j][0] == 0:
                    first_col = True

            # Keep the information about the 0 cell to first row and first col.
            for row in range(1, m):
                for col in range(1, n):
                    if matrix[row][col] == 0:
                        matrix[row][0] = 0
                        matrix[0][col] = 0

            # Set 0s according to the information in first row and first col
            for row in range(m):
                for col in range(n):
                    if not matrix[row][0] or not matrix[0][col]:
                        matrix[row][col] = 0

            # Set the first row and first col
            if first_row:
                for col in range(n):
                    matrix[0][col] = 0
            if first_col:
                for row in range(m):
                    matrix[row][0] = 0

    """
    [[0]]
    [[1,0],[2,2]]
    [[0,0,0,5],[4,3,1,4],[0,1,1,4],[1,2,1,3],[0,0,1,1]]
    """

