BinarySearch - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode


4. Median of two sorted arrays
------------------------------------------

.. code-block:: python

    There are two sorted arrays nums1 and nums2 of size m and n respectively.

    Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

    Example 1:

    nums1 = [1, 3]
    nums2 = [2]

    The median is 2.0



    Example 2:

    nums1 = [1, 2]
    nums2 = [3, 4]

    The median is (2 + 3)/2 = 2.5
    =================================================================
    class Solution(object):
      def findMedianSortedArrays(self, nums1, nums2):
        a, b = sorted((nums1, nums2), key=len)
        m, n = len(a), len(b)
        after = (m + n - 1) / 2
        lo, hi = 0, m
        while lo < hi:
          i = (lo + hi) / 2
          if after - i - 1 < 0 or a[i] >= b[after - i - 1]:
            hi = i
          else:
            lo = i + 1
        i = lo
        nextfew = sorted(a[i:i + 2] + b[after - i:after - i + 2])
        return (nextfew[0] + nextfew[1 - (m + n) % 2]) / 2.0


    =================================================================
    class Solution(object):
        """ Divide and Conquer inspired by find k-th number in sorted array.

        The complexity is of course O(log(M+N)).
        Similiar with the following answer except without slicing.
        https://discuss.leetcode.com/topic/6947/intuitive-python-o-log-m-n-solution-by-kth-smallest-in-the-two-sorted-arrays-252ms
        """

        def findMedianSortedArrays(self, nums1, nums2):
            n1, n2 = len(nums1), len(nums2)
            length = n1 + n2

            if length & 0x1:
                return self.find_kth_num(nums1, 0, n1, nums2, 0, n2, (length + 1) / 2)
            else:
                return (self.find_kth_num(nums1, 0, n1, nums2, 0, n2, length / 2) +
                        self.find_kth_num(nums1, 0, n1, nums2, 0, n2, length / 2 + 1)) / 2.0

        def find_kth_num(self, list1, begin1, end1, list2, begin2, end2, k):
            """ Find the kth number in two sorted list: list1 , list2

            Binary search as followers:
            Firstly cut list1 and list2 into two parts by t1 and t2, respectively.
                1. lis1_left ... list1[t1-th] ... list1_right,
                2. lis2_left ... list2[t2-th] ... list2_right
            Then compare value of list1[t1-th] and list2[t2-th] in list2.
            Three situations about the relation between list1[t1-th] and list2[t2-th]:
                1.  <  Equal the (k-t1)th number in list1_right and list_2 left.
                2.  >  Equal the (k-t2)th number in list1_left and list_2 right.
                3. ==  Find the k-th number.
            """
            n1, n2 = end1 - begin1, end2 - begin2

            # Make sure the first list is always shorter than the second
            if n1 > n2:
                return self.find_kth_num(list2, begin2, end2, list1, begin1, end1, k)
            if n1 == 0:
                return list2[begin2 + k - 1]
            if k == 1:
                return min(list1[begin1], list2[begin2])

            # Get the next search interval
            t1 = min(k / 2, n1)
            t2 = k - t1
            if list1[begin1 + t1 - 1] < list2[begin2 + t2 - 1]:
                return self.find_kth_num(list1, begin1 + t1, end1, list2, begin2, begin2 + t2, k - t1)
            elif list1[begin1 + t1 - 1] > list2[begin2 + t2 - 1]:
                return self.find_kth_num(list1, begin1, begin1 + t1, list2, begin2 + t2, end2, k - t2)
            else:
                return list1[begin1 + t1 - 1]


    """
    []
    [1]
    [1,3]
    [2]
    [1]
    [2,3,4,5,6]
    [2,3,4]
    [5,6,7]
    """


    """
    Excellent explanation can be found here:
    https://discuss.leetcode.com/topic/4996/share-my-o-log-min-m-n-solution-with-explanation

    In statistics, the median is used for dividing a set into two equal length subsets,
    that one subset is always greater than the other.

    First let's cut A into two parts at a random position i:

          left_A             |        right_A
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
    Since A has m elements, so there are m+1 kinds of cutting( i = 0 ~ m ).
    And we know: len(left_A) = i, len(right_A) = m - i .
    Note: when i = 0 , left_A is empty, and when i = m , right_A is empty.

    With the same way, cut B into two parts at a random position j:

          left_B             |        right_B
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]

    Put left_A and left_B into one set, and put right_A and right_B into another set.
    Let's name them left_part and right_part :

          left_part          |        right_part
    A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]
    B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]

    If we can ensure:

    1) len(left_part) == len(right_part)
    2) max(left_part) <= min(right_part)

    then we divide all elements in {A, B} into two parts with equal length,
    and one part is always greater than the other.
    Then median = (max(left_part) + min(right_part))/2.

    To ensure these two conditions, we just need to ensure:

    (1) i + j == m - i + n - j (or: m - i + n - j + 1)
        if n >= m, we just need to set: i = 0 ~ m, j = (m + n + 1)/2 - i
    (2) B[j-1] <= A[i] and A[i-1] <= B[j]
    (For simplicity, I presume A[i-1],B[j-1],A[i],B[j] are
    always valid even if i=0/i=m/j=0/j=n .
    I will talk about how to deal with these edge values at last.)

    So, all we need to do is:

    Searching i in [0, m], to find an object `i` that:
        B[j-1] <= A[i] and A[i-1] <= B[j], ( where j = (m + n + 1)/2 - i )

    When the object i is found, the median is:
        max(A[i-1], B[j-1]) (when m + n is odd)
        or (max(A[i-1], B[j-1]) + min(A[i], B[j]))/2 (when m + n is even)
    """



33. Search In Rotated Sorted Array
------------------------------------------

.. code-block:: python

    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

    You are given a target value to search. If found in the array return its index, otherwise return -1.

    You may assume no duplicate exists in the array.
    =================================================================
    class Solution(object):
      def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
          return -1
        left = 0
        right = len(nums) - 1
        while left <= right:
          mid = (right + left) / 2
          if nums[mid] == target:
            return mid
          if nums[mid] >= nums[left]:
            if nums[left] <= target <= nums[mid]:
              right = mid - 1
            else:
              left = mid + 1
          else:
            if nums[mid] <= target <= nums[right]:
              left = mid + 1
            else:
              right = mid - 1
        return -1


    =================================================================
    class Solution(object):
        def search(self, nums, target):
            nums_size = len(nums)
            start = 0
            end = nums_size - 1
            while start <= end:
                mid = (start + end) / 2
                num_mid = nums[mid]

                # Mid is in the left part of the rotated(if it's rotated) array.
                if num_mid >= nums[start]:
                    if nums[start] <= target < num_mid:
                        end = mid - 1
                    elif num_mid == target:
                        return mid
                    else:
                        start = mid + 1

                # The array must be rotated, and mid is in the right part
                else:
                    if num_mid < target <= nums[end]:
                        start = mid + 1
                    elif target == num_mid:
                        return mid
                    else:
                        end = mid - 1

            return -1

    """
    []
    0
    [1]
    1
    [8,11,13,1,3,4,5,7]
    7
    [4,5,6,7,8,1,2,3]
    8
    [5, 1, 3]
    1
    """




34. Search for a range
------------------------------------------

.. code-block:: python

    Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

    Your algorithm's runtime complexity must be in the order of O(log n).

    If the target is not found in the array, return [-1, -1].


    For example,
    Given [5, 7, 7, 8, 8, 10] and target value 8,
    return [3, 4].

    =================================================================
    class Solution(object):
      def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        l, r = 0, len(nums) - 1
        found = 0
        start, end = 0, 0
        while l < r:
          m = l + (r - l) / 2
          if target > nums[m]:
            l = m + 1
          else:
            if target == nums[m]:
              found += 1
            r = m - 1

        if nums[l] == target:
          found += 1

        start = r
        if nums[r] != target or r < 0:
          start = r + 1

        l, r = 0, len(nums) - 1
        while l < r:
          m = l + (r - l) / 2
          if target < nums[m]:
            r = m - 1
          else:
            if target == nums[m]:
              found += 1
            l = m + 1
        end = l
        if nums[l] != target:
          end = l - 1

        if found == 0:
          return [-1, -1]
        return [start, end]


    =================================================================
    class Solution(object):
        # log(n) here.
        def firstAppear(self, nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) / 2
                if target == nums[mid] and mid - 1 >= left and target == nums[mid - 1]:
                    right = mid - 1
                elif target == nums[mid]:
                    return mid
                elif target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1

        # log(n) again.
        def lastAppear(set, nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) / 2
                if target == nums[mid] and mid + 1 <= right and target == nums[mid + 1]:
                    left = mid + 1
                elif target == nums[mid]:
                    return mid
                elif target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1

        def searchRange(self, nums, target):
            return (self.firstAppear(nums, target), self.lastAppear(nums, target))

    """
    []
    0
    [1,1,1,1]
    1
    [1,2,3,4,5]
    3
    [1,2,3,4,5]
    6
    """



35. Search In Position
------------------------------------------

.. code-block:: python

    Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You may assume no duplicates in the array.


    Here are few examples.
    [1,3,5,6], 5 &#8594; 2
    [1,3,5,6], 2 &#8594; 1
    [1,3,5,6], 7 &#8594; 4
    [1,3,5,6], 0 &#8594; 0

    =================================================================
    class Solution(object):
      def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lo = 0
        hi = len(nums)
        while lo < hi:
          mid = lo + (hi - lo) / 2
          if nums[mid] > target:
            hi = mid
          elif nums[mid] < target:
            lo = mid + 1
          else:
            return mid
        return lo


    =================================================================
    class Solution(object):
        # Pythonic way.
        def searchInsert(self, nums, target):
            return len([x for x in nums if x < target])


    class Solution_2(object):
        def searchInsert(self, nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) / 2
                if target == nums[mid]:
                    return mid

                elif target > nums[mid]:
                    left = mid + 1

                else:
                    right = mid - 1

            return left

    """
    [1,3,5,6]
    5
    [1,3,5,6]
    2
    [1,3,5,6]
    7
    [1,3,5,6]
    0
    """



69. sqrtx
------------------------------------------

.. code-block:: python

    Implement int sqrt(int x).

    Compute and return the square root of x.
    =================================================================
    class Solution(object):
      def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        lo = 0
        hi = x
        while lo <= hi:
          mid = (hi + lo) / 2
          v = mid * mid
          if v < x:
            lo = mid + 1
          elif v > x:
            hi = mid - 1
          else:
            return mid
        return hi

    =================================================================
    class Solution(object):
        # Binary search.
        def mySqrt(self, x):
            low, high = 0, x
            while low <= high:
                mid = (low + high) / 2
                if mid ** 2 <= x < (mid + 1) ** 2:
                    return mid
                elif mid ** 2 > x:
                    high = mid - 1
                else:
                    low = mid + 1


    class Solution_2(object):
        # Newton iterative method
        # According to:
        # http://www.matrix67.com/blog/archives/361
        def mySqrt(self, x):
            if not x:
                return x
            val = x
            sqrt_x = (val + x * 1.0 / val) / 2.0
            while val - sqrt_x > 0.001:
                val = sqrt_x
                sqrt_x = (val + x * 1.0 / val) / 2.0

            return int(sqrt_x)


    class Solution_3(object):
        # Shorter Newton method.
        def mySqrt(self, x):
            val = x
            while val * val > x:
                val = (val + x / val) / 2
            return val

    """
    0
    1
    15
    90
    1010
    """



74. Search a 2D matrix
------------------------------------------

.. code-block:: python

    Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:



    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.




    For example,

    Consider the following matrix:


    [
      [1,   3,  5,  7],
      [10, 11, 16, 20],
      [23, 30, 34, 50]
    ]


    Given target = 3, return true.
    =================================================================
    class Solution(object):
      def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
          return False

        m = len(matrix)
        n = len(matrix[0])

        start, end = 0, m * n - 1
        while start + 1 < end:
          mid = start + (end - start) / 2
          if matrix[mid / n][mid % n] > target:
            end = mid
          elif matrix[mid / n][mid % n] < target:
            start = mid
          else:
            return True
        if matrix[start / n][start % n] == target:
          return True
        if matrix[end / n][end % n] == target:
          return True
        return False

    =================================================================
    class Solution(object):
        # Don't treat it as a 2D matrix, just treat it as a sorted list
        def searchMatrix(self, matrix, target):
            if not matrix:
                return False

            # Classic binary search: O(logmn)
            m_rows, n_cols = len(matrix), len(matrix[0])
            left, right = 0, m_rows * n_cols - 1

            while left <= right:
                mid = (left+right) / 2
                num = matrix[mid / n_cols][mid % n_cols]
                if num > target:
                    right = mid - 1
                elif num < target:
                    left = mid + 1
                else:
                    return True

            return False

    """
    [[]]
    0
    [[1]]
    0
    [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 50]]
    34
    [[1, 3, 5], [10, 11, 16], [23, 30, 34]]
    46
    """



81. Search In rotated sorted array 2
------------------------------------------

.. code-block:: python

    Follow up for "Search in Rotated Sorted Array":
    What if duplicates are allowed?

    Would this affect the run-time complexity? How and why?


    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

    Write a function to determine if a given target is in the array.

    The array may contain duplicates.
    =================================================================
    class Solution(object):
      def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        start, end = 0, len(nums) - 1
        while start + 1 < end:
          mid = start + (end - start) / 2
          if nums[mid] == target:
            return True
          if nums[start] < nums[mid]:
            if nums[start] <= target <= nums[mid]:
              end = mid
            else:
              start = mid
          elif nums[start] > nums[mid]:
            if nums[mid] <= target <= nums[end]:
              start = mid
            else:
              end = mid
          else:
            start += 1

        if nums[start] == target:
          return True
        if nums[end] == target:
          return True
        return False

    =================================================================
    class Solution(object):
        def search(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: bool
            """

            nums_size = len(nums)
            start = 0
            end = nums_size - 1

            while start <= end:
                mid = (start + end) / 2
                num_mid = nums[mid]

                # Mid is in the left part of the rotated(if it's rotated) array.
                if num_mid > nums[start]:
                    if nums[start] <= target < num_mid:
                        end = mid - 1
                    elif target == num_mid:
                        return True
                    else:
                        start = mid + 1

                # The array must be rotated, and mid is in the right part
                elif num_mid < nums[start]:
                    if num_mid < target <= nums[end]:
                        start = mid + 1
                    elif target == num_mid:
                        return True
                    else:
                        end = mid - 1

                # Can't make sure whether mid in the left part or right part.
                else:
                    # Find the target.
                    if target == num_mid:
                        return True
                    # Just add start with one until we can make sure.
                    # Of course, you can also minus end with one.
                    start += 1

            return False

    """
    []
    0
    [1]
    1
    [7,8,7,7,7]
    8
    [7,7,7,8,8]
    8
    """


153. find minimum in rotated sorted array
------------------------------------------

.. code-block:: python

    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

    Find the minimum element.

    You may assume no duplicate exists in the array.
    =================================================================
    class Solution(object):
      def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start, end, mid = 0, len(nums) - 1, 0
        while start + 1 < end:
          mid = start + (end - start) / 2
          if nums[start] <= nums[mid]:
            start = mid
          else:
            end = mid
        return min(nums[0], nums[start], nums[end])

    =================================================================
    class Solution(object):
        def findMin(self, nums):
            # assert(nums)
            left = 0
            right = len(nums) - 1
            # Make sure right is always in the right rotated part.
            # Left can be either in the left part or the minimum part.
            # So, when left and right is the same finally, we find the minimum.
            while left < right:
                # When there is no rotate, just return self.nums[start]
                if nums[left] < nums[right]:
                    return nums[left]

                mid = (left + right) / 2
                # mid is in the left part, so move the left point to mid+1.
                # finally left will reach to the minimum element.
                if nums[left] <= nums[mid]:
                    left = mid + 1
                else:
                    right = mid
            return nums[left]

    """
    [1]
    [1,2]
    [3,4,2]
    [7,8,9,0,2,4,5]
    """


154. find minimum in rotated sorted array 2
----------------------------------------------------

.. code-block:: python

    Follow up for "Find Minimum in Rotated Sorted Array":
    What if duplicates are allowed?

    Would this affect the run-time complexity? How and why?


    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

    (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

    Find the minimum element.

    The array may contain duplicates.
    =================================================================
    class Solution(object):
      def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = nums[0]
        start, end = 0, len(nums) - 1
        while start + 1 < end:
          mid = start + (end - start) / 2
          if nums[start] < nums[mid]:
            start = mid
          elif nums[start] > nums[mid]:
            end = mid
          else:
            start += 1
            ans = min(ans, nums[start])

        return min(ans, nums[start], nums[end])


    =================================================================
    class Solution(object):
        def findMin(self, nums):
            assert(nums)
            left = 0
            right = len(nums) - 1
            # Make sure right is always in the right rotated part.
            # Left can be either in the left part or the minimum part.
            # So, when left and right is the same finally, we find the minimum.
            while left < right:
                # When there is no rotate, just return self.nums[start]
                if nums[left] < nums[right]:
                    return nums[left]

                mid = (left + right) / 2
                # mid is in the left part, so move the left point to mid.
                if nums[left] < nums[mid]:
                    left = mid + 1
                elif nums[left] > nums[mid]:
                    right = mid
                # Can't make sure whether left is in the left part or not.
                # Just move to right for 1 step.
                else:
                    left += 1
            return nums[left]

    """
    [1]
    [7,8,9,9,9,10,2,2,2,3,4,4,5]
    """



162. find peak element
------------------------------------------

.. code-block:: python

    A peak element is an element that is greater than its neighbors.

    Given an input array where num[i] &ne; num[i+1], find a peak element and return its index.

    The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

    You may imagine that num[-1] = num[n] = -&infin;.

    For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.

    click to show spoilers.

    Note:
    Your solution should be in logarithmic complexity.


    Credits:Special thanks to @ts for adding this problem and creating all test cases.
    =================================================================
    class Solution(object):
      def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start, end = 0, len(nums) - 1
        while start + 1 < end:
          mid = start + (end - start) / 2
          if nums[mid] < nums[mid + 1]:
            start = mid
          else:
            end = mid
        if nums[start] > nums[end]:
          return start
        return end


    =================================================================
    class Solution(object):
        """
        Binary search
        Three possible situations(here target is just one of the peeks):
            1. left, left+1, ..., mid-1, mid, mid+1, ..target.., right
            2. left, left+1, ..target.., mid-1, mid, mid+1, ..., right
            3. left, left+1, ..., mid-1, mid(target), mid+1, ..., right
        """
        def findPeakElement(self, nums):
            if not nums:
                return 0
            right = len(nums) - 1
            left = 0

            while left < right:
                mid = (left + right) / 2
                if nums[mid] < nums[mid+1]:
                    left = mid + 1
                else:
                    right = mid

            return left

    """
if __name__ == '__main__':
    sol = Solution()
    print sol.findPeakElement([1])
    print sol.findPeakElement([1, 2])
    print sol.findPeakElement([1, 2, 3, 4])
    print sol.findPeakElement([1, 2, 3, 2, 1, 4, 1, 2, 3])
"""


222. count complete tree nodes
------------------------------------------

.. code-block:: python

    Given a complete binary tree, count the number of nodes.

    Definition of a complete binary tree from Wikipedia:
    In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
    =================================================================
    class Solution(object):
      def getHeight(self, root):
        height = 0
        while root:
          height += 1
          root = root.left
        return height

      def countNodes(self, root):
        count = 0
        while root:
          l, r = map(self.getHeight, (root.left, root.right))
          if l == r:
            count += 2 ** l
            root = root.right
          else:
            count += 2 ** r
            root = root.left
        return count

    =================================================================
    class Solution(object):
        def countNodes(self, root):
            if not root:
                return 0
            node_nums = 0
            tree_height = self.getHeight(root)
            while root:
                if self.getHeight(root.right) == tree_height - 1:
                    # root.left's subtree is a full complete binary tree
                    # and it's height is tree_height-1
                    node_nums += 1 << tree_height
                    root = root.right
                else:
                    # root.right's subtree is a full complete binary tree
                    # and it's height is tree_height-2
                    node_nums += 1 << (tree_height-1)
                    root = root.left

                tree_height -= 1

            return node_nums

        # Get complete BT's height, assume the root is height 0, increment then.
        def getHeight(self, root):
            if not root:
                return -1
            height = 0
            while root.left:
                root = root.left
                height += 1
            return height

    """
    []
    [1]
    [1,2,3,4,5,6,7,8,9,10]
    [1,2,3,4,5]
    """



230. Kth smallest element in a bst
------------------------------------------

.. code-block:: python

    Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

    Note:
    You may assume k is always valid, 1 &le; k &le; BST's total elements.

    Follow up:
    What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

    Credits:Special thanks to @ts for adding this problem and creating all test cases.
    =================================================================
    class Solution(object):
      def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = [(1, root)]
        while stack:
          cmd, p = stack.pop()
          if not p:
            continue
          if cmd == 0:
            k -= 1
            if k == 0:
              return p.val
          else:
            stack.append((1, p.right))
            stack.append((0, p))
            stack.append((1, p.left))

    =================================================================
    class Solution(object):
        def kthSmallest(self, root, k):
            count = self.get_nodes(root.left)
            while count + 1 != k:
                if count + 1 < k:
                    root = root.right
                    k = k - count - 1
                else:
                    root = root.left
                count = self.get_nodes(root.left)
            return root.val

        def get_nodes(self, root):
            if not root:
                return 0
            return 1 + self.get_nodes(root.left) + self.get_nodes(root.right)


    # Binary search recursive
    class Solution_2(object):
        def kthSmallest(self, root, k):
            count = self.get_nodes(root.left)
            if count+1 < k:
                return self.kthSmallest(root.right, k-count-1)
            elif count+1 == k:
                return root.val
            else:
                return self.kthSmallest(root.left, k)

        def get_nodes(self, root):
            if not root:
                return 0
            return 1 + self.get_nodes(root.left) + self.get_nodes(root.right)


    # DFS in-order iterative:
    class Solution_3(object):
        def kthSmallest(self, root, k):
            node_stack = []
            count, result = 0, 0
            while root or node_stack:
                if root:
                    node_stack.append(root)
                    root = root.left
                else:
                    if node_stack:
                        root = node_stack.pop()
                        result = root.val
                        count += 1
                        if count == k:
                            return result
                        root = root.right

            return -1   # never hit if k is valid


    # DFS in-order recursive:
    class Solution_4(object):
        def kthSmallest(self, root, k):
            self.k = k
            self.num = 0
            self.in_order(root)
            return self.num

        def in_order(self, root):
            if root.left:
                self.in_order(root.left)
            self.k -= 1
            if self.k == 0:
                self.num = root.val
                return
            if root.right:
                self.in_order(root.right)


    # DFS in-order recursive, Pythonic approach with generator:
    class Solution_5(object):
        def kthSmallest(self, root, k):
            for val in self.in_order(root):
                if k == 1:
                    return val
                else:
                    k -= 1

        def in_order(self, root):
            if root:
                for val in self.in_order(root.left):
                    yield val
                yield root.val
                for val in self.in_order(root.right):
                    yield val

    """
    [1]
    1
    [3,1,4,null,2]
    1
    [10,8,6,9,14,12,15,null,null,null,null,11]
    4
    [10,8,6,9,14,12,15,null,null,null,null,11]
    5
    """


275. H index 2
------------------------------------------

.. code-block:: python

    Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize your algorithm?
    =================================================================
    class Solution(object):
      def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not citations:
          return 0
        n = len(citations)
        start, end = 0, n - 1
        while start < end:
          mid = start + (end - start) / 2
          if citations[mid] >= n - mid:
            end = mid
          else:
            start = mid + 1
        return n - start if citations[start] != 0 else 0


    =================================================================
    class Solution(object):
        # Binary Search, Yes!!
        def hIndex(self, citations):
            length = len(citations)
            left = 0
            right = length - 1
            while left <= right:
                # Disapproval / operator here(more slower), can use // or >> 1
                # mid = (left + right) / 2
                mid = (left + right) >> 1
                if citations[mid] == length - mid:
                    return citations[mid]
                elif citations[mid] > length - mid:
                    right = mid - 1
                else:
                    left = mid + 1
            return length - (right + 1)


    """
    []
    [0]
    [23]
    [0,1]
    [1,1,1,1]
    [4,4,4,4]
    [0,1,4,5,6]
    """


278. First Bad Version
------------------------------------------

.. code-block:: python

    You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.



    Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.



    You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
    =================================================================

    class Solution(object):
      def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        lo = 1
        hi = n
        while lo < hi:
          mid = lo + (hi - lo) / 2
          if isBadVersion(mid):
            hi = mid
          else:
            lo = mid + 1
        return lo


    =================================================================
    class Solution(object):
        # Attention: the latest version of your product fails the quality check
        # That's saying, given n versions must have at least one bad version.
        def firstBadVersion(self, n):
            if n <= 0:
                return 0
            left, right = 1, n
            while left < right:
                mid = (left + right) / 2
                if isBadVersion(mid):
                    right = mid
                else:
                    left = mid + 1
            return right



367. Valid perfect square
------------------------------------------

.. code-block:: python

    Given a positive integer num, write a function which returns True if num is a perfect square else False.


    Note: Do not use any built-in library function such as sqrt.


    Example 1:

    Input: 16
    Returns: True



    Example 2:

    Input: 14
    Returns: False



    Credits:Special thanks to @elmirap for adding this problem and creating all test cases.
    =================================================================
    class Solution(object):
      def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        r = num
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        r = (r + num / r) / 2
        return r * r == num



    =================================================================
    class Solution(object):
        # Binary Search
        def isPerfectSquare(self, num):
            low, high = 0, num
            while low <= high:
                mid = (low + high) / 2
                if mid ** 2 == num:
                    return True
                elif mid ** 2 > num:
                    high = mid - 1
                else:
                    low = mid + 1
            return False


    class Solution_2(object):
        # Truth: A square number is 1+3+5+7+...  Time Complexity O(sqrt(N))
        def isPerfectSquare(self, num):
            i = 1
            while num > 0:
                num -= i
                i += 2
            return num == 0


    class Solution_3(object):
        # Newton Method.  Time Complexity is close to constant.
        # According to: https://en.wikipedia.org/wiki/Newton%27s_method
        def isPerfectSquare(self, num):
            val = num
            while val ** 2 > num:
                val = (val + num / val) / 2
            return val * val == num

    """
    0
    1
    121
    12321
    2147483647
    """



378. kth smallest element in a sorted matrix
------------------------------------------------

.. code-block:: python

    Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.


    Note that it is the kth smallest element in the sorted order, not the kth distinct element.


    Example:

    matrix = [
       [ 1,  5,  9],
       [10, 11, 13],
       [12, 13, 15]
    ],
    k = 8,

    return 13.



    Note:
    You may assume k is always valid, 1 &le; k &le; n2.
    =================================================================
    import heapq


    class Solution(object):
      def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        visited = {(0, 0)}
        heap = [(matrix[0][0], (0, 0))]

        while heap:
          val, (i, j) = heapq.heappop(heap)
          k -= 1
          if k == 0:
            return val
          if i + 1 < len(matrix) and (i + 1, j) not in visited:
            heapq.heappush(heap, (matrix[i + 1][j], (i + 1, j)))
            visited.add((i + 1, j))
          if j + 1 < len(matrix) and (i, j + 1) not in visited:
            heapq.heappush(heap, (matrix[i][j + 1], (i, j + 1)))
            visited.add((i, j + 1))



    =================================================================
    class Solution(object):
        """ Heap merge is helpfull.
        """
        def kthSmallest(self, matrix, k):
            import heapq
            return list(heapq.merge(*matrix))[k - 1]


    class Solution(object):
        """ Binary Search can solve this too.
        """
        def kthSmallest(self, matrix, k):


    """
    [[1]]
    1
    [[1,2,3], [4,5,6], [7,8,9]]
    3
    [[ 1, 5, 9], [10, 11, 13], [12, 13, 15]]
    8
    """


