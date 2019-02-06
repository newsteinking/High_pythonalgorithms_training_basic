chapter 1: Array - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

1.Two Sum
--------------------

.. code-block:: python

    """

    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Example:
    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].

    """

    class Solution(object):
        def twoSum(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            dic = dict()
            for index,value in enumerate(nums):
                sub = target - value
                if sub in dic:
                    return [dic[sub],index]
                else:
                    dic[value] = index

27.Remove Element
--------------------------

.. code-block:: python

    """

    Given an array and a value, remove all instances of that value in place and return the new length.

    Do not allocate extra space for another array, you must do this in place with constant memory.

    The order of elements can be changed. It doesn't matter what you leave beyond the new length.

    Example:
    Given input array nums = [3,2,2,3], val = 3

    Your function should return length = 2, with the first two elements of nums being 2.

    Subscribe to see which companies asked this question.

    """


    class Solution(object):
        def removeElement(self, nums, val):
            """
            :type nums: List[int]
            :type val: int
            :rtype: int
            """
            index = 0
            for i in range(0,len(nums)):
                if nums[i] != val:
                    nums[index] = nums[i]
                    index = index + 1
            return index

35.Search Insert Position
-------------------------------

.. code-block:: python


    """

    Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You may assume no duplicates in the array.

    Here are few examples.
    [1,3,5,6], 5  2
    [1,3,5,6], 2   1
    [1,3,5,6], 7   4
    [1,3,5,6], 0   0

    """


    class Solution(object):
        def searchInsert(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: int
            """
            left = 0
            right = len(nums)-1
            while left <= right:
                mid = (right - left) / 2 + left
                if nums[mid] == target:
                    return mid
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

53.Maximum Subarray
-------------------------------

.. code-block:: python


    """

    Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

    For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
    the contiguous subarray [4,-1,2,1] has the largest sum = 6.

    click to show more practice.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def maxSubArray(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            curSum=maxSum=nums[0]
            for i in range(1,len(nums)):
                curSum = max(nums[i],curSum+nums[i])
                maxSum = max(curSum,maxSum)
            return maxSum

66.Plus One
-------------------------------

.. code-block:: python


    """

    Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.

    You may assume the integer do not contain any leading zero, except the number 0 itself.

    The digits are stored such that the most significant digit is at the head of the list.

    """

    class Solution(object):
        def plusOne(self, digits):
            """
            :type digits: List[int]
            :rtype: List[int]
            """
            sum = 0
            for i in digits:
                sum = sum * 10 + i
            return [int(x) for x in str(sum+1)]


88. Merge Sorted Array
-------------------------------

.. code-block:: python


    """

    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

    Note:
    You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def merge(self, nums1, m, nums2, n):
            """
            :type nums1: List[int]
            :type m: int
            :type nums2: List[int]
            :type n: int
            :rtype: void Do not return anything, modify nums1 in-place instead.
            """
            while m > 0 and n > 0:
                if nums1[m - 1] > nums2[n - 1]:
                    nums1[m + n - 1] = nums1[m - 1]
                    m = m - 1
                else:
                    nums1[m + n - 1] = nums2[n - 1]
                    n = n - 1
            if n > 0:
                nums1[:n] = nums2[:n]


