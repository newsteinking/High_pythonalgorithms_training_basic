Array - Easy
=======================================




`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

1.Two Sum
--------------------

.. code-block:: python

    주어진 정수 배열을 이용하여 두 숫자의 인덱스를 반환하여 특정 대상에 합산합니다.

    각 입력에는 정확히 하나의 솔루션이 있다고 가정 할 수 있으며 동일한 요소를 두 번 사용할 수 없습니다

    Example:
    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].

    힌트:Dictionary를 사용하여 두 항을 뺀 값을 계속 넣어준다.
    

    ===========================================================
    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Example:
    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].



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


118. Pascal's Triangle
-------------------------------

.. code-block:: python

    """

    Given numRows, generate the first numRows of Pascal's triangle.

    For example, given numRows = 5,
    Return

    [
         [1],
        [1,1],
       [1,2,1],
      [1,3,3,1],
     [1,4,6,4,1]
    ]
    Subscribe to see which companies asked this question.

    """


    class Solution(object):
        def generate(self, numRows):
            """
            :type numRows: int
            :rtype: List[List[int]]
            """
            if numRows == 0:return []
            res = [[1]]
            for i in range(1,numRows):
                res.append(map(lambda x,y:x+y,res[-1]+[0],[0]+res[-1]))
            return res

119. Pascal's Triangle 2
-------------------------------

.. code-block:: python


    """

    Given an index k, return the kth row of the Pascal's triangle.

    For example, given k = 3,
    Return [1,3,3,1].

    """

    class Solution(object):
        def getRow(self, rowIndex):
            """
            :type rowIndex: int
            :rtype: List[int]
            """
            res = [1]
            for i in range(1, rowIndex + 1):
                res = list(map(lambda x, y: x + y, res + [0], [0] + res))
            return res




121. Best Time to Buy and Sell Stock
--------------------------------------------

.. code-block:: python

    """

    Say you have an array for which the ith element is the price of a given stock on day i.

    If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

    Example 1:
    Input: [7, 1, 5, 3, 6, 4]
    Output: 5

    max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
    Example 2:
    Input: [7, 6, 4, 3, 1]
    Output: 0

    In this case, no transaction is done, i.e. max profit = 0.

    """

    class Solution(object):
        def maxProfit(self, prices):
            """
            :type prices: List[int]
            :rtype: int
            """
            curSum=maxSum=0
            for i in range(1,len(prices)):
                curSum=max(0,curSum+prices[i]-prices[i-1])
                maxSum = max(curSum,maxSum)
            return maxSum

122. Best Time to Buy and Sell Stock 2
-----------------------------------------

.. code-block:: python

    """

    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

    """

    class Solution(object):
        def maxProfit(self, prices):
            """
            :type prices: List[int]
            :rtype: int
            """
            return sum(max(prices[i+1]-prices[i],0) for i in range(len(prices)-1))


167. Two Sum 2
-------------------------------

.. code-block:: python

    """

    Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

    The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

    You may assume that each input would have exactly one solution and you may not use the same element twice.

    Input: numbers={2, 7, 11, 15}, target=9
    Output: index1=1, index2=2

    """

    class Solution(object):
        def twoSum(self, numbers, target):
            """
            :type numbers: List[int]
            :type target: int
            :rtype: List[int]
            """
            res = dict()
            for i in range(0,len(numbers)):
                sub = target - numbers[i]
                if sub in res.keys():
                    return [res[sub]+1,i+1]
                else:
                    res[numbers[i]] = i
            return []


169. Majority Element
-------------------------------

.. code-block:: python


    """

    Given an array of size n, find the majority element. The majority element is the element that appears more than �뙄 n/2 �뙅 times.

    You may assume that the array is non-empty and the majority element always exist in the array.

    Credits:
    Special thanks to @ts for adding this problem and creating all test cases.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def majorityElement(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            cand = nums[0]
            count = 1
            for i in nums[1:]:
                if count == 0:
                    cand, count = i, 1
                else:
                    if i == cand:
                        count = count + 1
                    else:
                        count = count - 1
            return cand

    class Solution(object):
        def majorityElement(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return sorted(nums)[len(nums)/2]


189. Rotate Array
-------------------------------

.. code-block:: python


    """

    Rotate an array of n elements to the right by k steps.

    For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

    Note:
    Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.

    [show hint]

    Related problem: Reverse Words in a String II

    Credits:
    Special thanks to @Freezen for adding this problem and creating all test cases.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def rotate(self, nums, k):
            """
            :type nums: List[int]
            :type k: int
            :rtype: void Do not return anything, modify nums in-place instead.
            """
            n = len(nums)
            nums[:] = nums[n - k:] + nums[:n - k]



217. Contains Duplicate
-------------------------------

.. code-block:: python

    """

    Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def containsDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            if not nums:
                return False
            dic = dict()
            for num in nums:
                if num in dic:
                    return True
                dic[num] = 1
            return False


    class Solution(object):
        def containsDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            return len(nums) != len(set(nums))


219. Contains Duplicate 2
-------------------------------

.. code-block:: python

    """

    Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

    """

    class Solution(object):
        def containsNearbyDuplicate(self, nums, k):
            """
            :type nums: List[int]
            :type k: int
            :rtype: bool
            """
            dic = dict()
            for index,value in enumerate(nums):
                if value in dic and index - dic[value] <= k:
                    return True
                dic[value] = index
            return False

268. Missing Number
-------------------------------

.. code-block:: python

    """
    Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

    For example,
    Given nums = [0, 1, 3] return 2.

    Note:
    Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

    """
    class Solution(object):
        def missingNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            n = len(nums)
            return n * (n+1) / 2 - sum(nums)



283. Move Zeroes
-------------------------------

.. code-block:: python

    """

    Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

    For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

    Note:
    You must do this in-place without making a copy of the array.
    Minimize the total number of operations.
    Credits:
    Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    """

    class Solution(object):
        def moveZeroes(self, nums):
            """
            :type nums: List[int]
            :rtype: void Do not return anything, modify nums in-place instead.
            """
            index = 0
            for num in nums:
                if num != 0:
                    nums[index] = num
                    index += 1
            for i in range(index,len(nums)):
                nums[i] = 0


414. Third Maximum Number
-------------------------------

.. code-block:: python

    """

    Given a non-empty array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).

    Example 1:
    Input: [3, 2, 1]

    Output: 1

    Explanation: The third maximum is 1.
    Example 2:
    Input: [1, 2]

    Output: 2

    Explanation: The third maximum does not exist, so the maximum (2) is returned instead.
    Example 3:
    Input: [2, 2, 3, 1]

    Output: 1

    Explanation: Note that the third maximum here means the third maximum distinct number.
    Both numbers with value 2 are both considered as second maximum.

    """

    class Solution(object):
        def thirdMax(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            max1 = max2= max3=None
            for num in nums:
                if num > max1:
                    max2,max3 = max1,max2
                    max1=num
                elif num > max2 and num < max1:
                    max2,max3= num,max2
                elif num > max3 and num < max2:
                    max3 = num
            return max1 if max3==None else max3

448. Find All Numbers Disappeared in Array
-------------------------------------------------

.. code-block:: python

    """

    Given an array of integers where 1  xxxxx  n (n = size of array), some elements appear twice and others appear once.

    Find all the elements of [1, n] inclusive that do not appear in this array.

    Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

    Example:

    Input:
    [4,3,2,7,8,2,3,1]

    Output:
    [5,6]

    """

    class Solution(object):
        def findDisappearedNumbers(self, nums):
            """
            :type nums: List[int]
            :rtype: List[int]
            """
            for i in range(len(nums)):
                index = abs(nums[i]) - 1
                nums[index] = -abs(nums[index])
            return [i + 1 for i in range(len(nums)) if nums[i] > 0]



485. Max Consecutive Ones
-------------------------------

.. code-block:: python


    """

    Given a binary array, find the maximum number of consecutive 1s in this array.

    Example 1:
    Input: [1,1,0,1,1,1]
    Output: 3
    Explanation: The first two digits or the last three digits are consecutive 1s.
        The maximum number of consecutive 1s is 3.
    Note:

    The input array will only contain 0 and 1.
    The length of input array is a positive integer and will not exceed 10,000

    """

    class Solution(object):
        def findMaxConsecutiveOnes(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            cnt = 0
            ans = 0
            for num in nums:
                if num == 1:
                    cnt = cnt + 1
                    ans = max(ans,cnt)
                else:
                    cnt = 0
            return ans

532. K-diff Pairs in an Array
-------------------------------

.. code-block:: python

    """

    Given an array of integers and an integer k, you need to find the number of unique k-diff pairs in the array. Here a k-diff pair is defined as an integer pair (i, j), where i and j are both numbers in the array and their absolute difference is k.

    Example 1:
    Input: [3, 1, 4, 1, 5], k = 2
    Output: 2
    Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
    Although we have two 1s in the input, we should only return the number of unique pairs.
    Example 2:
    Input:[1, 2, 3, 4, 5], k = 1
    Output: 4
    Explanation: There are four 1-diff pairs in the array, (1, 2), (2, 3), (3, 4) and (4, 5).
    Example 3:
    Input: [1, 3, 1, 5, 4], k = 0
    Output: 1
    Explanation: There is one 0-diff pair in the array, (1, 1).
    Note:
    The pairs (i, j) and (j, i) count as the same pair.
    The length of the array won't exceed 10,000.
    All the integers in the given input belong to the range: [-1e7, 1e7].

    """


    class Solution(object):
        def findPairs(self, nums, k):
            """
            :type nums: List[int]
            :type k: int
            :rtype: int
            """
            if k>0:
                return len(set(nums) & set(n+k for n in nums))
            elif k==0:
                return sum(v>1 for v in collections.Counter(nums).values())
            else:
                return 0






561. Array Partition 1
-------------------------------

.. code-block:: python


    """

    Given an array of 2n integers, your task is to group these integers into n pairs of integer, say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

    Example 1:
    Input: [1,4,3,2]

    Output: 4
    Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).
    Note:
    n is a positive integer, which is in the range of [1, 10000].
    All the integers in the array will be in the range of [-10000, 10000].

    """


    class Solution(object):
        def arrayPairSum(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return sum(sorted(nums)[::2])


566. Reshape the Matrix
-------------------------------

.. code-block:: python

    """

    In MATLAB, there is a very useful function called 'reshape', which can reshape a matrix into a new one with different size but keep its original data.

    You're given a matrix represented by a two-dimensional array, and two positive integers r and c representing the row number and column number of the wanted reshaped matrix, respectively.

    The reshaped matrix need to be filled with all the elements of the original matrix in the same row-traversing order as they were.

    If the 'reshape' operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.

    Example 1:
    Input:
    nums =
    [[1,2],
     [3,4]]
    r = 1, c = 4
    Output:
    [[1,2,3,4]]
    Explanation:
    The row-traversing of nums is [1,2,3,4]. The new reshaped matrix is a 1 * 4 matrix, fill it row by row by using the previous list.
    Example 2:
    Input:
    nums =
    [[1,2],
     [3,4]]
    r = 2, c = 4
    Output:
    [[1,2],
     [3,4]]
    Explanation:
    There is no way to reshape a 2 * 2 matrix to a 2 * 4 matrix. So output the original matrix.
    Note:
    The height and width of the given matrix is in range [1, 100].
    The given r and c are all positive.

    """

    class Solution(object):
        def matrixReshape(self, nums, r, c):
            """
            :type nums: List[List[int]]
            :type r: int
            :type c: int
            :rtype: List[List[int]]
            """
            if len(nums) * len(nums[0]) != r * c:
                return nums
            else:
                onerow = [nums[i][j] for i in range(len(nums)) for j in range(len(nums[0]))]
                return [onerow[t * c:(t + 1) * c] for t in range(r)]



581. Shortest Unsorted Continuous Subarray
---------------------------------------------------

.. code-block:: python


    """

    Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too.

    You need to find the shortest such subarray and output its length.

    Example 1:
    Input: [2, 6, 4, 8, 10, 9, 15]
    Output: 5
    Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
    Note:
    Then length of the input array is in range [1, 10,000].
    The input array may contain duplicates, so ascending order here means <=.

    """

    class Solution(object):
        def findUnsortedSubarray(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            all_same = [a == b for (a, b) in zip(nums, sorted(nums))]
            return 0 if all(all_same) else len(nums) - all_same.index(False) - all_same[::-1].index(False)


