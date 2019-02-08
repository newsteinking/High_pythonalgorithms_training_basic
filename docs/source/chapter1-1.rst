Array - Easy
=======================================




`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

1.Two Sum
--------------------

주어진 정수 배열을 이용하여 두 숫자의 인덱스를 반환하여 특정 대상에 합산합니다.

각 입력에는 정확히 하나의 솔루션이 있다고 가정 할 수 있으며 동일한 요소를 두 번 사용할 수 없습니다

Example:
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

힌트:Dictionary를 사용하여 두 항을 뺀 값을 계속 넣어준다.



Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

.. code-block:: python



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

배열과 값이 주어지면 해당 값의 인스턴스를 모두 제거하고 새 길이를 반환합니다.

다른 배열을 위해 여분의 공간을 할당하지 마십시오. 일정한 메모리를 가지고 이를 수행해야합니다.

요소의 순서를 변경할 수 있습니다. 새로운 길이를 넘어 무엇을 남겨야할지  상관 없습니다.


Given an array and a value, remove all instances of that value in place and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example:
Given input array nums = [3,2,2,3], val = 3

Your function should return length = 2, with the first two elements of nums being 2.

Subscribe to see which companies asked this question.


.. code-block:: python





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

정렬 된 배열과 대상 값이 주어지면 대상을 찾으면 색인을 반환합니다. 그렇지 않은 경우 색인이 순서대로 삽입 된 경우 색인을 리턴하십시오.


Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Here are few examples.
[1,3,5,6], 5  2
[1,3,5,6], 2   1
[1,3,5,6], 7   4
[1,3,5,6], 0   0

.. code-block:: python



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


가장 큰 합을 가진 배열 (적어도 하나의 숫자 포함)에서 인접한 부분 배열을 찾습니다.


Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.




.. code-block:: python


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

비어 있지 않은 정수가 아닌 숫자의 배열로 표현되고, 정수에 1을 더한 값이 주어집니다.

숫자 0 그 자체를 제외하고 앞에 0을 포함하지 않는 정수로 가정 할 수 있습니다.

가장 중요한 자릿수가 목록의 머리에 있도록 자릿수가 저장됩니다.

ex) [3,4,5,6,7]  ==> 34567 ==> 34567+1 ===> 345678 ==> [3,4,5,6,7,8]


Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.


.. code-block:: python



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

두 개의 정렬 된 정수 배열 nums1과 nums2가 주어지면 nums2를 nums1에 하나의 정렬 된 배열로 병합하십시오.

노트 :
nums1에는 nums2의 추가 요소를 보유 할 수있는 충분한 공간 (m + n보다 크거나 같은 크기)이 있다고 가정 할 수 있습니다.
nums1과 nums2에서 초기화되는 요소의 수는 각각 m과 n입니다.

Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]


Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
The number of elements initialized in nums1 and nums2 are m and n respectively.




.. code-block:: python



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


88.2 Merge Sorted Array 2
-------------------------------

 두 array 크기만큼 합치고 정렬하는 문제이다.


Input :  arr1[] = { 1, 3, 4, 5}
         arr2[] = {2, 4, 6, 8}
Output : arr3[] = {1, 2, 3, 4, 5, 6, 7, 8}

Input  : arr1[] = { 5, 8, 9}
         arr2[] = {4, 7, 8}
Output : arr3[] = {4, 5, 7, 8, 8, 9}


.. code-block:: python


    class Solution(object):

        def mergeArrays(self,arr1, arr2, n1, n2):
            arr3 = [None] * (n1 + n2)
            i = 0
            j = 0
            k = 0

            # Traverse both array
            while i < n1 and j < n2:

                # Check if current element
                # of first array is smaller
                # than current element of
                # second array. If yes,
                # store first array element
                # and increment first array
                # index. Otherwise do same
                # with second array
                if arr1[i] < arr2[j]:
                    arr3[k] = arr1[i]
                    k = k + 1
                    i = i + 1
                else:
                    arr3[k] = arr2[j]
                    k = k + 1
                    j = j + 1


            # Store remaining elements
            # of first array
            while i < n1:
                arr3[k] = arr1[i];
                k = k + 1
                i = i + 1

            # Store remaining elements
            # of second array
            while j < n2:
                arr3[k] = arr2[j];
                k = k + 1
                j = j + 1
            print("Array after merging")
            for i in range(n1 + n2):
                print(str(arr3[i]), end = " ")


    sam=Solution()

    # Driver code
    arr1 = [1, 3, 5, 7]
    n1 = len(arr1)

    arr2 = [2, 4, 6, 8]
    n2 = len(arr2)



print(sam.mergeArrays(arr1,arr2,n1,n2))





118. Pascal's Triangle
-------------------------------


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

.. code-block:: python



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

118.2 Pascal's Triangle
-------------------------------
Enter number of rows: 5
                     1
                  1      1
               1      2      1
            1      3      3      1
         1      4      6      4      1


.. code-block:: python


    n=int(input("Enter number of rows: "))
    a=[]
    for i in range(n):
        a.append([])
        a[i].append(1)
        for j in range(1,i):
            a[i].append(a[i-1][j-1]+a[i-1][j])
        if(n!=0):
            a[i].append(1)
    for i in range(n):
        print("   "*(n-i),end=" ",sep=" ")
        for j in range(0,i+1):
            print('{0:6}'.format(a[i][j]),end=" ",sep=" ")
        print()

118.3 Pascal's Triangle 3
-------------------------------
[1]
[1, 1]
[1, 2, 1]
[1, 3, 3, 1]
[1, 4, 6, 4, 1]
[1, 5, 10, 10, 5, 1]

.. code-block:: python

    def pascal_triangle(n):
       trow = [1]
       y = [0]
       for x in range(max(n,0)):
          print(trow)
          trow=[l+r for l,r in zip(trow+y, y+trow)]
       return n>=1
    pascal_triangle(6)

119. Pascal's Triangle 2
-------------------------------

[1, 1]
[1, 2, 1]
[1, 3, 3, 1]
[1, 4, 6, 4, 1]
[1, 5, 10, 10, 5, 1]
[1, 5, 10, 10, 5, 1]


Given an index k, return the kth row of the Pascal's triangle.

For example, given k = 3,
Return [1,3,3,1].

.. code-block:: python




    class Solution(object):
        def getRow(self, rowIndex):
            """
            :type rowIndex: int
            :rtype: List[int]
            """
            res = [1]
            for i in range(1, rowIndex + 1):
                res = list(map(lambda x, y: x + y, res + [0], [0] + res))
                print(res)
            return res




121. Best Time to Buy and Sell Stock
--------------------------------------------

하루에 한 Transaction이 이루어져야 한다.
[7, 1, 5, 3, 6, 4]
day 1 (price 7) day 2 (price 1) day 3 (price 5)
여기서 buy 가격(낮은 가격이 ) 먼저 나온후 selling 가격이 나와야 한다.



Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock),
design an algorithm to find the maximum profit.

Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
Example 2:
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.

.. code-block:: python



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

    ======================================================
    class Solution:
        # @param prices, a list of integer
        # @return an integer
        def maxProfit(self, prices):
            minValue = float("inf")
            maxBenefit = 0
            for price  in prices:
                if minValue > price:
                    minValue = price
                if maxBenefit < price - minValue:
                    maxBenefit = price - minValue
            return maxBenefit


122. Best Time to Buy and Sell Stock 2
-----------------------------------------

여러번의 Transaction이 가능한 경우이다.


Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions
as you like (ie, buy one and sell one share of the stock multiple times). However,
you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).




.. code-block:: python



    class Solution(object):
        def maxProfit(self, prices):
            """
            :type prices: List[int]
            :rtype: int
            """
            return sum(max(prices[i+1]-prices[i],0) for i in range(len(prices)-1))


167. Two Sum 2
-------------------------------

이미 오름차순으로 정렬 된 정수 배열을 감안할 때 두 개의 숫자가 특정 대상 번호와 더해진다.

함수 twoSum은 두 숫자의 인덱스를 반환하여 대상에 추가합니다. 여기서 index1은 index2보다 작아야합니다. 반환 된 답변 (index1과 index2 모두)은 0부터 시작하지 않습니다.

각 입력에는 정확히 하나의 솔루션이 있다고 가정 할 수 있으며 동일한 요소를 두 번 사용할 수 없습니다.


Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2

.. code-block:: python



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
    ==================================================
        def twoSum2(self,nums, target):
            lookup = dict((v, i) for i, v in enumerate(nums)) # N
            for i, v in enumerate(nums):  # N
                if target - v in lookup and i != lookup[target - v]: # average constant
                    return [lookup[target - v], i]  # constant

169. Majority Element
-------------------------------

주어진 크기의 배열 n, 다수 요소를 찾으십시오. 대부분의 요소는 n / 2 번 이상 나타나는 요소입니다.

배열이 비어 있지 않고 배열에 주 요소가 항상 있다고 가정 할 수 있습니다.

핵심)리스트에서 리스트 갯수의 (n/2) 보다 많이 나타나는 숫자 구하기

Input : 3 3 4 2 4 4 2 4 4
Output : 4

Input : 3 3 4 2 4 4 2 4
Output : NONE

Given an array of size n, find the majority element. The majority element is the element that appears more than  n/2  times.

You may assume that the array is non-empty and the majority element always exist in the array.

Credits:
Special thanks to @ts for adding this problem and creating all test cases.




.. code-block:: python


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

    class Solution2(object):
        def majorityElement(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return sorted(nums)[len(nums)/2]
    =============================================
    from collections import Counter

    def majority(arr):

        # convert array into dictionary
        freqDict = Counter(arr)

        # traverse dictionary and check majority element
        size = len(arr)
        for (key,val) in freqDict.items():
             if (val > (size/2)):
                 print(key)
                 return
        print('None')

    # Driver program
    if __name__ == "__main__":
        arr = [3,3,4,2,4,4,2,4,4]
        majority(arr)

189. Rotate Array
-------------------------------

Rotate an array of n elements to the right by k steps.

For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

Note:
Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.

[show hint]

Related problem: Reverse Words in a String II

Credits:
Special thanks to @Freezen for adding this problem and creating all test cases.

Subscribe to see which companies asked this question.


.. code-block:: python

    class Solution(object):
        def rotate(self, nums, k):
            """
            :type nums: List[int]
            :type k: int
            :rtype: void Do not return anything, modify nums in-place instead.
            """
            n = len(nums)
            nums[:] = nums[n - k:] + nums[:n - k]

            return nums


    test=Solution()
    print(test.rotate([1,2,3,4,5,6,7], 3))

    ========================================
    from collections import deque
    d=deque([1,2,3,4,5])
    print(d)
    deque([1, 2, 3, 4, 5])
    d.rotate(2)
    print(d)



217. Contains Duplicate
-------------------------------


Given an array of integers, find if the array contains any duplicates.
Your function should return true if any value appears at least twice in the array,
and it should return false if every element is distinct.

Subscribe to see which companies asked this question.


.. code-block:: python



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


    class Solution2(object):
        def containsDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            return len(nums) != len(set(nums))


219. Contains Duplicate 2
-------------------------------

정수 배열과 정수 k가 주어지면 두 개의 다른 인덱스 i가 있는지 알아보십시오.
와 j가 배열에서 nums [i] = nums [j]이고 i와 j의 절대 차가 많아야 k가되도록 배열에 넣습니다.
핵심) 주어진 숫자 K값 범위 안에 반복되는 숫자가 있는지 확인하는 것임

Given an array of integers and an integer k, find out whether there are two distinct indices i
and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

.. code-block:: python



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

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

For example,
Given nums = [0, 1, 3] return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?


.. code-block:: python


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


핵심)Array에서 0은 모두 끝쪽으로 보내는 알고리즘

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.
Credits:
Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.


.. code-block:: python



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

핵심) Array에서 3번째 큰수를 찾는 알고리즘
      3번째 수가 없으면 가장 큰수를 리턴하고
      동일 숫자는 하나로 취급한다.

Given a non-empty array of integers, return the third maximum number in this array.
If it does not exist, return the maximum number. The time complexity must be in O(n).

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


.. code-block:: python



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

    ==================================================
    class Solution(object):
        def thirdMax(self, nums):
            l = []
            for n in set(nums):
                bisect.insort(l, -n)
                if len(l)>3:
                    l.pop()
            return -l[2] if len(l)>2 else -l[0]

    ============================================
    import sys
    def thirdLargest(arr, arr_size):

        # There should be
        # atleast three elements
        if (arr_size < 3):

            print(" Invalid Input ")
            return


        # Find first
        # largest element
        first = arr[0]
        for i in range(1, arr_size):
            if (arr[i] > first):
                first = arr[i]

        # Find second
        # largest element
        second = -sys.maxsize
        for i in range(0, arr_size):
            if (arr[i] > second and
                arr[i] < first):
                second = arr[i]

        # Find third
        # largest element
        third = -sys.maxsize
        for i in range(0, arr_size):
            if (arr[i] > third and
                arr[i] < second):
                third = arr[i]

        print("The Third Largest",
              "element is", third)

    # Driver Code
    arr = [12, 13, 1,
           10, 34, 16]
    n = len(arr)
    thirdLargest(arr, n)

    # This code is contributed
    # by Smitha

448. Find All Numbers Disappeared in Array
-------------------------------------------------


Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.
Example:

Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]

.. code-block:: python



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

Given a binary array, find the maximum number of consecutive 1st in this array.

Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.
Note:

The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000

.. code-block:: python


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

.. code-block:: python


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

Given an array of 2nd integers, your task is to group these integers into n pairs of integer,
say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

Example 1:
Input: [1,4,3,2]

Output: 4
Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).
Note:
n is a positive integer, which is in the range of [1, 10000].
All the integers in the array will be in the range of [-10000, 10000].


.. code-block:: python


    class Solution(object):
        def arrayPairSum(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return sum(sorted(nums)[::2])


566. Reshape the Matrix
-------------------------------

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



.. code-block:: python



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


Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too.

You need to find the shortest such subarray and output its length.

Example 1:
Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
Note:
Then length of the input array is in range [1, 10,000].
The input array may contain duplicates, so ascending order here means <=.

.. code-block:: python




    class Solution(object):
        def findUnsortedSubarray(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            all_same = [a == b for (a, b) in zip(nums, sorted(nums))]
            return 0 if all(all_same) else len(nums) - all_same.index(False) - all_same[::-1].index(False)


