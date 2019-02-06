Bit manipulation - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

136. Single Number
--------------------

.. code-block:: python

    """

    Given an array of integers, every element appears twice except for one. Find that single one.

    Note:
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

    """
    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            res = 0
            for i in nums:
                res = res ^ i
            return res


169. Majority Element
----------------------------

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



190. Reverse Bits
--------------------

.. code-block:: python

    """

    Reverse bits of a given 32 bits unsigned integer.

    For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000).

    Follow up:
    If this function is called many times, how would you optimize it?

    Related problem: Reverse Integer

    Credits:
    Special thanks to @ts for adding this problem and creating all test cases.

    """

    class Solution:
        # @param n, an integer
        # @return an integer
        def reverseBits(self, n):
            stack = []
            while n:
                stack.append(n % 2)
                n = n / 2
            while len(stack) < 32:
                stack.append(0)
            ret = 0
            for num in stack:
                ret = ret * 2 + num
            return ret



191. Number of 1 Bits
------------------------------

.. code-block:: python

    """

    Write a function that takes an unsigned integer and returns the number of xx1' bits it has (also known as the Hamming weight).

    For example, the 32-bit integer xx11' has binary representation 00000000000000000000000000001011, so the function should return 3.

    Credits:
    Special thanks to @ts for adding this problem and creating all test cases.



    """

    class Solution(object):
        def hammingWeight(self, n):
            """
            :type n: int
            :rtype: int
            """
            count = 0
            while n:
                n = n & (n-1)
                count = count + 1
            return count


231. Power of Two
--------------------

.. code-block:: python

    """

    Given an integer, write a function to determine if it is a power of two.

    Credits:
    Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    """

    class Solution(object):
        def isPowerOfTwo(self, n):
            """
            :type n: int
            :rtype: bool
            """
            return n>0 and not (n & n-1)



268. Missing Number
-----------------------

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




    class Solution(object):
        def missingNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            result = len(nums)
            for index,value in enumerate(nums):
                result ^= index
                result ^= value
            return result


342. Power of Four
--------------------

.. code-block:: python

    """

    Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

    Example:
    Given num = 16, return true. Given num = 5, return false.

    Follow up: Could you solve it without loops/recursion?

    Credits:
    Special thanks to @yukuairoy for adding this problem and creating all test cases.

    """

    class Solution(object):
        def isPowerOfFour(self, num):
            """
            :type num: int
            :rtype: bool
            """
            return num>0 and (num & num-1)==0 and (num-1)%3==0



371. Sum of Two Integers
--------------------------------

.. code-block:: python

    """

    Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

    Example:
    Given a = 1 and b = 2, return 3.

    Credits:
    Special thanks to @fujiaozhu for adding this problem and creating all test cases.

    """

    class Solution(object):
        def getSum(self, a, b):
            """
            :type a: int
            :type b: int
            :rtype: int
            """
            if a==0:
                return b
            if b==0:
                return a
            while b:
                carry = a & b
                a = a ^ b
                b = carry << 1
            return a


    """python solution"""
    class Solution(object):
        def getSum(self, a, b):
            """
            :type a: int
            :type b: int
            :rtype: int
            """
            # 32 bits integer max
            MAX = 0x7FFFFFFF
            # 32 bits interger min
            MIN = 0x80000000
            # mask to get last 32 bits
            mask = 0xFFFFFFFF
            while b != 0:
                # ^ get different bits and & gets double 1s, << moves carry
                a, b = (a ^ b) & mask, ((a & b) << 1) & mask
            # if a is negative, get a's 32 bits complement positive first
            # then get 32-bit positive's Python complement negative

            return a if a <= MAX else ~(a ^ mask)


389. Find the Difference
--------------------------------

.. code-block:: python

    """

    Given two strings s and t which consist of only lowercase letters.

    String t is generated by random shuffling string s and then add one more letter at a random position.

    Find the letter that was added in t.

    Example:

    Input:
    s = "abcd"
    t = "abcde"

    Output:
    e

    Explanation:
    'e' is the letter that was added.

    """


    class Solution(object):
        def findTheDifference(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: str
            """
            res = 0
            for i in s+t:
                res ^= ord(i)
            return chr(res)

461. Hamming Distance
--------------------------------

.. code-block:: python

    """

    The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

    Given two integers x and y, calculate the Hamming distance.

    Note:
    0 xxx x, y < 231.

    Example:

    Input: x = 1, y = 4

    Output: 2

    Explanation:
    1   (0 0 0 1)
    4   (0 1 0 0)
           xxxx xxxx

    The above arrows point to positions where the corresponding bits are different.

    """
    class Solution(object):
        def hammingDistance(self, x, y):
            """
            :type x: int
            :type y: int
            :rtype: int
            """
            t = x ^ y
            count = 0
            while t != 0:
                count = count + 1
                t = t & (t-1)
            return count


476. Number Complement
--------------------------------

.. code-block:: python

    """

    Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.

    Note:
    The given integer is guaranteed to fit within the range of a 32-bit signed integer.
    You could assume no leading zero bit in the integer xxx  binary representation.
    Example 1:
    Input: 5
    Output: 2
    Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
    Example 2:
    Input: 1
    Output: 0
    Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.

    """

    class Solution(object):
        def findComplement(self, num):
            """
            :type num: int
            :rtype: int
            """
            i=1
            while i<=num:
                i = i << 1
            return (i-1) ^ num


