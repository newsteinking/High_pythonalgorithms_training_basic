Math - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

66. Plus One
--------------------------------------

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


168. Excel Sheet Column Title
--------------------------------------

.. code-block:: python


    """

    Given a positive integer, return its corresponding column title as appear in an Excel sheet.

    For example:

        1 -> A
        2 -> B
        3 -> C
        ...
        26 -> Z
        27 -> AA
        28 -> AB
    Credits:
    Special thanks to @ifanchu for adding this problem and creating all test cases.

    """
    class Solution(object):
        def convertToTitle(self, n):
            """
            :type n: int
            :rtype: str
            """
            res = ''
            while n:
                res = chr((n - 1) % 26 + ord('A')) + res
                n = (n - 1) / 26
            return res



171. Excel Sheet Column Number
--------------------------------------

.. code-block:: python

    """

    Related to question Excel Sheet Column Title

    Given a column title as appear in an Excel sheet, return its corresponding column number.

    For example:

        A -> 1
        B -> 2
        C -> 3
        ...
        Z -> 26
        AA -> 27
        AB -> 28

    """

    class Solution(object):
        def titleToNumber(self, s):
            """
            :type s: str
            :rtype: int
            """
            res = 0
            for num in s:
                res = res * 26 + (ord(num)-ord('A')+1)
            return res




172. Factorial Trailing Zeroes
--------------------------------------

.. code-block:: python

    """

    Given an integer n, return the number of trailing zeroes in n!.

    Note: Your solution should be in logarithmic time complexity.

    """

    class Solution(object):
        def trailingZeroes(self, n):
            zeroCnt = 0
            while n > 0:
                n = n/ 5
                zeroCnt += n

            return zeroCnt


202. Happy Number
--------------------------------------

.. code-block:: python

    """

    Write an algorithm to determine if a number is "happy".

    A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

    Example: 19 is a happy number

    12 + 92 = 82
    82 + 22 = 68
    62 + 82 = 100
    12 + 02 + 02 = 1
    Credits:
    Special thanks to @mithmatt and @ts for adding this problem and creating all test cases.

    """
    class Solution(object):
        def isHappy(self, n):
            """
            :type n: int
            :rtype: bool
            """
            c = set()
            while not n in c:
                c.add(n)
                n = sum([int(x) ** 2 for x in str(n)])
            return n==1


    class Solution(object):
        def isHappy(self, n):
            """
            :type n: int
            :rtype: bool
            """
            slow = n
            quick = sum([int(x) ** 2 for x in str(n)])
            while quick != slow:
                quick = sum([int(x) ** 2 for x in str(quick)])
                quick = sum([int(x) ** 2 for x in str(quick)])
                slow = sum([int(x) ** 2 for x in str(slow)])
            return slow == 1





204. Count Primes
--------------------------------------

.. code-block:: python

    """

    Description:

    Count the number of prime numbers less than a non-negative number, n.

    """

    class Solution(object):
        def countPrimes(self, n):
            """
            :type n: int
            :rtype: int
            """
            if n < 3:
                return 0
            res = [True] * n
            res[0] = res[1] = False
            for i in range(2,int(math.sqrt(n)) + 1):
                res[i*i:n:i] = [False] * len(res[i*i:n:i])
            return sum(res)


231. Power of Two
--------------------------------------

.. code-block:: python


    """
    Given an integer, write a function to determine if it is a power of two.
    """

    class Solution(object):
        def isPowerOfTwo(self, n):
            """
            :type n: int
            :rtype: bool
            """
            return n>0 and not (n & n-1)

258. Add Digits
--------------------------------------

.. code-block:: python


    """

    Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

    For example:

    Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.

    Follow up:
    Could you do it without any loop/recursion in O(1) runtime?

    Credits:
    Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    """

    class Solution(object):
        def addDigits(self, num):
            """
            :type num: int
            :rtype: int
            """
            return (num % 9 or 9) if num else 0

268. Missing Number
--------------------------------------

.. code-block:: python

    """

    Write a program to check whether a given number is an ugly number.

    Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.

    Note that 1 is typically treated as an ugly number.

    """
    class Solution(object):
        def isUgly(self, num):
            """
            :type num: int
            :rtype: bool
            """
            if num==0:
                return False
            if num==1:
                return True
            while num % 2 == 0:
                num /= 2
            while num % 3 == 0:
                num /= 3
            while num % 5 == 0:
                num /=5
            return num == 1


367. Valid Perfect Square
--------------------------------------

.. code-block:: python

    """

    Given a positive integer num, write a function which returns True if num is a perfect square else False.

    Note: Do not use any built-in library function such as sqrt.

    Example 1:

    Input: 16
    Returns: True
    Example 2:

    Input: 14
    Returns: False

    """

    class Solution(object):
        def isPerfectSquare(self, num):
            """
            :type num: int
            :rtype: bool
            """
            r = num
            while r*r > num:
                r = (r + num/r) / 2
            return r*r == num



441. Arranging Coins
--------------------------------------




453. Minimum Moves to Equal Array Elements
-------------------------------------------------

.. code-block:: python

    """

    Given a non-empty integer array of size n, find the minimum number of moves required to make all array elements equal, where a move is incrementing n - 1 elements by 1.

    Example:

    Input:
    [1,2,3]

    Output:
    3

    Explanation:
    Only three moves are needed (remember each move increments two elements):

    [1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4]

    """

    class Solution(object):
        def minMoves(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            return sum(nums) - min(nums) * len(nums)

598. Range Addition 2
--------------------------------------

.. code-block:: python


    """

    Given an m * n matrix M initialized with all 0's and several update operations.

    Operations are represented by a 2D array, and each operation is represented by an array with two positive integers a and b, which means M[i][j] should be added by one for all 0 <= i < a and 0 <= j < b.

    You need to count and return the number of maximum integers in the matrix after performing all the operations.

    Example 1:
    Input:
    m = 3, n = 3
    operations = [[2,2],[3,3]]
    Output: 4
    Explanation:
    Initially, M =
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]

    After performing [2,2], M =
    [[1, 1, 0],
     [1, 1, 0],
     [0, 0, 0]]

    After performing [3,3], M =
    [[2, 2, 1],
     [2, 2, 1],
     [1, 1, 1]]

    So the maximum integer in M is 2, and there are four of it in M. So return 4.
    Note:
    The range of m and n is [1,40000].
    The range of a is [1,m], and the range of b is [1,n].
    The range of operations size won't exceed 10,000.


    """

    class Solution(object):
        def maxCount(self, m, n, ops):
            """
            :type m: int
            :type n: int
            :type ops: List[List[int]]
            :rtype: int
            """
            if not ops:
                return m*n
            m = ops[0][0]
            n = ops[0][1]
            for a in ops[1:]:
                m = min(m,a[0])
                n = min(n,a[1])
            return m * n