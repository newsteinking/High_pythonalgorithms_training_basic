Dynamic Programming - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode


53. Maximum Subarray
------------------------------------------

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


62. Unique Paths
------------------------------------------

.. code-block:: python


    """

    A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

    The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

    How many possible unique paths are there?




    """

    class Solution(object):
        def uniquePaths(self, m, n):
            """
            :type m: int
            :type n: int
            :rtype: int
            """
            if m > n:
                m, n = m - 1, n - 1
            else:
                m, n = n - 1, m - 1
            if m <= 0 or n <= 0:
                return 1
            t = m + n
            sub = 1
            for i in range(1, n):
                t = t * (m + n - i)
                sub = sub * (i + 1)

            return t / sub




    class Solution(object):
        def uniquePaths(self, m, n):
            """
            :type m: int
            :type n: int
            :rtype: int
            """
            count = [[0 for t in range(n)] for x in range(m)]
            count[0][0] = 1
            for i in range(m):
                for j in range(n):
                    if i == 0 and j == 0:
                        count[i][j] = 1
                    elif i == 0:
                        count[i][j] = count[i][j - 1]
                    elif j == 0:
                        count[i][j] = count[i - 1][j]
                    else:
                        count[i][j] = count[i - 1][j] + count[i][j - 1]
            return count[m - 1][n - 1]



63. Unique Paths 2
------------------------------------------

.. code-block:: python


    """

    Follow up for "Unique Paths":

    Now consider if some obstacles are added to the grids. How many unique paths would there be?

    An obstacle and empty space is marked as 1 and 0 respectively in the grid.

    For example,
    There is one obstacle in the middle of a 3x3 grid as illustrated below.

    [
      [0,0,0],
      [0,1,0],
      [0,0,0]
    ]
    The total number of unique paths is 2.

    Note: m and n will be at most 100.

    """

    class Solution(object):
        def uniquePathsWithObstacles(self, obstacleGrid):
            """
            :type obstacleGrid: List[List[int]]
            :rtype: int
            """
            if not obstacleGrid:
                return 0
            for i in range(len(obstacleGrid)):
                for j in range(len(obstacleGrid[0])):
                    if obstacleGrid[i][j] == 1:
                        obstacleGrid[i][j] = 0
                    elif i == 0 and j == 0:
                        obstacleGrid[i][j] = 1
                    elif i == 0:
                        obstacleGrid[i][j] = obstacleGrid[i][j - 1]
                    elif j == 0:
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j]
                    else:
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]
            return obstacleGrid[i][j]





64. Minimum Path Sum
------------------------------------------

.. code-block:: python

    """

    Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

    Note: You can only move either down or right at any point in time.

    """

    class Solution(object):
        def minPathSum(self, grid):
            """
            :type grid: List[List[int]]
            :rtype: int
            """
            for i in range(len(grid)-1,-1,-1):
                for j in range(len(grid[0])-1,-1,-1):
                    if j == len(grid[0])-1 and i== len(grid)-1:
                        continue
                    elif j == len(grid[0])-1:
                        grid[i][j] = grid[i+1][j] + grid[i][j]
                    elif i == len(grid)-1:
                        grid[i][j] = grid[i][j] + grid[i][j+1]
                    else:
                        grid[i][j] = grid[i][j] + min(grid[i+1][j],grid[i][j+1])
            return grid[0][0]


70. Climibing Stairs
------------------------------------------

.. code-block:: python


    """

    You are climbing a stair case. It takes n steps to reach to the top.

    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

    Note: Given n will be a positive integer.

    """

    class Solution(object):
        def climbStairs(self, n):
            """
            :type n: int
            :rtype: int
            """
            a = 1
            b = 1
            for i in range(2,n+1):
                a,b = b,a+b
            return b






121. Best Time To Buy and Sell Stock
------------------------------------------

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
