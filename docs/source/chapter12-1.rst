Greedy - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

45. Jump Game 2
--------------------

.. code-block:: python

    Given an array of non-negative integers, you are initially positioned at the first index of the array.


    Each element in the array represents your maximum jump length at that position.


    Your goal is to reach the last index in the minimum number of jumps.



    For example:
    Given array A = [2,3,1,1,4]


    The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.)



    Note:
    You can assume that you can always reach the last index.

    =================================================================
    class Solution(object):
      def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pos = 0
        ans = 0
        bound = len(nums)
        while pos < len(nums) - 1:
          dis = nums[pos]
          farthest = posToFarthest = 0
          for i in range(pos + 1, min(pos + dis + 1, bound)):
            canReach = i + nums[i]
            if i == len(nums) - 1:
              return ans + 1
            if canReach > farthest:
              farthest = canReach
              posToFarthest = i
          ans += 1
          pos = posToFarthest
        return ans

    =================================================================
    class Solution(object):
        def jump(self, nums):
            """ When you can reach position i, find the next longest distance you can reach.

            Once we can reach position i, we can find the next longest distance by iterate all
            the position before position i.

            Of course, you can think it as a BFS problem.
            Where nodes in level i are all the nodes that can be reached in i-1th jump.
            For more explnation, goto:
            https://discuss.leetcode.com/topic/3191/o-n-bfs-solution
            """
            if len(nums) == 1:
                return 0

            last = nums[0]
            step = 1
            index = 1
            while last < len(nums) - 1:
                max_distance = 0
                while index <= last:
                    if nums[index] + index > max_distance:
                        max_distance = nums[index] + index
                    index += 1

                last = max_distance
                step += 1

            return step

    """
    [0]
    [2,5,0,3]
    [2,3,1,1,4]
    [3,1,8,1,1,1,1,1,5]
    """



55. Jump Game
--------------------

.. code-block:: python

    Given an array of non-negative integers, you are initially positioned at the first index of the array.


    Each element in the array represents your maximum jump length at that position.


    Determine if you are able to reach the last index.



    For example:
    A = [2,3,1,1,4], return true.


    A = [3,2,1,0,4], return false.


    =================================================================
    class Solution(object):
      def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pos = 0
        bound = len(nums)
        while pos < len(nums) - 1:
          dis = nums[pos]
          if dis == 0:
            return False
          farthest = posToFarthest = 0
          for i in range(pos + 1, min(pos + dis + 1, bound)):
            canReach = i + nums[i]
            if i == len(nums) - 1:
              return True
            if canReach > farthest:
              farthest = canReach
              posToFarthest = i
          pos = posToFarthest
        return True if pos >= len(nums) - 1 else False

    =================================================================

    class Solution(object):
        def canJump(self, nums):
            """
            The main idea is to see if current element can be
            reached by previous max jump.
            If not, return false. If true, renew the max jump.
            """
            length = len(nums)
            index, max_distance = 0, nums[0]

            while index < length:
                # Prune here.
                if max_distance >= length - 1:
                    return True

                if max_distance >= index:
                    max_distance = max(max_distance, index + nums[index])
                else:
                    # Current position cannot be reached.
                    return False
                index += 1

            return True

    """
    [0]
    [2,3,1,1,4]
    [3,2,1,0,4]
    [1,3,5,0,0,0,0,0]
    """


122. Best time to buy and sell stock
-------------------------------------------

.. code-block:: python


    Say you have an array for which the ith element is the price of a given stock on day i.

    Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

    =================================================================
    class Solution(object):
      def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        ans = 0
        for i in range(1, len(prices)):
          if prices[i] > prices[i - 1]:
            ans += prices[i] - prices[i - 1]
        return ans


    =================================================================
    class Solution(object):
        """ As long as there is a price gap, we gain a profit.
        """
        def maxProfit(self, prices):
            max_profit = 0
            for i in range(1, len(prices)):
                diff = prices[i] - prices[i - 1]
                if diff > 0:
                    max_profit += diff
            return max_profit

    """
    []
    [3,4,5,6,2,4]
    [6,5,4,3,2,1]
    [1,2,3,4,3,2,1,9,11,2,20]
    """



134. Gas Station
--------------------

.. code-block:: python


    There are N gas stations along a circular route, where the amount of gas at station i is gas[i].



    You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.



    Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.



    Note:
    The solution is guaranteed to be unique.

    =================================================================
    class Solution(object):
      def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """

        totalgas = 0
        totalcost = 0
        start = 0
        balance = 0
        for i in range(0, len(gas)):
          totalgas += gas[i]
          totalcost += cost[i]

        for i in range(0, len(gas)):
          balance += gas[i] - cost[i]
          if balance < 0:
            start = i + 1
            balance = 0

        if totalcost <= totalgas:
          return start
        return -1


    =================================================================
    class Solution(object):
        """
        Consider we start at gas station i, and until j we firstly run out of gas.

        That's say remain(i,j) = R(i) + ... + R(j) < 0, R(i) >= 0, R(j) < 0
        and remain(i, m) >= 0, where i =< m < j,
        We assume R(k) = gas(k) - cost(k) here.

        Further more, we can make sure remain(m+1, k) < 0.
        Just because remain(i,j) < 0 and remain(i, m) >= 0.
        So, next we just need to start from index k+1.

        So, firstly find all the (i,j) pairs, but just need to record the last j.
        Then if there is an unique(it's guaranteed) solution, it must be (j+1)
        """
        def canCompleteCircuit(self, gas, cost):
            station_num = len(gas)
            mark_station = -1
            all_remain = 0
            remain_gas = 0
            for i in range(station_num):
                all_remain += (gas[i]-cost[i])
                remain_gas += (gas[i]-cost[i])
                if remain_gas < 0:
                    mark_station = i
                    remain_gas = 0

            if all_remain >= 0:
                return (mark_station + 1) % station_num
            else:
                return -1

            return -1

    """
    [4]
    [5]
    [1,10,2,3,4,5,6]
    [2,4,3,4,5,6,7]
    [1,2,3,4,5,6,10]
    [1,2,2,3,4,15,4]
    [2,0,1,2,3,4,0]
    [0,1,0,0,0,0,11]
    """



316. Remove duplicate letters
-------------------------------------

.. code-block:: python

    Given a string which contains only lowercase letters, remove duplicate letters so that every letter appear once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.



    Example:


    Given "bcabc"
    Return "abc"


    Given "cbacdcbc"
    Return "acdb"


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.


    =================================================================
    class Solution(object):
      def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = {}
        count = {}
        for c in s:
          d[c] = d.get(c, 0) + 1
          count[c] = count.get(c, 0) + 1
        stack = []
        cache = set()
        for c in s:
          if c not in cache:
            while stack and stack[-1] > c and d[stack[-1]] > 1 and d[stack[-1]] != 1 and count[stack[-1]] > 0:
              cache.discard(stack.pop())
            stack.append(c)
            cache.add(c)
          count[c] -= 1
        return "".join(stack)

    =================================================================
    class Solution(object):
        def removeDuplicateLetters(self, s):
            """
            Given the string s, the greedy choice is the smallest s[i],
            s.t. the suffix s[i .. ] contains all the unique letters.
            After determining the greedy choice s[i], get a new string by:
                1. removing all letters to the left of s[i],
                2. removing all s[i] in s[i+1:].
            We then recursively solve the sub problem s'.
            """

            if not s:
                return ""

            # 1. Find out the last appeared position for each letter;
            char_dict = {}
            for i, c in enumerate(s):
                char_dict[c] = i

            # 2. Find out the smallest index (2) from the map in step 1;
            pos = len(s)
            for i in char_dict.values():
                if i < pos:
                    pos = i

            # 3. The first letter in the final result must be
            #    the smallest letter from index 0 to index (2);
            char = s[0]
            res_pos = 0
            for i in range(1, pos+1):
                if s[i] < char:
                    char = s[i]
                    res_pos = i
            # 4. Find out remaining letters with the new s.
            new_s = [c for c in s[res_pos+1:] if c != char]
            return char + self.removeDuplicateLetters("".join(new_s))


    # Use Stack to avoid recursive, more quickly.
    class Solution_2(object):
        def removeDuplicateLetters(self, s):
            char_dict = {}
            used = {}
            for c in s:
                char_dict[c] = char_dict.get(c, 0) + 1
                used[c] = False

            res = []        # Use as a Stack.
            for c in s:
                char_dict[c] -= 1
                if used[c]:
                    continue

                while res and res[-1] > c and char_dict[res[-1]] > 0:
                    used[res[-1]] = False
                    res.pop()

                res.append(c)
                used[c] = True
            return "".join(res)

    """
    ""
    "bcabc"
    "abacb"
    "cbacdcbc"
    """



330. Patching Array
--------------------

.. code-block:: python

    Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.


    Example 1:
    nums = [1, 3], n = 6
    Return 1.

    Combinations of nums are [1], [3], [1,3], which form possible sums of: 1, 3, 4.
    Now if we add/patch 2 to nums, the combinations are: [1], [2], [3], [1,3], [2,3], [1,2,3].
    Possible sums are 1, 2, 3, 4, 5, 6, which now covers the range [1, 6].
    So we only need 1 patch.

    Example 2:
    nums = [1, 5, 10], n = 20
    Return 2.
    The two patches can be [2, 4].

    Example 3:
    nums = [1, 2, 2], n = 5
    Return 0.

    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def minPatches(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: int
        """
        i = 0
        patches = 0
        miss = 1
        while miss <= n:
          if i < len(nums) and nums[i] <= miss:
            miss += nums[i]
            i += 1
          else:
            miss += miss
            patches += 1
        return patches


    =================================================================
    class Solution(object):
        """
        Let miss_num be the smallest sum in [0,n] that we might be missing.
        Meaning we already know we can build all sums in [0,miss). Then
            1. If we have a number num <= miss in the given array,
               we can add it to those smaller sums to build all sums in [0,miss+num).
            2. If we don't, then we must add such a number to the array,
               and it's best(GREEDY) to add miss itself, to maximize the reach.

        Here is a thinking process, maybe helpful.
        https://leetcode.com/discuss/83272/share-my-thinking-process
        """
        def minPatches(self, nums, n):
            miss_num = 1
            index = 0
            patch_cnt = 0
            length = len(nums)
            while miss_num <= n:
                if index < length and nums[index] <= miss_num:
                    miss_num += nums[index]
                    index += 1
                else:
                    patch_cnt += 1
                    miss_num <<= 1
                    # miss_num += miss_num
            return patch_cnt
    """
    [1,3]
    6
    [1, 5, 10]
    20
    [1, 2, 2]
    5
    """



