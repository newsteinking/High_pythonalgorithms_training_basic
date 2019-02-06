Bit manipulation - Easy 2
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

29. divide two integers
------------------------------------------------

.. code-block:: python


    Divide two integers without using multiplication, division and mod operator.


    If it is overflow, return MAX_INT.

    =================================================================
    class Solution(object):
      def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if divisor == 0:
          return 0x7fffffff
        sign = 1
        if dividend * divisor < 0:
          sign = -1
        ans = 0
        cnt = 1
        dividend = abs(dividend)
        divisor = abs(divisor)
        subsum = divisor
        while dividend >= divisor:
          while (subsum << 1) <= dividend:
            cnt <<= 1
            subsum <<= 1
          ans += cnt
          cnt = 1
          dividend -= subsum
          subsum = divisor
        return max(min(sign * ans, 0x7fffffff), -2147483648)

    =================================================================
    class Solution(object):
        # According to:
        # https://leetcode.com/discuss/38997/detailed-explained-8ms-c-solution
        # Key concept:
        # division simply requires us to find how many times we can subtract the
        # divisor from the the dividend without making the dividend negative.

        def divide(self, dividend, divisor):
            if divisor == 0:
                return -1

            # Make sure it's positive or negative
            positive = (dividend < 0) is (divisor < 0)
            dividend, divisor = abs(dividend), abs(divisor)
            answer = 0

            while dividend >= divisor:
                multiple, temp = 1, divisor
                while dividend >= (temp << 1):
                    multiple <<= 1
                    temp <<= 1
                dividend -= temp
                answer += multiple

            if not positive:
                answer = -answer

            if (answer > 2147483647) or (answer < -2147483648):
                return 2147483647
            return answer

    """
    0
    1
    12
    3
    125
    -4
    1
    -1
    """


78. subsets
------------------------------------------------

.. code-block:: python

    Given a set of distinct integers, nums, return all possible subsets.

    Note: The solution set must not contain duplicate subsets.


    For example,
    If nums = [1,2,3], a solution is:



    [
      [3],
      [1],
      [2],
      [1,2,3],
      [1,3],
      [2,3],
      [1,2],
      []
    ]

    =================================================================
    class Solution(object):
      def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def dfs(nums, index, path, ans):
          ans.append(path)
          [dfs(nums, i + 1, path + [nums[i]], ans) for i in range(index, len(nums))]

        ans = []
        dfs(nums, 0, [], ans)
        return ans

    =================================================================
    class Solution(object):
        def subsets(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """

            subsets = []
            n = len(nums)
            nums.sort()
            # We know there are totally 2^n subsets,
            # becase every num may in or not in one subsets.
            # So we check the jth(0<=j<n) bit for every ith(0=<i<2^n) subset.
            # If jth bit is 1, then nums[j] in the subset.
            sum_sets = 2 ** n
            for i in range(sum_sets):
                cur_set = []
                for j in range(n):
                    power = 2 ** j
                    if i & power == power:
                        cur_set.append(nums[j])

                subsets.append(cur_set)

            return subsets

    """
    [0]
    []
    [1,2,3,4,7,8]
    """



136. single number
------------------------------------------------

.. code-block:: python


    Given an array of integers, every element appears twice except for one. Find that single one.


    Note:
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

    =================================================================
    class Solution(object):
      def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
          nums[0] ^= nums[i]
        return nums[0]

    =================================================================
    class Solution(object):
        def singleNumber(self, nums):
            num = nums[0]
            for i in nums[1:]:
                num = num ^ i
            return num

    """
    [1]
    [1,2,3,4,4,3,2]
    """



137. single number 2
------------------------------------------------

.. code-block:: python


    Given an array of integers, every element appears three times except for one, which appears exactly once. Find that single one.



    Note:
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

    =================================================================
    class Solution(object):
      def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        def singleNumberK(nums, k):
          ret = 0
          count = [0] * 32
          for i in range(0, 32):
            for num in nums:
              if num & (1 << i):
                count[i] += 1
            if count[i] % 3 != 0:
              ret |= 1 << i
          if ret > 0x7fffffff:
            ret -= 0x100000000
          return ret

        return singleNumberK(nums, 3)

    =================================================================
    class Solution(object):
        """
        If you sum the ith bit of all numbers and mod 3,
        it must be either 0 or 1 due to the constraint of this problem
        where each number must appear either three times or once.
        This will be the ith bit of that "single number".

        Refer to:
        https://discuss.leetcode.com/topic/455/constant-space-solution
        """
        def singleNumber(self, nums):
            bit_record = [0] * 32
            result = 0
            for i in range(32):
                for n in nums:
                    bit_record[i] += (n >> i) & 0x1
                bit_val = bit_record[i] % 3
                result |= bit_val << i

            # Int in python is an object and has no upper limit,
            # If you do 1<<31, you get 2147483648 other than -2147483648
            return result - 2**32 if result >= 2**31 else result


    class Solution_2(object):
        """
        Use two-bits represents the sum(should be 0/3, 1, 2) of all num's i-th bit.
        Twice-Once(the two bits): 00(0, 3)-->01(1)-->10(2)-->00(0, 3)
        Then we need to set rules for 'once' and 'twice' so that they act as we hopes.
            once = once ^ n & (~twice)
            twice = twice ^ n & (~once)

        Since each of the 32 bits follow the same rules,
        we can calculate them all at once.  Refer to:
        https://discuss.leetcode.com/topic/2031/challenge-me-thx/17
        """
        def singleNumber(self, nums):
            once, twice = 0, 0
            for n in nums:
                once = once ^ n & (~twice)
                twice = twice ^ n & (~once)
            return once


    """
    [1]
    [1,1,3,1]
    [1,1,1,2,2,2,3,4,4,4]
    [-2,-2,1,1,-3,1,-3,-3,-4,-2]
    """



201. bitwise and of numbers range
------------------------------------------------

.. code-block:: python

    Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.


    For example, given the range [5, 7], you should return 4.


    Credits:Special thanks to @amrsaqr for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        while m < n:
          n = n & n - 1
        return n

    =================================================================
    class Solution(object):
        """ Refer to
        https://leetcode.com/discuss/32115/bit-operation-solution-java

        The idea is very simple:
            1. last bit of (odd number & even number) is 0.
            2. when m != n, there is at least an odd number and an even number,
            so the last bit position result is 0;
            3. when m == n: just return m.

        For example: m = xy, n = xz, m < n, so y < z. Here x, y, z are some bits.
        And x is all the shared bits of the high position.
        y < z, so bitwise AND of all numbers in [xy, xz] is x0...0
        """
        # Recursive
        def rangeBitwiseAnd(self, m, n):
            if m == n:
                return m
            else:
                return self.rangeBitwiseAnd(m >> 1, n >> 1) << 1

        # Iteration
        def rangeBitwiseAnd_1(self, m, n):
            if m == 0:
                return 0
            trans_count = 0
            while m < n:
                m >>= 1
                n >>= 1
                trans_count += 1
            return m << trans_count

        # Another simple solution
        def rangeBitwiseAnd_2(self, m, n):
            while m < n:
                n = n & (n-1)
            return n

    """
    0
    0
    12
    12
    0
    2147483647
    """



260. single number 3
------------------------------------------------

.. code-block:: python


    Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.


    For example:


    Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].


    Note:

    The order of the result is not important. So in the above example, [5, 3] is also correct.
    Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?



    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        xor = 0
        for num in nums:
          xor ^= num

        xor = xor & -xor
        a, b = 0, 0
        for num in nums:
          if num & xor:
            a ^= num
          else:
            b ^= num

        return a, b

    =================================================================
    class Solution(object):
        # Clear explanation according to
        # https://leetcode.com/discuss/60408/sharing-explanation-of-the-solution
        def singleNumber(self, nums):
            xor_res = 0
            for num in nums:
                xor_res ^= num

            # Assume the two different numbers diff at ith bit(i is the rightmost).
            # Then we can get 0x000...1...000, 1 is the ith bit by the following.
            xor_res &= -xor_res
            num_a, num_b = 0, 0
            for num in nums:
                # All the numbers can be partitioned into
                # two groups according to their bits at location i.
                # The first group consists of all numbers whose bits at i is 0.
                # The second group consists of all numbers whose bits at i is 1.
                # The two different number a and b is in the two different groups.
                if num & xor_res == 0:
                    num_a ^= num
                else:
                    num_b ^= num
            return [num_a, num_b]

    """
    [-1,0]
    [1, 2, 1, 3, 2, 5]
    [-1,-1,-2,-2,-3,-3,-3,-3,4,-5]
    """



318. maximum product from all buildings
------------------------------------------------

.. code-block:: python


    Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters.
    You may assume that each word will contain only lower case letters.
    If no such two words exist, return 0.



    Example 1:


    Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
    Return 16
    The two words can be "abcw", "xtfn".


    Example 2:


    Given ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]
    Return 4
    The two words can be "ab", "cd".


    Example 3:


    Given ["a", "aa", "aaa", "aaaa"]
    Return 0
    No such pair of words.


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        bitmap = [0] * len(words)
        mask = 0x01
        ans = 0
        for i in range(0, len(words)):
          word = words[i]
          for c in word:
            bitmap[i] |= (mask << (ord(c) - ord('a')))
        for i in range(0, len(words)):
          for j in range(0, i):
            if bitmap[i] & bitmap[j] == 0:
              ans = max(ans, len(words[i]) * len(words[j]))

        return ans

    =================================================================
    class Solution(object):
        def maxProduct(self, words):
            max_product = 0
            length = len(words)
            bit_record = [0] * length
            # Use 1bit to represent each letter, and
            # use 32bit(Int variable, bitMap[i]) to represent the set of each word
            for i in xrange(length):
                for c in words[i]:
                    bit_record[i] |= 1 << (ord(c) - ord("a"))

            for i in xrange(length):
                for j in xrange(i+1, length):
                    # If the AND of two bitmap element equals to 0,
                    # these two words do not have same letter.
                    if not bit_record[i] & bit_record[j]:
                        product = len(words[i]) * len(words[j])
                        if product > max_product:
                            max_product = product
            return max_product

    """
    []
    ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
    ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]
    ["a", "aa", "aaa", "aaaa"]
    """


338. Counting bits
------------------------------------------------

.. code-block:: python


    Given a non negative integer number num. For every numbers i in the range 0 &le; i &le; num calculate the number of 1's in their binary representation and return them as an array.


    Example:
    For num = 5 you should return [0,1,1,2,1,2].


    Follow up:

    It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
    Space complexity should be O(n).
    Can you do it like a boss? Do it without using any builtin function like __builtin_popcount  in c++ or in any other language.



    Credits:Special thanks to @ syedee  for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        if num == 0:
          return [0]
        ans = [0, 1]
        j = 0
        for i in range(2, num + 1):
          ans.append(ans[i & (i - 1)] + 1)
        return ans

    =================================================================
    class Solution(object):
        def countBits(self, num):
            """
            f[i] = f[i / 2] + i % 2
            or
            f[i] = f[i&(i-1)] + 1, i&(i-1) drops the lowest set bit
            """
            ans = [0] * (num + 1)
            for i in xrange(1, num + 1):
                ans[i] = ans[i >> 1] + (i & 0x1)
                # ans[i] = ans[i & (i - 1)] + 1
            return ans

    """
    0
    1
    12
    """


