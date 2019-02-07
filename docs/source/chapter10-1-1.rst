Dynamic Programming - Easy 2
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

10. Regular Expression Matching
-------------------------------------

.. code-block:: python

    Implement regular expression matching with support for '.' and '*'.


    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.

    The matching should cover the entire input string (not partial).

    The function prototype should be:
    bool isMatch(const char *s, const char *p)

    Some examples:
    isMatch("aa","a") �넂 false
    isMatch("aa","aa") �넂 true
    isMatch("aaa","aa") �넂 false
    isMatch("aa", "a*") �넂 true
    isMatch("aa", ".*") �넂 true
    isMatch("ab", ".*") �넂 true
    isMatch("aab", "c*a*b") �넂 true


    =================================================================
    class Solution(object):
      def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        dp[0][0] = True
        for j in range(1, len(p) + 1):
          if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]

        for i in range(1, len(s) + 1):
          for j in range(1, len(p) + 1):
            if p[j - 1] != "*":
              dp[i][j] = dp[i - 1][j - 1] and (s[i - 1] == p[j - 1] or p[j - 1] == ".")
            else:
              dp[i][j] = dp[i][j - 2] or dp[i - 1][j] and (p[j - 2] == s[i - 1] or p[j - 2] == ".")
        return dp[-1][-1]

    =================================================================
    class Solution(object):
        def isMatch(self, s, p):
            """
            Dynamic Programming
            dp[i][j] represents isMatch(p[0...i], s[0...j]), default is False;
                dp[i][-1] represents isMatch(p[0...i], "")
                dp[-1][j] represents isMatch("", s[0...j])
            """
            if not s:
                # .*.*.*.* Return True, others return False.
                if len(p) % 2 != 0:
                    return False
                for k in range(1, len(p), 2):
                    if p[k] != "*":
                        return False
                return True

            # dp = [[False] * (len(s)+1)] * (len(p)+1)
            dp = [[False for col in range(len(s) + 1)]
                  for row in range(len(p) + 1)]
            dp[-1][-1] = True

            for i in range(len(p)):
                for j in range(len(s)):
                    """
                    p[i] is "*", so dp[i][j] =
                        1. dp[i-2][j]      # * matches 0 element in s;
                        2. dp[i-2][j-1]    # * matches 1 element in s;
                        3. dp[i][j-1]      # * matches more than one in s.
                    """
                    if p[i] == "*":
                        m_0 = dp[i - 2][j]
                        m_1 = (p[i - 1] == "." or p[i - 1] == s[j]) and dp[i - 2][j - 1]
                        m_more = (p[i - 1] == "." or p[i - 1] == s[j]) and dp[i][j - 1]
                        dp[i][j] = m_0 or m_1 or m_more

                        # p[i] matches "" is equal p[i-2] matches "".
                        dp[i][-1] = dp[i - 2][-1]

                    else:
                        dp[i][j] = (dp[i - 1][j - 1] and
                                    (p[i] == s[j] or p[i] == "."))
                        # p[i] doesn't match ""
                        dp[i][-1] = False

            return dp[len(p) - 1][len(s) - 1]


    """
    "aaa"
    "ab*a"
    ""
    "c*c*"
    "aaa"
    "aaaa"
    "aaabc"
    "a*bc"
    "aab"
    "c*a*b"
    "ab"
    ".*c"
    "aaaaabaccbbccababa"
    "a*b*.*c*c*.*.*.*c"
    """



32. Longest Valid Parentheses
-------------------------------------

.. code-block:: python

    Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.


    For "(()", the longest valid parentheses substring is "()", which has length = 2.


    Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.


    =================================================================
    class Solution(object):
      def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = [0 for _ in range(0, len(s))]
        left = 0
        ans = 0
        for i in range(0, len(s)):
          if s[i] == "(":
            left += 1
          elif left > 0:
            left -= 1
            dp[i] = dp[i - 1] + 2
            j = i - dp[i]
            if j >= 0:
              dp[i] += dp[j]
            ans = max(ans, dp[i])
        return ans


    =================================================================
    class Solution(object):
        def longestValidParentheses(self, s):
            """
            According to:
            https://leetcode.com/discuss/8092/my-dp-o-n-solution-without-using-stack

            dp[i]: the longest length of valid parentheses which ends at i. Then:

            1. s[i] is '(', dp[i] = 0
            2. s[i] is ')'
                a. s[i-dp[i-1]-1] == '(': dp = dp[i-1] + 2 + dp[i-dp[i-1]-2]
                b. dp[i] = 0

            Just think about what does s[i-dp[i-1]-1] == '(' mean.
            """
            if not s:
                return 0

            dp = [0] * len(s)
            max_len = 0
            for i in xrange(1, len(s)):
                if s[i] == ")" and i - 1 - dp[i - 1] >= 0 and s[i - 1 - dp[i - 1]] == "(":
                    dp[i] = dp[i - 1] + 2 + dp[i - dp[i - 1] - 2]
                    max_len = max(max_len, dp[i])

            return max_len

    """
    ""
    ")"
    "()"
    "))"
    "(((()()()))("
    "(((()()()))())"
    """


44. WildCard Matching
-------------------------------------

.. code-block:: python

    Implement wildcard pattern matching with support for '?' and '*'.


    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).

    The matching should cover the entire input string (not partial).

    The function prototype should be:
    bool isMatch(const char *s, const char *p)

    Some examples:
    isMatch("aa","a") &rarr; false
    isMatch("aa","aa") &rarr; true
    isMatch("aaa","aa") &rarr; false
    isMatch("aa", "*") &rarr; true
    isMatch("aa", "a*") &rarr; true
    isMatch("ab", "?*") &rarr; true
    isMatch("aab", "c*a*b") &rarr; false


    =================================================================
    class Solution(object):
      def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        i = j = 0
        lenS = len(s)
        lenP = len(p)
        lastMatchPos = 0
        lastStarPos = -1
        while i < len(s):
          if j < lenP and p[j] in (s[i], "?"):
            i += 1
            j += 1
          elif j < lenP and p[j] == "*":
            lastMatchPos = i
            lastStarPos = j
            j += 1
          elif lastStarPos > -1:
            i = lastMatchPos + 1
            lastMatchPos += 1
            j = lastStarPos + 1
          else:
            return False
        while j < lenP and p[j] == "*":
          j += 1
        return j == lenP


    =================================================================
    class Solution(object):
        def isMatch(self, s, p):
            """ Dynamic Programming

            dp[i][j] represents isMatch(p[0...i-1], s[0...j-1]), default is False;
            dp[i][0]: isMatch(p[0...i], ""), dp[0][j]: isMatch("", s[0...j])
            dp[0][0] represents

            If p[i] is "*", dp[i+1][j+1] =
                1. dp[i][j+1]        # * matches 0 element in s;
                2. dp[i][j]          # * matches 1 element in s;
                3. dp[i+1][j]        # * matches more than one in s.
            """
            if not s:
                if p.count('*') != len(p):
                    return False
                return True

            # Optimized for the big data.
            if len(p) - p.count('*') > len(s):
                return False

            # Initinal process
            dp = [[False for col in range(len(s) + 1)] for row in range(len(p) + 1)]
            dp[0][0] = True     # isMatch("", "") = True
            for i in range(len(p)):
                dp[i + 1][0] = dp[i][0] and p[i] == '*'

            for i in range(len(p)):
                for j in range(len(s)):
                    if p[i] == "*":
                        dp[i + 1][j + 1] = dp[i][j + 1] or dp[i][j] or dp[i + 1][j]
                    else:
                        dp[i + 1][j + 1] = dp[i][j] and (p[i] == s[j] or p[i] == "?")

            return dp[len(p)][len(s)]

    """
    "aa"
    "a"
    "aa"
    "aa"
    "aaa"
    "aa"
    "aa"
    "*"
    "aa"
    "a*"
    "ab"
    "?*"
    "aab"
    "c*a*b"
    """



53. Maximum subarray
-------------------------------------

.. code-block:: python

    Find the contiguous subarray within an array (containing at least one number) which has the largest sum.


    For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
    the contiguous subarray [4,-1,2,1] has the largest sum = 6.


    click to show more practice.

    More practice:

    If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.


    =================================================================
    class Solution(object):
      def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
          return 0
        preSum = maxSum = nums[0]
        for i in range(1, len(nums)):
          preSum = max(preSum + nums[i], nums[i])
          maxSum = max(maxSum, preSum)
        return maxSum

    =================================================================
    class Solution(object):
        # O(n) space
        def maxSubArray(self, nums):
            num_len = len(nums)
            # dp[i]: Largest sum of contiguous subarray start from i
            dp = [-1] * num_len
            max_sum = dp[num_len - 1] = nums[num_len - 1]

            for i in range(num_len - 2, -1, -1):
                dp[i] = max(nums[i], dp[i+1]+nums[i])
                max_sum = max(dp[i], max_sum)

            return max_sum


    class Solution_2(object):
        # DP same with the previous, but O(1) space
        def maxSubArray(self, nums):
            num_len = len(nums)
            max_sum = pre_sum = nums[num_len - 1]

            for i in range(num_len - 2, -1, -1):
                pre_sum = max(nums[i], pre_sum+nums[i])
                max_sum = max(pre_sum, max_sum)

            return max_sum

    """
    [-1]
    [1]
    [-9,-2,-3,-5,-3]
    [-2,1,-3,4,-1,2,1,-5,4]
    """



70. Climbing Stairs
-------------------------------------

.. code-block:: python

    You are climbing a stair case. It takes n steps to reach to the top.

    Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?


    Note: Given n will be a positive integer.


    =================================================================
    class Solution(object):
      def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
          return 1
        pre, ppre = 1, 1
        for i in range(2, n + 1):
          tmp = pre
          pre = ppre + pre
          ppre = tmp
        return pre


    =================================================================
    class Solution(object):
        def climbStairs(self, n):
            """
            :type n: int
            :rtype: int
            """
            if not n:
                return 1

            dp = [0 for i in range(n)]
            dp[0] = 1
            if n > 1:
                dp[1] = 2
            for i in range(2, n):
                dp[i] = dp[i-1] + dp[i-2]

            return dp[n-1]



72. Edit Distance
-------------------------------------

.. code-block:: python

    Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)



    You have the following 3 operations permitted on a word:



    a) Insert a character
    b) Delete a character
    c) Replace a character


    =================================================================
    class Solution(object):
      def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if len(word1) == 0 or len(word2) == 0:
          return max(len(word1), len(word2))

        dp = [[0] * (len(word2) + 1) for _ in range(0, len(word1) + 1)]
        dp[0][0] = 0

        for i in range(0, len(word1) + 1):
          for j in range(0, len(word2) + 1):
            if i == 0:
              dp[i][j] = j
            elif j == 0:
              dp[i][j] = i
            else:
              cond1 = dp[i][j - 1] + 1
              cond2 = dp[i - 1][j] + 1
              cond3 = 0
              if word1[i - 1] == word2[j - 1]:
                cond3 = dp[i - 1][j - 1]
              else:
                cond3 = dp[i - 1][j - 1] + 1
              dp[i][j] = min(cond1, cond2, cond3)
        return dp[-1][-1]


    =================================================================
    class Solution(object):
        def minDistance(self, word1, word2):
            """
            :type word1: str
            :type word2: str
            :rtype: int
            """
            len_w1 = len(word1)
            len_w2 = len(word2)

            # dp[i][j]: minimum number of steps convert word1[0,i) to word2[0,j)
            dp = [[0 for j in range(len_w2+1)] for i in range(len_w1+1)]

            # initial the dp array
            dp[0][0] = 0
            for j in range(1, len_w2+1):
                dp[0][j] = j
            for i in range(1, len_w1+1):
                dp[i][0] = i

            for i in range(1, len_w1+1):
                for j in range(1, len_w2+1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(
                            dp[i-1][j-1] + 1,
                            dp[i-1][j] + 1,
                            dp[i][j-1] + 1,)

            return dp[len_w1][len_w2]



87. Scramble String
-------------------------------------

.. code-block:: python

    Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.


    Below is one possible representation of s1 = "great":


        great
       /    \
      gr    eat
     / \    /  \
    g   r  e   at
               / \
              a   t


    To scramble the string, we may choose any non-leaf node and swap its two children.


    For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".


        rgeat
       /    \
      rg    eat
     / \    /  \
    r   g  e   at
               / \
              a   t


    We say that "rgeat" is a scrambled string of "great".


    Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".


        rgtae
       /    \
      rg    tae
     / \    /  \
    r   g  ta  e
           / \
          t   a


    We say that "rgtae" is a scrambled string of "great".


    Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.


    =================================================================
    class Solution(object):
      def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        n = len(s1)
        m = len(s2)
        if sorted(s1) != sorted(s2):
          return False

        if n < 4 or s1 == s2:
          return True

        for i in range(1, n):
          if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
            return True
          if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i]):
            return True
        return False

    =================================================================
    class Solution(object):
        def isScramble(self, s1, s2):
            """
            :type s1: str
            :type s2: str
            :rtype: bool
            """
            if (len(s1) != len(s2)) or not len(s1) or not len(s2):
                return False

            if s1 == s2:
                return True

            str_l = len(s1)
            # dp[l][i][j]: whether s1[i:i+l+1] is a scrambled string of s2[j:j+l+1]
            dp = [[[False for i in xrange(str_l)]
                   for j in xrange(str_l)] for l in xrange(str_l)]

            # Initialization: dp[0][i][j], s1[i] is a scrambled string of s2[j]
            for i in xrange(str_l):
                for j in xrange(str_l):
                    dp[0][i][j] = True if s1[i] == s2[j] else False

            for l in xrange(1, str_l):
                # The length of current substring is l+1
                for i in xrange(str_l-l):
                    for j in xrange(str_l-l):
                        # Split the l+1 string into two parts,
                        # k is the length of first part, so 1 <= k <= l;
                        for k in range(1, l+1):
                            scramble_1 = dp[k-1][i][j] and dp[l-k][k+i][k+j]
                            scramble_2 = dp[k-1][i][j+l-k+1] and dp[l-k][i+k][j]
                            dp[l][i][j] = (scramble_1 or scramble_2)
                            if dp[l][i][j]:
                                break
            return dp[str_l-1][0][0]

    """
    "great"
    "rgeta"
    "great"
    "rgtae"
    """

    # Implement with recursion
    """
    class Solution(object):
        def isScramble(self, s1, s2):
            if (len(s1) != len(s2)) or not len(s1) or not len(s2):
                return False

            if sorted(s1) != sorted(s2):
                return False

            if s1 == s2:
                return True

            length = len(s1)
            for i in range(1, length):
                if (self.isScramble(s1[:i],s2[:i])
                    and self.isScramble(s1[i:],s2[i:])):
                    return True
                if (self.isScramble(s1[:i],s2[length-i:])
                    and self.isScramble(s1[i:],s2[:length-i])):
                    return True
            return False
    """



91. Decode Ways
-------------------------------------

.. code-block:: python

    A message containing letters from A-Z is being encoded to numbers using the following mapping:



    'A' -> 1
    'B' -> 2
    ...
    'Z' -> 26



    Given an encoded message containing digits, determine the total number of ways to decode it.



    For example,
    Given encoded message "12",
    it could be decoded as "AB" (1 2) or "L" (12).



    The number of ways decoding "12" is 2.


    =================================================================
    class Solution(object):
      def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
          return 0
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        dp[1] = 0 if s[0] == "0" else 1
        for i in range(1, len(s)):
          pre = int(s[i - 1])
          cur = int(s[i])
          num = pre * 10 + cur
          if cur != 0:
            dp[i + 1] += dp[i]
          if pre != 0 and 0 < num <= 26:
            dp[i + 1] += dp[i - 1]

        return dp[-1]

    =================================================================
    class Solution(object):

        def numDecodings(self, s):
            """
            :type s: str
            :rtype: int
            """
            if not s or s[0] == "0":
                return 0

            len_s = len(s)
            # dp[i]: total number of ways to decode s[0:i)
            dp = [1 for i in range(len_s + 1)]
            for i in range(1, len_s):
                pre_num = ord(s[i - 1]) - ord('0')
                cur_num = ord(s[i]) - ord('0')
                num = pre_num * 10 + cur_num

                if cur_num == 0:
                    if num > 26 or num == 0:
                        return 0
                    else:
                        dp[i+1] = dp[i-1]

                else:
                    if num <= 26 and pre_num != 0:
                        dp[i + 1] = dp[i] + dp[i - 1]
                    else:
                        dp[i + 1] = dp[i]

            return dp[len_s]

    """
    ""
    "123"
    "1238"
    "172731349111222"
    "0"
    "10203"
    """


97. Interleaving String
-------------------------------------

.. code-block:: python

    Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.



    For example,
    Given:
    s1 = "aabcc",
    s2 = "dbbca",


    When s3 = "aadbbcbcac", return true.
    When s3 = "aadbbbaccc", return false.

    =================================================================
    class Solution(object):
      def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        d = {}
        s3 = list(s3)
        if len(s1) + len(s2) != len(s3):
          return False

        def dfs(s1, i, s2, j, d, path, s3):
          if (i, j) in d:
            return d[(i, j)]

          if path == s3:
            return True

          if i < len(s1):
            if s3[i + j] == s1[i]:
              path.append(s1[i])
              if dfs(s1, i + 1, s2, j, d, path, s3):
                return True
              path.pop()
              d[(i + 1, j)] = False

          if j < len(s2):
            if s3[i + j] == s2[j]:
              path.append(s2[j])
              if dfs(s1, i, s2, j + 1, d, path, s3):
                return True
              path.pop()
              d[(i, j + 1)] = False

          return False

        return dfs(s1, 0, s2, 0, d, [], s3)

    =================================================================
    class Solution(object):
        def isInterleave(self, s1, s2, s3):
            """
            :type s1: str
            :type s2: str
            :type s3: str
            :rtype: bool
            """
            if not s1:
                if s2 == s3:
                    return True
                else:
                    return False
            s1_l = len(s1)
            s2_l = len(s2)
            s3_l = len(s3)
            if s3_l != s1_l + s2_l:
                return False

            # dp[i][j] is true when s3[i+j-1] is formed by the interleaving of
            # s1[:i](previous i chars of s1) and s2[:j](previous j chars of s2).
            dp = [[False for j in xrange(s2_l+1)] for i in xrange(s1_l+1)]
            dp[0][0] = True

            for i in xrange(1, s1_l+1):
                if s1[i-1] == s3[i-1]:
                    dp[i][0] = True
                else:
                    break

            for j in xrange(1, s2_l+1):
                if s2[j-1] == s3[j-1]:
                    dp[0][j] = True
                else:
                    break

            for i in xrange(1, s1_l+1):
                for j in xrange(1, s2_l+1):
                    if (s1[i-1] == s3[i+j-1] and dp[i-1][j] or
                            s2[j-1] == s3[i+j-1] and dp[i][j-1]):
                        dp[i][j] = True

            return dp[s1_l][s2_l]

    """
    ""
    "a"
    "a"
    "aa"
    "ab"
    "abaa"
    "aabcc"
    "dbbca"
    "aadbbbaccc"
    "aaaabbbb"
    "ddaacccc"
    "addaacaaabbbcccb"
    """



