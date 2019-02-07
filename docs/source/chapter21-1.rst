String - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

5. Longest palindromic substring
------------------------------------

.. code-block:: python

    Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

    Example:

    Input: "babad"

    Output: "bab"

    Note: "aba" is also a valid answer.



    Example:

    Input: "cbbd"

    Output: "bb"


    =================================================================
    class Solution(object):
      def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        left = right = 0
        n = len(s)
        for i in range(n - 1):
          if 2 * (n - i) + 1 < right - left + 1:
            break
          l = r = i
          while l >= 0 and r < n and s[l] == s[r]:
            l -= 1
            r += 1
          if r - l - 2 > right - left:
            left = l + 1
            right = r - 1
          l = i
          r = i + 1
          while l >= 0 and r < n and s[l] == s[r]:
            l -= 1
            r += 1
          if r - l - 2 > right - left:
            left = l + 1
            right = r - 1
        return s[left:right + 1]


    =================================================================
    class Solution(object):
        # Easy to understand.  Refer to
        # https://leetcode.com/discuss/32204/simple-c-solution-8ms-13-lines
        def longestPalindrome(self, s):
            if not s:
                return ""
            s_len = len(s)
            if s_len == 1:
                return s
            max_start, max_end = 0, 1   # Make sure s[start:end] is palindrome
            i = 0
            while i < s_len:
                # No need to check the remainming, pruning here
                if max_end - max_start >= (s_len-i) * 2 - 1:
                    break
                left, right = i, i+1
                # Skip duplicate characters i.
                while right < s_len and s[right-1] == s[right]:
                    right += 1
                i = right
                while left-1 >= 0 and right < s_len and s[left-1] == s[right]:
                    left -= 1
                    right += 1
                if right-left > max_end-max_start:
                    max_start = left
                    max_end = right
            return s[max_start:max_end]

    """
    ""
    "a"
    "aa"
    "dcc"
    "aaaa"
    "cccd"
    "ccccdc"
    "abcdefead"
    """



6. Zigzag Conversion
-----------------------

.. code-block:: python

    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
    A P L S I I G
    Y   I   R


    And then read line by line: "PAHNAPLSIIGYIR"


    Write the code that will take a string and make this conversion given a number of rows:

    string convert(string text, int nRows);

    convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".

    =================================================================
    class Solution(object):
      def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows <= 1:
          return s
        n = len(s)
        ans = []
        step = 2 * numRows - 2
        for i in range(numRows):
          one = i
          two = -i
          while one < n or two < n:
            if 0 <= two < n and one != two and i != numRows - 1:
              ans.append(s[two])
            if one < n:
              ans.append(s[one])
            one += step
            two += step
        return "".join(ans)


    =================================================================
    class Solution(object):
        def convert(self, s, numRows):
            """
            :type s: str
            :type numRows: int
            :rtype: str
            """
            if not s:
                return ""
            if numRows == 1:
                return s

            len_s = len(s)
            zigzag_list = []
            magic_number = 2 * numRows - 2

            for row in range(numRows):
                index = row
                while index < len_s:
                    zigzag_list.append(s[index])
                    if row != 0 and row != numRows - 1:
                        next_num = magic_number + index - 2 * row
                        if next_num < len_s:
                            zigzag_list.append(s[next_num])
                    index += magic_number

            return "".join(zigzag_list)

    """
    ""
    1
    "ABC"
    1
    "PAYPALISHIRING"
    5
    """


8. String to integer atoi
---------------------------

.. code-block:: python

    Implement atoi to convert a string to an integer.

    Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.


    Notes:
    It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.


    Update (2015-02-10):
    The signature of the C++ function had been updated. If you still see your function signature accepts a const char * argument, please click the reload button  to reset your code definition.


    spoilers alert... click to show requirements for atoi.

    Requirements for atoi:

    The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

    The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

    If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

    If no valid conversion could be performed, a zero value is returned. If the correct value is out of the range of representable values, INT_MAX (2147483647) or INT_MIN (-2147483648) is returned.



    =================================================================
    class Solution(object):
      def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        s = s.strip()
        sign = 1
        if not s:
          return 0
        if s[0] in ["+", "-"]:
          if s[0] == "-":
            sign = -1
          s = s[1:]
        ans = 0
        for c in s:
          if c.isdigit():
            ans = ans * 10 + int(c)
          else:
            break
        ans *= sign
        if ans > 2147483647:
          return 2147483647
        if ans < -2147483648:
          return -2147483648
        return ans


    =================================================================
    class Solution(object):
        MAX_INT = 2**31 - 1
        MIN_INT = - 2**31

        def myAtoi(self, strs):
            """ We need to handle four cases:

            1. discards all leading whitespaces
            2. sign of the number
            3. overflow
            4. invalid input
            """
            strs = strs.strip()
            if not strs:
                return 0

            sign, i = 1, 0
            if strs[i] == '+':
                i += 1
            elif strs[i] == '-':
                i += 1
                sign = -1

            num = 0
            while i < len(strs):
                if strs[i] < '0' or strs[i] > '9':
                    break
                if num > self.MAX_INT or (num * 10 + int(strs[i]) > self.MAX_INT):
                    return self.MAX_INT if sign == 1 else self.MIN_INT
                else:
                    num = num * 10 + int(strs[i])
                i += 1

            return num * sign

    """
    ""
    "  12a"
    "  a12"
    "  +12"
    "  +-12"
    "2147483648"
    "-2147483648"
    """



14. Longest Common Prefix
------------------------------

.. code-block:: python

    Write a function to find the longest common prefix string amongst an array of strings.

    =================================================================
    class Solution(object):
      def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
          return ""
        i = 0
        j = 0
        end = 0
        while j < len(strs) and i < len(strs[j]):
          if j == 0:
            char = strs[j][i]
          else:
            if strs[j][i] != char:
              break

          if j == len(strs) - 1:
            i += 1
            j = 0
            end += 1
          else:
            j += 1

        return strs[j][:end]


    =================================================================
    class Solution(object):
        def longestCommonPrefix(self, strs):
            """
            :type strs: List[str]
            :rtype: str
            """
            if not strs:
                return ""
            common_prefix = strs[0]
            for string in strs:
                min_len = min(len(string), len(common_prefix))
                mark = 0        # Record the longest commen prefix index right now.
                for i in range(min_len):
                    mark = i
                    if string[i] == common_prefix[i]:
                        i += 1
                        mark = i
                    else:
                        if i == 0:
                            return ""
                        break
                common_prefix = common_prefix[:mark]

            return common_prefix

    """
    []
    ["abcd", "abc", "ab","acdef"]
    ["abc", "abcd", "d"]
    """


17. Letter combination of a phone number
---------------------------------------------

.. code-block:: python

    Given a digit string, return all possible letter combinations that the number could represent.



    A mapping of digit to letters (just like on the telephone buttons) is given below.



    Input:Digit string "23"
    Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].



    Note:
    Although the above answer is in lexicographical order, your answer could be in any order you want.


    =================================================================
    class Solution(object):
      def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits) == 0:
          return []

        d = {1: "", 2: "abc", 3: "def", 4: "ghi", 5: "jkl", 6: "mno", 7: "pqrs", 8: "tuv", 9: "wxyz"}

        def dfs(digits, index, path, res, d):
          if index == len(digits):
            res.append("".join(path))
            return

          digit = int(digits[index])
          for c in d.get(digit, []):
            path.append(c)
            dfs(digits, index + 1, path, res, d)
            path.pop()

        res = []
        dfs(digits, 0, [], res, d)
        return res


    =================================================================
    class Solution(object):
        def letterCombinations(self, digits):
            """
            :type digits: str
            :rtype: List[str]
            """

            phone_letters = {0: [" "],
                             1: ["*"],
                             2: ["a", "b", "c"],
                             3: ["d", "e", "f"],
                             4: ["g", "h", "i"],
                             5: ["j", "k", "l"],
                             6: ["m", "n", "o"],
                             7: ["p", "q", "r", "s"],
                             8: ["t", "u", "v"],
                             9: ["w", "x", "y", "z"],
                             }
            if digits:
                all_str = phone_letters[ord(digits[0]) - ord("0")]
            else:
                return []

            for i in range(1, len(digits)):
                all_str = self.combination(
                    all_str,
                    phone_letters[ord(digits[i]) - ord("0")])

            return all_str

        # return string which combines a in str_a with b in str_b
        def combination(self, str_a, str_b):
            combine_str = []
            for a in str_a:
                for b in str_b:
                    combine_str.append(a + b)

            return combine_str

    """
    ""
    "37"
    "1234"
    """



28. Implement StrStr
------------------------

.. code-block:: python

    Implement strStr().


    Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

    =================================================================
    class Solution(object):
      def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(haystack) == len(needle):
          if haystack == needle:
            return 0
          else:
            return -1

        for i in range(0, len(haystack)):
          k = i
          j = 0
          while j < len(needle) and k < len(haystack) and haystack[k] == needle[j]:
            j += 1
            k += 1
          if j == len(needle):
            return i
        return -1 if needle else 0


    =================================================================
    lass Solution(object):
        # Not notoriously hard-to-understand algorithm KMP
        def strStr(self, haystack, needle):
            # Return 0 if needle is ""
            if not needle:
                return 0
            length = len(haystack)

            # If one char in haystack is same with needle[0],
            # then verify the other chars in needle.
            for i in range(length-len(needle)+1):
                if haystack[i:i+len(needle)] == needle:
                    return i

            return -1

    """
    ""
    ""
    "abaa"
    "aa"
    "aaabbb"
    "abbb"
    """



38. Count and Say
--------------------

.. code-block:: python

    The count-and-say sequence is the sequence of integers with the first five terms as following:

    1.     1
    2.     11
    3.     21
    4.     1211
    5.     111221



    1 is read off as "one 1" or 11.
    11 is read off as "two 1s" or 21.
    21 is read off as "one 2, then one 1" or 1211.



    Given an integer n, generate the nth term of the count-and-say sequence.



    Note: Each term of the sequence of integers will be represented as a string.


    Example 1:

    Input: 1
    Output: "1"



    Example 2:

    Input: 4
    Output: "1211"



    =================================================================
    class Solution(object):
      def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        ans = "1"
        n -= 1
        while n > 0:
          res = ""
          pre = ans[0]
          count = 1
          for i in range(1, len(ans)):
            if pre == ans[i]:
              count += 1
            else:
              res += str(count) + pre
              pre = ans[i]
              count = 1
          res += str(count) + pre
          ans = res
          n -= 1
        return ans


    =================================================================
    class Solution(object):
        """ Quite straight-forward solution.

        We generate k-th string, and from k-th string we generate k+1-th string,
        until we generate n-th string.
        """
        def countAndSay(self, n):
            if n <= 1:
                return "1"

            pre_str = "1"
            for i in range(2, n + 1):
                # Get the ith count-and-say sequence by scan pre_str
                length = len(pre_str)
                current_str = ""

                # Count and say the pre_str
                index = 0
                while index < length:
                    char = pre_str[index]
                    repeat = 0
                    pos = index + 1
                    while pos < length and pre_str[pos] == char:
                        repeat += 1
                        pos += 1

                    current_str += str(repeat + 1) + char
                    index = pos

                pre_str = current_str

            return pre_str

    """
    1
    5
    15
    """



58. Length of Last word
--------------------------------

.. code-block:: python

    Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

    If the last word does not exist, return 0.

    Note: A word is defined as a character sequence consists of non-space characters only.


    For example,
    Given s = "Hello World",
    return 5.


    =================================================================
    class Solution(object):
      def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
          return 0
        s = s.split()
        if len(s) > 0:
          return len(s[-1])
        return 0

    =================================================================
    class Solution(object):
        def lengthOfLastWord(self, s):
            s = s.strip()
            length = 0
            while length < len(s) and s[-(length + 1)] != " ":
                length += 1

            return length


    class Solution_2(object):
        def lengthOfLastWord(self, s):
            return len(s.strip().split(' ')[-1])

    """
    ""
    "are"
    "we are teams"
    "we are teams    "
    """


67. Add Binary
--------------------

.. code-block:: python

    Given two binary strings, return their sum (also a binary string).



    For example,
    a = "11"
    b = "1"
    Return "100".


    =================================================================
    class Solution(object):
      def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        diff = abs(len(a) - len(b))
        if len(a) > len(b):
          b = "0" * diff + b
        else:
          a = "0" * diff + a

        ret = ""
        carry = 0
        ai, bi = len(a) - 1, len(b) - 1
        al, bl = len(a), len(b)
        while ai >= 0 and bi >= 0:
          ac, bc = a[ai], b[bi]
          if ac == "1" and bc == "1":
            if carry == 1:
              ret += "1"
            else:
              ret += "0"
            carry = 1
          elif ac == "0" and bc == "0":
            if carry == 1:
              ret += "1"
            else:
              ret += "0"
            carry = 0
          else:
            if carry == 1:
              ret += "0"
            else:
              ret += "1"

          ai -= 1
          bi -= 1

        if carry == 1:
          ret += "1"
        return ret[::-1]


    =================================================================
    class Solution(object):
        def addBinary(self, a, b):
            """ Recursively binary add.
            """
            if len(a) == 0:
                return b
            if len(b) == 0:
                return a

            if a[-1] == '1' and b[-1] == '1':
                return self.addBinary(self.addBinary(a[:-1], b[:-1]), '1') + '0'
            elif a[-1] == '0' and b[-1] == '0':
                return self.addBinary(a[:-1], b[:-1]) + '0'
            else:
                return self.addBinary(a[:-1], b[:-1]) + '1'


    class Solution_2(object):
        def addBinary(self, a, b):
            """Iteratively way.
            """
            carry_in, index = '0', 0
            result = ""

            while index < max(len(a), len(b)) or carry_in == '1':
                num_a = a[-1 - index] if index < len(a) else '0'
                num_b = b[-1 - index] if index < len(b) else '0'

                val = int(num_a) + int(num_b) + int(carry_in)
                result = str(val % 2) + result
                carry_in = '1' if val > 1 else '0'
                index += 1
            return result


    class Solution_3(object):
        def addBinary(self, a, b):
            return bin(eval("0b" + a) + eval("0b" + b))[2:]


    """
    "0"
    "0"
    "111000"
    "111111111"
    """



68. Text Justification
---------------------------

.. code-block:: python

    Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified.



    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly L characters.



    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.



    For the last line of text, it should be left justified and no extra space is inserted between words.



    For example,
    words: ["This", "is", "an", "example", "of", "text", "justification."]
    L: 16.



    Return the formatted lines as:

    [
       "This    is    an",
       "example  of text",
       "justification.  "
    ]




    Note: Each word is guaranteed not to exceed L in length.



    click to show corner cases.

    Corner Cases:


    A line other than the last line might contain only one word. What should you do in this case?
    In this case, that line should be left-justified.



    =================================================================
    class Solution(object):
      def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        ans = []
        line = []
        lens = map(len, words)
        idx = 0
        curLen = 0
        while idx < len(words):
          if curLen == 0:
            curLen = lens[idx]
          else:
            curLen += lens[idx] + 1
          line.append(words[idx])
          idx += 1
          if curLen > maxWidth:
            curLen = 0
            line.pop()
            idx -= 1
            if len(line) == 1:
              ans.append(line[0] + " " * (maxWidth - len(line[0])))
              line = []
              continue
            spaces = maxWidth - sum(map(len, line))
            avgSpace = spaces / (len(line) - 1)
            extraSpace = spaces % (len(line) - 1)
            res = ""
            for i in range(0, len(line)):
              res += line[i]
              if i < len(line) - 1:
                res += " " * (avgSpace + (extraSpace > 0))
                extraSpace -= 1
            ans.append(res)
            line = []
          elif idx == len(words):
            res = ""
            for i in range(0, len(line)):
              res += line[i]
              if i < len(line) - 1:
                res += " "
            res += " " * (maxWidth - len(res))
            ans.append(res)
        return ans


    =================================================================
    class Solution(object):
        def fullJustify(self, words, maxWidth):
            """ Straightforward solution for the problem

            Refer to:
            https://discuss.leetcode.com/topic/25970/concise-python-solution-10-lines

            Once you determine that there are only k words that can fit on a given line,
            you know what the total length of those words is cur_letters.
            Then the rest are spaces, and there are L = (maxWidth - cur_letters) of spaces.

            The trick here is to use mod operation to manage the spaces.
            The "or 1" part is for dealing with the edge case len(cur) == 1.
            """
            ans, cur_words, cur_letters = [], [], 0
            for w in words:
                if len(cur_words) + cur_letters + len(w) > maxWidth:
                    pad_space_cnt = maxWidth - cur_letters
                    for i in range(pad_space_cnt):
                        cur_words[i % (len(cur_words) - 1 or 1)] += ' '
                    ans.append(''.join(cur_words))

                    cur_words, cur_letters = [], 0

                cur_words.append(w)
                cur_letters += len(w)

            return ans + [' '.join(cur_words).ljust(maxWidth)]

    """
    ["a"]
    1
    [""]
    2
    ["This", "is", "an", "example", "of", "text", "justification."]
    15
    ["This", "is", "an", "example", "of", "text", "justification."]
    16
    ["This", "is", "an", "example", "of", "text", "justification."]
    20
    ["What","must","be","shall","be."]
    12
    """



151. Reverse words in a string
------------------------------------

.. code-block:: python

    Given an input string, reverse the string word by word.



    For example,
    Given s = "the sky is blue",
    return "blue is sky the".



    Update (2015-02-12):
    For C programmers: Try to solve it in-place in O(1) space.


    click to show clarification.

    Clarification:



    What constitutes a word?
    A sequence of non-space characters constitutes a word.
    Could the input string contain leading or trailing spaces?
    Yes. However, your reversed string should not contain leading or trailing spaces.
    How about multiple spaces between two words?
    Reduce them to a single space in the reversed string.



    =================================================================
    class Solution(object):
      def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return " ".join(s.split()[::-1])


    =================================================================
    class Solution(object):
        def reverseWords(self, s):
            return " ".join(s.split()[::-1])

    """
    if __name__ == "__main__":
        sol = Solution()
        print sol.reverseWords("AAA BBB   ")
        print sol.reverseWords(" BBB   CC  ")
    """




165. Compare version numbers
-------------------------------

.. code-block:: python

    Compare two version numbers version1 and version2.
    If version1 &gt; version2 return 1, if version1 &lt; version2 return -1, otherwise return 0.

    You may assume that the version strings are non-empty and contain only digits and the . character.
    The . character does not represent a decimal point and is used to separate number sequences.
    For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

    Here is an example of version numbers ordering:
    0.1 &lt; 1.1 &lt; 1.2 &lt; 13.37

    Credits:Special thanks to @ts for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1 = version1.split(".")
        v2 = version2.split(".")
        i = 0
        while i < len(v1) and i < len(v2):
          v1Seg, v2Seg = int(v1[i]), int(v2[i])
          if v1Seg > v2Seg:
            return 1
          elif v1Seg < v2Seg:
            return -1
          else:
            i += 1
        if i < len(v1) and int("".join(v1[i:])) != 0:
          return 1
        if i < len(v2) and int("".join(v2[i:])) != 0:
          return -1
        return 0


    =================================================================
    class Solution(object):
        def compareVersion(self, version1, version2):
            ver_list_1 = version1.split(".")
            ver_list_2 = version2.split(".")

            len_1 = len(ver_list_1)
            len_2 = len(ver_list_2)
            for i in range(len_1):
                ver_list_1[i] = int(ver_list_1[i])
            for i in range(len_2):
                ver_list_2[i] = int(ver_list_2[i])

            len_max = max(len_1, len_2)
            for i in range(len_1, len_max):
                ver_list_1.append(0)
            for i in range(len_2, len_max):
                ver_list_2.append(0)

            for i in range(len_max):
                if ver_list_1[i] < ver_list_2[i]:
                    return -1
                elif ver_list_1[i] > ver_list_2[i]:
                    return 1
                else:
                    pass
            return 0

    """
    "01"
    "1"
    "1.0"
    "1"
    "1.2.3.4"
    "1.2.3.4.5"
    "1.12.13"
    "1.13"
    """



344. Reverse String
--------------------

.. code-block:: python

    Write a function that takes a string as input and returns the string reversed.


    Example:
    Given s = "hello", return "olleh".

    =================================================================
    class Solution(object):
      def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]

    =================================================================
    class Solution(object):
        def reverseString(self, s):
            return s[::-1]


    class Solution_2(object):
        def reverseString(self, s):
            left, right = 0, len(s) - 1
            s = list(s)
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
            return "".join(s)


    """
    ""
    "hello"
    "  HELLO "
    """
