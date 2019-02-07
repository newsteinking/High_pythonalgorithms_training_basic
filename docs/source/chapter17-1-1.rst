Math - Easy 2
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

12 Integer to Roman
--------------------

.. code-block:: python

    Given an integer, convert it to a roman numeral.


    Input is guaranteed to be within the range from 1 to 3999.

    =================================================================
    class Solution(object):
      def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        ans = ""
        values = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
        literals = ["M", "D", "C", "L", "X", "V", "I"]
        for idx in [0, 2, 4]:
          k = num / values[literals[idx]]
          re = (num % values[literals[idx]]) / values[literals[idx + 2]]
          ans += k * literals[idx]
          if re >= 9:
            ans += literals[idx + 2] + literals[idx]
          elif re >= 5:
            ans += literals[idx + 1] + (re - 5) * literals[idx + 2]
          elif re == 4:
            ans += literals[idx + 2] + literals[idx + 1]
          else:
            ans += re * literals[idx + 2]
          num %= values[literals[idx + 2]]
        return ans


    =================================================================
    class Solution(object):
        def intToRoman(self, num):
            """
            :type num: int
            :rtype: str
            """
            integer_symbols = [["I", "IV", "V", "IX"],
                               ["X", "XL", "L", "XC"],
                               ["C", "CD", "D", "CM"],
                               ["M"]
                               ]
            roman_str = ""
            counter = 0

            while num != 0:
                single = num % 10

                if single in [1, 2, 3]:
                    roman_str = single * integer_symbols[counter][0] + roman_str
                elif single == 4:
                    roman_str = integer_symbols[counter][1] + roman_str
                elif single == 5:
                    roman_str = integer_symbols[counter][2] + roman_str
                elif single in [6, 7, 8]:
                    roman_str = integer_symbols[counter][2] +\
                        (single - 5) * integer_symbols[counter][0] +\
                        roman_str
                elif single == 9:
                    roman_str = integer_symbols[counter][3] + roman_str
                else:
                    num = num / 10
                    counter += 1
                    continue
                num = num / 10
                counter += 1

            return roman_str

    """
    1
    100
    3999
    """



13. Roman to Integer
------------------------

.. code-block:: python

    Given a roman numeral, convert it to an integer.

    Input is guaranteed to be within the range from 1 to 3999.

    =================================================================
    class Solution(object):
      def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        ans = 0
        for i in range(0, len(s) - 1):
          c = s[i]
          cafter = s[i + 1]
          if d[c] < d[cafter]:
            ans -= d[c]
          else:
            ans += d[c]
        ans += d[s[-1]]
        return ans



    =================================================================
    class Solution(object):
        def romanToInt(self, s):
            """
            :type s: str
            :rtype: int
            """
            symbols_integer = {"I": 1, "V": 5, "X": 10, "L": 50,
                               "C": 100, "D": 500, "M": 1000,
                               "IV": 4, "IX": 9, "XL": 40, "XC": 90,
                               "CD": 400, "CM": 900
                               }
            length = len(s)
            integer = 0
            isPass = False
            for i in range(length):
                # Subtractive notation use this symbol
                if isPass:
                    isPass = False
                    continue
                # Just add the integer
                if s[i] in symbols_integer and s[i:i + 2] not in symbols_integer:
                    integer = integer + symbols_integer[s[i]]
                    isPass = False
                    continue

                # Subtractive notation is used as follows.
                if s[i:i + 2] in symbols_integer:
                    integer = integer + symbols_integer[s[i:i + 2]]
                    isPass = True

            return integer

    """
    "DCXXI"
    "CDCM"
    """


43. Multiply Strings
-----------------------

.. code-block:: python

    Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2.

    Note:

    The length of both num1 and num2 is < 110.
    Both num1 and num2 contains only digits 0-9.
    Both num1 and num2 does not contain any leading zero.
    You must not use any built-in BigInteger library or convert the inputs to integer directly.


    =================================================================
    class Solution(object):
      def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        ans = [0] * (len(num1) + len(num2))
        for i, n1 in enumerate(reversed(num1)):
          for j, n2 in enumerate(reversed(num2)):
            ans[i + j] += int(n1) * int(n2)
            ans[i + j + 1] += ans[i + j] / 10
            ans[i + j] %= 10
        while len(ans) > 1 and ans[-1] == 0:
          ans.pop()
        return "".join(map(str, ans[::-1]))


    =================================================================
    class Solution(object):
        def multiply(self, num1, num2):
            """ Simulation the manual way we do multiplication.

            Start from right to left, perform multiplication on every pair of digits.
            And add them together.

            There is a good graph explanation.  Refer to:
            https://discuss.leetcode.com/topic/30508/easiest-java-solution-with-graph-explanation
            """
            m, n = len(num1), len(num2)
            pos = [0] * (m + n)
            for i in range(m - 1, -1, -1):
                for j in range(n - 1, -1, -1):
                    multi = int(num1[i]) * int(num2[j])
                    pos_sum = pos[i + j + 1] + multi

                    # Update pos[i+j], pos[i+j+1]
                    pos[i + j] += pos_sum / 10
                    pos[i + j + 1] = pos_sum % 10

            first_not_0 = 0
            while first_not_0 < m + n and pos[first_not_0] == 0:
                first_not_0 += 1

            return "".join(map(str, pos[first_not_0:] or [0]))

    """
    "0"
    "1"
    "123"
    "123"
    "12121212121212125"
    "121232323499999252"
    """


48. Rotate Image
--------------------

.. code-block:: python

    You are given an n x n 2D matrix representing an image.

    Rotate the image by 90 degrees (clockwise).

    Note:
    You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.


    Example 1:

    Given input matrix =
    [
      [1,2,3],
      [4,5,6],
      [7,8,9]
    ],

    rotate the input matrix in-place such that it becomes:
    [
      [7,4,1],
      [8,5,2],
      [9,6,3]
    ]



    Example 2:

    Given input matrix =
    [
      [ 5, 1, 9,11],
      [ 2, 4, 8,10],
      [13, 3, 6, 7],
      [15,14,12,16]
    ],

    rotate the input matrix in-place such that it becomes:
    [
      [15,13, 2, 5],
      [14, 3, 4, 1],
      [12, 6, 8, 9],
      [16, 7,10,11]
    ]


    =================================================================
    class Solution(object):
      def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        if len(matrix) == 0:
          return
        h = len(matrix)
        w = len(matrix[0])
        for i in range(0, h):
          for j in range(0, w / 2):
            matrix[i][j], matrix[i][w - j - 1] = matrix[i][w - j - 1], matrix[i][j]

        for i in range(0, h):
          for j in range(0, w - 1 - i):
            matrix[i][j], matrix[w - 1 - j][h - 1 - i] = matrix[w - 1 - j][h - 1 - i], matrix[i][j]



    =================================================================
    class Solution(object):
        def rotate(self, matrix):
            """Rotate the image by 90 degrees (clockwise).

            :type matrix: List[List[int]]
            :rtype: void Do not return anything, modify matrix in-place instead.

            After rotate, the element in A[i][j] moves to A[j][n-1-i].  So we can
            Firstly reverse up to down : A[i][j]     --> A[n-1-i][j]
            Then then swap the symmetry: A[n-1-i][j] --> A[j][n-1-i]

            1 2 3     7 8 9     7 4 1
            4 5 6  => 4 5 6  => 8 5 2
            7 8 9     1 2 3     9 6 3
            """
            length = len(matrix)
            for i in range(length / 2):
                matrix[i], matrix[length - 1 - i] = matrix[length - 1 - i], matrix[i]

            for i in range(length):
                for j in range(i):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


    class Solution_2(object):
        def rotate(self, matrix):
            """Pythonic way which is amazing.

            According to:
            https://leetcode.com/discuss/82450/1-line-in-python
            """
            matrix[::] = zip(*matrix[::-1])

    """
    [[1]]
    [[1,2], [3,4]]
    [[1,2,3], [4,5,6], [7,8,9]]
    """


50. powx-n
--------------------

.. code-block:: python

    Implement pow(x, n).

    =================================================================
    class Solution(object):
      def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
          n = -n
          x = 1 / x
        ans = 1
        while n:
          if n & 1:
            ans *= x
          x *= x
          n >>= 1
        return ans


    =================================================================
    class Solution(object):
        def myPow(self, x, n):
            abs_half = abs(n) / 2

            if n == 0:
                return 1.00

            elif n > 0:
                result = self.myPow(x * x, abs_half)
                if n & 1 == 1:
                    result *= x
                return result

            else:
                result = 1 / self.myPow(x * x, abs_half)
                if abs(n) & 1 == 1:
                    result *= 1 / x
                return result

    """
    8.88023
    3
    2
    1
    2.2
    -100
    """

60. Permutation Sequence
-------------------------------

.. code-block:: python

    The set [1,2,3,&#8230;,n] contains a total of n! unique permutations.

    By listing and labeling all of the permutations in order,
    We get the following sequence (ie, for n = 3):

    "123"
    "132"
    "213"
    "231"
    "312"
    "321"



    Given n and k, return the kth permutation sequence.

    Note: Given n will be between 1 and 9 inclusive.

    =================================================================
    class Solution(object):
      def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        visited = [0 for i in range(n)]
        fact = [math.factorial(n - i - 1) for i in range(n)]
        ans = ""
        k -= 1
        for i in range(n):
          t = k / fact[i]
          for j in range(n):
            if not visited[j]:
              if t == 0:
                break
              t -= 1
          ans += str(j + 1)
          k %= fact[i]
          visited[j] = 1
        return ans


    =================================================================
    class Solution(object):
        def getPermutation(self, n, k):
            """According to:

            https://leetcode.com/discuss/42700/explain-like-im-five-java-solution-in-o-n
            The logic is as follows:
            For n-th numbers, the permutations can be divided into (n-1)! groups,
            For the n-1 th numbers, can be divided to (n-2)! groups, and so on.
            Thus k/(n-1)! indicates the index of current number,
            and k%(n-1)! denotes remaining index for the remaining n-1 numbers.
            We keep doing this until n reaches 0, then we get n numbers permutations that is kth.
            """
            factorial = [1] * n
            for i in xrange(1, n):
                factorial[i] = i * factorial[i - 1]

            if k > factorial[n - 1] * n or k <= 0:
                return -1

            remain_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            result, pos, k = [], n, k - 1
            while pos:
                cur_num = k / factorial[pos - 1]
                k %= factorial[pos - 1]
                target_num = remain_list[cur_num]
                remain_list.remove(target_num)
                result.append(str(target_num))
                pos -= 1

            return "".join(result)

    """
    9
    23
    9
    24
    9
    25
    """


166. Fraction to Recurring Decimal
----------------------------------------

.. code-block:: python

    Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

    If the fractional part is repeating, enclose the repeating part in parentheses.

    For example,

    Given numerator = 1, denominator = 2, return "0.5".
    Given numerator = 2, denominator = 1, return "2".
    Given numerator = 2, denominator = 3, return "0.(6)".



    Credits:Special thanks to @Shangrila for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        ans = "-" if numerator * denominator < 0 else ""
        numerator = abs(numerator)
        denominator = abs(denominator)
        ans += str(numerator / denominator)
        if numerator % denominator:
          ans += "."
        numerator = (numerator % denominator) * 10
        if numerator == 0:
          return ans
        d = {}
        res = []
        while True:
          r = numerator % denominator
          v = numerator / denominator
          if numerator in d:
            idx = d[numerator]
            return ans + "".join(res[:idx]) + "(" + "".join(res[idx:]) + ")"
          res.append(str(v))
          if v == 0:
            d[numerator] = len(res) - 1
            numerator *= 10
            continue
          d[numerator] = len(res) - 1
          numerator = r * 10
          if r == 0:
            return ans + "".join(res)
        return ans + "".join(res)



    =================================================================
    class Solution(object):
        def fractionToDecimal(self, numerator, denominator):
            # Calcluate the abs's decimal and then add the symbol
            negative = 0
            if numerator * denominator < 0:
                negative = 1
            numerator, denominator = abs(numerator), abs(denominator)

            answer = []
            answer.append(str(numerator/denominator))
            remainder = numerator % denominator
            if remainder:
                answer.append(".")
            # Keep the start position of the repeating part
            remainder_start = {}
            while remainder:
                remainder *= 10
                if remainder in remainder_start:
                    answer.insert(remainder_start[remainder], "(")
                    answer.append(")")
                    break
                else:
                    remainder_start[remainder] = len(answer)
                    answer.append(str(remainder/denominator))
                    remainder = remainder % denominator
            if negative:
                answer.insert(0, "-")
                return "".join(answer)
            else:
                return "".join(answer)

    """
    1
    9
    -1
    999
    2
    2
    -50
    -8
    -50
    8
    """


179. Largest Number
--------------------

.. code-block:: python

    Given a list of non negative integers, arrange them such that they form the largest number.

    For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.

    Note: The result may be very large, so you need to return a string instead of an integer.

    Credits:Special thanks to @ts for adding this problem and creating all test cases.

    =================================================================
    class Solution:
      # @param {integer[]} nums
      # @return {string}
      def largestNumber(self, nums):
        def cmpFunc(a, b):
          stra, strb = str(a), str(b)
          if stra + strb < strb + stra:
            return -1
          elif stra + strb > strb + stra:
            return 1
          else:
            return 0

        nums.sort(cmp=cmpFunc, reverse=True)
        return "".join(str(num) for num in nums) if sum(nums) != 0 else "0"



    =================================================================
    def comp(a, b):
        return int(a + b > b + a) * 2 - 1


    class Solution(object):
        def largestNumber(self, nums):
            nums = map(str, nums)
            nums.sort(cmp=comp, reverse=True)
            return str(int("".join(nums)))

    """
    [1]
    [1,2,3,21]
    [1,2,3,23]
    """


223. Rectangle Area
--------------------

.. code-block:: python

    Find the total area covered by two rectilinear rectangles in a 2D plane.
    Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.




    Assume that the total area is never beyond the maximum possible value of int.


    Credits:Special thanks to @mithmatt for adding this problem, creating the above image and all test cases.

    =================================================================
    class Solution(object):
      def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        area = (C - A) * (D - B) + (G - E) * (H - F)
        overlap = max(min(C, G) - max(A, E), 0) * max(min(D, H) - max(B, F), 0)
        return area - overlap


    =================================================================
    class Solution(object):
        def computeArea(self, A, B, C, D, E, F, G, H):
            size_1 = (C-A) * (D-B)
            size_2 = (G-E) * (H-F)
            left = max(A, E)
            bottom = max(B, F)
            right = min(C, G)
            top = min(D, H)

            # There is an area coverd by both the two rectangle
            if left < right and bottom < top:
                return size_1 + size_2 - (top-bottom) * (right-left)
            else:
                return size_1 + size_2

    """
    -2
    -2
    2
    2
    -2
    -2
    2
    2
    0
    0
    0
    0
    -1
    -1
    1
    1
    """

233. Number of Digit One
-------------------------------

.. code-block:: python

    Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.


    For example:
    Given n = 13,
    Return 6, because digit 1 occurred in the following numbers: 1, 10, 11, 12, 13.

    =================================================================
    class Solution(object):
      def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        m = 1
        ones = 0
        while m <= n:
          r = (n / m) % 10
          if r > 1:
            ones += m
          elif r == 1:
            ones += n % m + 1

          ones += (n / (m * 10)) * m
          m *= 10
        return ones



    =================================================================
    class Solution(object):
        # Recursive solution
        def countDigitOne(self, n):
            if n <= 0:
                return 0
            elif n < 10:
                return 1
            else:
                units = n % 10
                tens = n / 10
                count = self.countDigitOne(tens - 1) * 10 + tens
                n /= 10
                while n:
                    if n % 10 == 1:
                        count = count + 1 + units
                    n = n / 10

                if units >= 1:
                    count += 1
                return count

    """
    -1
    6
    12
    234545
    """


238. Product of array except self
--------------------------------------

.. code-block:: python

    Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

    Solve it without division and in O(n).

    For example, given [1,2,3,4], return [24,12,8,6].

    Follow up:
    Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose of space complexity analysis.)

    =================================================================
    class Solution(object):
      # better way
      def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
          dp[i] = dp[i - 1] * nums[i - 1]
        prod = 1
        for i in reversed(range(0, len(nums))):
          dp[i] = dp[i] * prod
          prod *= nums[i]
        return dp


    =================================================================
    class Solution(object):
        def productExceptSelf(self, nums):
            nums_len = len(nums)
            products = [1] * nums_len
            # Product of left part before the current position
            for i in range(1, nums_len):
                products[i] = products[i-1] * nums[i-1]

            # Mul the product of right part after the current position
            right_procudt = 1
            for j in range(nums_len-1, -1, -1):
                products[j] *= right_procudt
                right_procudt *= nums[j]

            return products

    """
    [0,0]
    [1,2,3,4,5]
    [1,2,3,4,0]
    """


273. Integer to English
--------------------------

.. code-block:: python

    Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 231 - 1.


    For example,

    123 -> "One Hundred Twenty Three"
    12345 -> "Twelve Thousand Three Hundred Forty Five"
    1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"

    =================================================================
    units = {1: "", 100: " Hundred", 1000: " Thousand", 1000000: " Million", 1000000000: " Billion"}
    tenToTwenty = {10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen",
                   17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty"}
    tens = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty", 7: "Seventy", 8: "Eighty", 9: "Ninety"}
    digit = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


    class Solution(object):
      def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        global units, tenToTwenty, tens, digit
        ans = []

        def getNum(number):
          global units, tenToTwenty, tens, digit
          divider = 1000
          ans = []
          h = number / 100
          if h != 0:
            ans.append(digit[h] + " Hundred")
          number = number % 100
          if number in tenToTwenty:
            ans.append(tenToTwenty[number])
          else:
            t = number / 10
            if t != 0:
              ans.append(tens[t])
            number = number % 10
            d = number
            if d != 0:
              ans.append(digit[d])
          return " ".join(ans)

        divider = 1000000000
        while num > 0:
          res = num / divider
          if res != 0:
            ans.append(getNum(res) + units[divider])
          num = num % divider
          divider /= 1000
        if not ans:
          return "Zero"
        return " ".join(ans)



    =================================================================
    class Solution(object):
        def numberToWords(self, num):
            self.words_conv = {
                0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
                5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
                10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
                14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
                18: "Eighteen", 19: "Nineteen", 20: "Twenty", 30: "Thirty",
                40: "Forty", 50: "Fifty", 60: "Sixty", 70: "Seventy", 80: "Eighty",
                90: "Ninety", 100: "Hundred", 1000: "Thousand", 1000000: "Million",
                1000000000: "Billion"
            }

            if num <= 20:
                return self.words_conv[num]
            elif num < 999:
                return self.convert_three(num)
            else:
                bill, bill_str = num / 1000000000, ""
                mill, mill_str = num % 1000000000 / 1000000, ""
                thou, thou_str = num % 1000000 / 1000, ""
                hund, hund_str = num % 1000, ""
                if bill:
                    bill_str = self.convert_three(bill) + " Billion "
                if mill:
                    mill_str = self.convert_three(mill) + " Million "
                if thou:
                    thou_str = self.convert_three(thou) + " Thousand "
                if hund:
                    hund_str = self.convert_three(hund)
                str = bill_str + mill_str + thou_str + hund_str
                # Erase the tailing space, when num = 1000...
                while str[-1] == " ":
                    str = str[:-1]
                return str

        def convert_three(self, num):
            # assert(num < 1000)
            if num < 100:
                return self.convert_two(num)
            else:
                str = self.words_conv[num/100] + " " + self.words_conv[100]
                other = self.convert_two(num % 100)
                if other:
                    str = str + " " + other
                return str

        def convert_two(self, num):
            # assert(num < 100)
            if not num:
                return ""
            if num <= 20:
                return self.words_conv[num]
            else:
                if num % 10 != 0:
                    return (
                        self.words_conv[num/10*10] + " " +
                        self.words_conv[num % 10])
                else:
                    return self.words_conv[num/10*10]

    """
    0
    9
    10
    14
    20
    22
    99
    100
    101
    999
    1000
    1001
    1999
    9999
    1000010
    1010010
    1110010
    1110001
    2001000000
    2000001000
    2111111001
    2147483647
    """


292. Nim Game
--------------------

.. code-block:: python

    You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove the stones.



    Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.



    For example, if there are 4 stones in the heap, then you will never win the game: no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return not n % 4 == 0



    =================================================================
    class Solution(object):
        # Just get the conclusion from the following second way.
        def canWinNim(self, n):
            if n % 4:
                return True
            else:
                return False


    class Solution_2(object):
        # Easy to understand, need more memory.
        # Can be optimized by using static variable.
        def canWinNim(self, n):
            dp = [True] * (n+1)
            if n > 3:
                dp[4] = False
                for i in range(5, n+1):
                    if dp[i-1] and dp[i-2] and dp[i-3]:
                        dp[i] = False
            return dp[n]

    """
    1
    8
    12
    245
    12345
    """


319. bulb switcher
--------------------

.. code-block:: python

    There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb.

    Find how many bulbs are on after n rounds.



    Example:

    Given n = 3.
    At first, the three bulbs are [off, off, off].
    After first round, the three bulbs are [on, on, on].
    After second round, the three bulbs are [on, off, on].
    After third round, the three bulbs are [on, off, off].
    So you should return 1, because there is only one bulb is on.

    =================================================================
    class Solution(object):
      def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        return int(n ** 0.5)



    =================================================================
    class Solution(object):
        def bulbSwitch(self, n):
            """
            A bulb ends up on iff it is switched an odd number of times.
            Call them bulb 1 to bulb n.
            Bulb i is switched in round d if and only if d divides i.
            So bulb i ends up on if and only if it has an odd number of divisors.
            """
            return int(n ** 0.5)
            """
            count = 0
            for i in xrange(1, n+1):
                if i * i < (n+1):
                    count += 1
                else:
                    break
            return count
            """
    """
    0
    1
    2
    3
    4
    12
    1908
    """

326. Power of three
--------------------

.. code-block:: python

    Given an integer, write a function to determine if it is a power of three.


    Follow up:
    Could you do it without using any loop / recursion?


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n > 0:
          return (1162261467 % n) == 0
        else:
          return False



    =================================================================
    class Solution(object):
        def isPowerOfThree(self, n):
            # 3 ** 19 = 1162261467
            return n > 0 and not (1162261467 % n)

    """
    -1
    0
    1
    27
    72
    """


335. Self Crossing
--------------------

.. code-block:: python

    You are given an array x of n positive numbers. You start at point (0,0) and moves x[0] metres to the north, then x[1] metres to the west,
    x[2] metres to the south,
    x[3] metres to the east and so on. In other words, after each move your direction changes
    counter-clockwise.


        Write a one-pass algorithm with O(1) extra space to determine, if your path crosses itself, or not.



    Example 1:

    Given x = [2, 1, 1, 2],
    ?????
    ?   ?
    ???????>
        ?

    Return true (self crossing)




    Example 2:

    Given x = [1, 2, 3, 4],
    ????????
    ?      ?
    ?
    ?
    ?????????????>

    Return false (not self crossing)




    Example 3:

    Given x = [1, 1, 1, 1],
    ?????
    ?   ?
    ?????>

    Return true (self crossing)



    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def isSelfCrossing(self, x):
        """
        :type x: List[int]
        :rtype: bool
        """
        if len(x) < 4:
          return False
        for i in range(3, len(x)):
          if x[i] >= x[i - 2] and x[i - 1] <= x[i - 3]:
            return True
          if i >= 4 and x[i - 1] == x[i - 3] and x[i] + x[i - 4] >= x[i - 2]:
            return True
          if i >= 5 and x[i - 1] <= x[i - 3] and x[i - 3] <= x[i - 1] + x[i - 5] and x[i] + x[i - 4] >= x[i - 2] and x[
            i - 4] <= x[i - 2]:
            return True
        return False



    =================================================================
    """
    According to: https://leetcode.com/discuss/88153/another-python

    Draw a line of length a. Then draw further lines of lengths b, c, etc.
    How does the a-line get crossed?
    From the left by the d-line or from the right by the f-line.

               b                              b
       +----------------+             +----------------+
       |                |             |                |
       |                |             |                | a
     c |                |           c |                |
       |                | a           |                |    f
       +------------------>           |             <---------+
                d       |             |                |      | e
                        |             |                       |
                                      +-----------------------+
                                                  d
    The "special case" of the e-line stabbing the a-line from below.

    """


    class Solution(object):
        def isSelfCrossing(self, x):
            if not x or len(x) < 4:
                return False
            b = c = d = e = f = 0  # Initinal
            for a in x:
                if d >= b > 0 and (a >= c > 0 or (a >= c-e >= 0 and f >= d-b)):
                    return True
                b, c, d, e, f = a, b, c, d, e
            return False

    """
    []
    [2,2]
    [1,1,1,1]
    [6,4,3,2,2,1,5]
    [1,1,2,2,3,3,4,4]
    """



343. Integer Break
--------------------

.. code-block:: python


    Given a positive integer n, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get.



    For example, given n = 2, return 1 (2 = 1 + 1); given n = 10, return 36 (10 = 3 + 3 + 4).



    Note: You may assume that n is not less than 2 and not larger than 58.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 3:
          return n - 1
        if n % 3 == 0:
          return 3 ** (n / 3)
        if n % 3 == 1:
          return 3 ** ((n / 3) - 1) * 4
        if n % 3 == 2:
          return 3 ** (n / 3) * 2


    =================================================================
    class Solution(object):
        """ Magic factor 2 and 3.

        Break the numbers into magic factor only 2 and 3 if number >= 4,
        Then we will get the max product.

        If we break a number N into two factors x, N-x, product is p=x(N-x).
        To get the maximum of p,  x=N/2 when N is even, x=(N-1)/2 when N is odd.
        If x can be break again and the product is bigger than x, then break recursively.

        Now the question is, for a given number N, when to stop break.  It's clearly that:
        (N/2)*(N/2) < N (N is even),     then N < 4,  N = 2
        (N-1)/2 *(N+1)/2 < N (N id odd), then N < 5,  N = 3, N = 1

        Thus, the factors of the perfect product should only be 2 or 3.

        According to:
        https://discuss.leetcode.com/topic/45341/an-simple-explanation-of-the-math-part-and-a-o-n-solution
        """
        def integerBreak(self, n):
            assert(n >= 2)
            if n == 2 or n == 3:
                return n - 1
            three_cnt = n / 3
            two_cnt = (n - three_cnt * 3) / 2

            # We should minus one 3 and add two 2,  number may be 10, 13
            if n - three_cnt * 3 == 1:
                two_cnt = 2
                three_cnt -= 1
            return 3 ** three_cnt * (2 ** two_cnt)

    """
    2
    7
    10
    102
    """
