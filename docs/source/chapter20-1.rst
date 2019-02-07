Stack - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

20. Valid Parentheses
--------------------

.. code-block:: python

    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

    The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.


    =================================================================
    class Solution(object):
      def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        d = ["()", "[]", "{}"]
        for i in range(0, len(s)):
          stack.append(s[i])
          if len(stack) >= 2 and stack[-2] + stack[-1] in d:
            stack.pop()
            stack.pop()
        return len(stack) == 0


    =================================================================
    class Solution(object):
        def isValid(self, s):
            """
            :type s: str
            :rtype: bool
            """
            parenthes_stack = []
            match_rule = {")": "(", "]": "[", "}": "{"}
            for symble in s:
                if symble == "(" or symble == "[" or symble == "{":
                    parenthes_stack.append(symble)

                else:
                    # Check if stack top matches the current ), ] or }.
                    if (parenthes_stack and
                            parenthes_stack[-1] == match_rule[symble]):
                        parenthes_stack.pop()
                    else:
                        return False
            # Have some symbles that is not matched.
            if parenthes_stack:
                return False

            return True

    """
    ""
    "["
    "()()()()[]"
    "()((()){})"
    """


32. Longest Valid Parentheses
------------------------------------

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
            https://leetcode.com/discuss/7609/my-o-n-solution-using-a-stack

            If current character is '(', push its index to the stack.
            If current character is ')':
            1. top of stack is '(', just find a matching pair so pop from the stack.
            2. Otherwise, we push the index of ')' to the stack.

            Finally the stack will only contain the indices of characters which cannot be matched.
            Then the substring between adjacent indices should be valid parentheses.
            """
            stack, longest = [0], 0
            for i in xrange(1, len(s)):
                if s[i] == '(':
                    stack.append(i)
                else:
                    if stack and s[stack[-1]] == '(':
                        stack.pop()
                        valid_len = (i - stack[-1]) if stack else i + 1
                        longest = max(longest, valid_len)
                    else:
                        stack.append(i)
            return longest

    """
    ""
    ")"
    "()"
    "))"
    "(((()()()))("
    "(((()()()))())"
    """


71. Simplify Path
--------------------

.. code-block:: python

    Given an absolute path for a file (Unix-style), simplify it.

    For example,
    path = "/home/", => "/home"
    path = "/a/./b/../../c/", => "/c"


    click to show corner cases.

    Corner Cases:



    Did you consider the case where path = "/../"?
    In this case, you should return "/".
    Another corner case is the path might contain multiple slashes '/' together, such as "/home//foo/".
    In this case, you should ignore redundant slashes and return "/home/foo".

    =================================================================
    class Solution(object):
      def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        path = path.split("/")
        stack = []
        for p in path:
          if p in ["", "."]:
            continue
          if p == "..":
            if stack:
              stack.pop()
          else:
            stack.append(p)
        return "/" + "/".join(stack)


    =================================================================
    class Solution(object):
        def simplifyPath(self, path):
            """
            :type path: str
            :rtype: str
            """
            if not path:
                return "/"

            stack = []
            path_str = ""
            index = 0
            while index < len(path):
                char = path[index]
                if char == "/":
                    # './' respresent current directory
                    if path_str == "." or path_str == "":
                        path_str = ""

                    # '../' represent parent directory
                    elif path_str == "..":
                        if stack:
                            stack.pop()
                        path_str = ""

                    # 'path/': push path to stack
                    else:
                        stack.append(path_str)
                        path_str = ""
                else:
                    path_str += char
                index += 1

            # Append the last path
            if path_str == "..":
                if stack:
                    stack.pop()
            elif path_str == "." or path_str == "":
                pass
            else:
                stack.append(path_str)
            return "/" + "/".join(stack)

    """
    ""
    "/"
    "/.."
    "/home.as//"
    "/home.as"
    "/a/./b/../../c/"
    "/a/./b/../../c/../"
    "/a/./b/c/../.."
    "/..."
    """



84. Largest Rectangle in Histogram
--------------------------------------

.. code-block:: python

    Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.




    Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].




    The largest rectangle is shown in the shaded area, which has area = 10 unit.



    For example,
    Given heights = [2,1,5,6,2,3],
    return 10.



    =================================================================
    class Solution(object):
      def largestRectangleArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
          return 0
        height.append(-1)
        stack = []
        ans = 0
        for i in range(0, len(height)):
          while stack and height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1 if stack else i
            ans = max(ans, h * w)
          stack.append(i)
        height.pop()
        return ans


    =================================================================

    class Solution(object):
        def largestRectangleArea(self, height):
            """
            :type height: List[int]
            :rtype: int
            """
            # Add a bar of height 0 after the tail.
            height.append(0)
            size = len(height)
            no_decrease_stack = [0]
            max_size = height[0]

            i = 0
            while i < size:
                cur_num = height[i]
                # If the height of current bar is higher than the stack top,
                # or the stack is empty, push current index to stack
                if (not no_decrease_stack or
                        cur_num > height[no_decrease_stack[-1]]):
                    no_decrease_stack.append(i)
                    i += 1

                # The current height is lower or same than the top,
                # then pop until current height is higher than the top.
                else:
                    index = no_decrease_stack.pop()
                    if no_decrease_stack:
                        width = i - no_decrease_stack[-1] - 1
                    else:
                        width = i
                    max_size = max(max_size, width * height[index])

            return max_size

    """
    []
    [2,1,5,6,2,3]
    [2,1,5,6,2,2,2,3,3]
    [2,2,2,2]
    """



85. Maximal Rectangle
------------------------------

.. code-block:: python

    Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.


    For example, given the following matrix:

    1 0 1 0 0
    1 0 1 1 1
    1 1 1 1 1
    1 0 0 1 0

    Return 6.



    =================================================================
    class Solution(object):
      def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """

        def histogram(height):
          if not height:
            return 0
          height.append(-1)
          stack = []
          ans = 0
          for i in range(0, len(height)):
            while stack and height[i] < height[stack[-1]]:
              h = height[stack.pop()]
              w = i - stack[-1] - 1 if stack else i
              ans = max(ans, h * w)
            stack.append(i)
          return ans

        ans = 0
        dp = [[0] * len(matrix[0]) for _ in range(0, len(matrix))]
        for i in reversed(range(0, len(matrix))):
          if i == len(matrix) - 1:
            dp[i] = [int(h) for h in matrix[i]]
          else:
            for j in range(0, len(matrix[0])):
              if matrix[i][j] != "0":
                dp[i][j] = dp[i + 1][j] + 1
          ans = max(ans, histogram(dp[i]))
        return ans


    =================================================================
    class Solution(object):
        def maximalRectangle(self, matrix):
            """
            :type matrix: List[list[str]]
            :rtype: int
            """

            if not matrix:
                return 0

            m_rows = len(matrix)
            n_cols = len(matrix[0])

            # Pre-process: to make every row be a histogram
            process_matrix = [
                [0 for col in range(n_cols)] for row in range(m_rows)]
            for row in range(m_rows):
                for col in range(n_cols):
                    if row == 0:
                        if matrix[row][col] == "1":
                            process_matrix[row][col] = 1

                    else:
                        num = 1 if matrix[row][col] == "1" else 0
                        process_matrix[row][col] = num * (
                            num + process_matrix[row-1][col])

            # Find every max size of row.
            max_size = 0
            for row in range(m_rows):
                max_row_size = self.largestRectangleArea(process_matrix[row])
                max_size = max(max_row_size, max_size)
            return max_size

        # Find the largest rectangle in a histogram
        def largestRectangleArea(self, height):
            # Add a bar of height 0 after the tail.
            height.append(0)
            size = len(height)
            no_decrease_stack = [0]
            max_size = height[0]

            i = 0
            while i < size:
                cur_num = height[i]
                # If the height of current bar is higher than the stack top,
                # or the stack is empty, push current index to stack
                if (not no_decrease_stack or
                        cur_num > height[no_decrease_stack[-1]]):
                    no_decrease_stack.append(i)
                    i += 1

                # The current height is lower or same than the top,
                # then pop until current height is higher than the top.
                else:
                    index = no_decrease_stack.pop()
                    if no_decrease_stack:
                        width = i - no_decrease_stack[-1] - 1
                    else:
                        width = i
                    max_size = max(max_size, width * height[index])

            return max_size

    """
    []
    [["1","0","1","0"], ["1","1","1","1"]]
    """



94. Binary tree inorder traversal
------------------------------------

.. code-block:: python

    Given a binary tree, return the inorder traversal of its nodes' values.


    For example:
    Given binary tree [1,null,2,3],

       1
        \
         2
        /
       3



    return [1,3,2].


    Note: Recursive solution is trivial, could you do it iteratively?

    =================================================================
    class Solution(object):
      def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res, stack = [], [(1, root)]
        while stack:
          p = stack.pop()
          if not p[1]: continue
          stack.extend([(1, p[1].right), (0, p[1]), (1, p[1].left)]) if p[0] != 0 else res.append(p[1].val)
        return res


    =================================================================
    class Solution(object):
        def inorderTraversal(self, root):
            """
            :type root: TreeNode
            :rtype: List[int]
            """
            if not root:
                return []

            tree_stack = []
            inorder_tra = []
            while root or tree_stack:
                # Go along the left child
                if root:
                    tree_stack.append(root)
                    root = root.left
                # Meet a left, go back to the parent node
                else:
                    if not tree_stack:
                        root = None
                        continue
                    node = tree_stack.pop()
                    inorder_tra.append(node.val)
                    root = node.right

            return inorder_tra

    """
    []
    [1]
    [1,2,3,null,null,4,null,null,5]
    """



150. Evaluate Reverse Polish notation
-----------------------------------------------

.. code-block:: python

    Evaluate the value of an arithmetic expression in Reverse Polish Notation.



    Valid operators are +, -, *, /. Each operand may be an integer or another expression.



    Some examples:

      ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
      ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6


    =================================================================
    class Solution(object):
      def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
          if token in ["+", "-", "*", "/"]:
            b, a = stack.pop(), stack.pop()
            if token == "+":
              res = a + b
            if token == "-":
              res = a - b
            if token == "*":
              res = a * b
            if token == "/":
              res = int(float(a) / float(b))
            stack.append(res)
          else:
            stack.append(int(token))
        return stack.pop()

    =================================================================
    class Solution(object):
        def evalRPN(self, tokens):
            value_stack = []
            for token in tokens:
                if token in "+-*/":
                    operand_2 = value_stack.pop()
                    operand_1 = value_stack.pop()
                    negative = 1
                    if operand_1 * operand_2 < 0:
                        negative = -1

                    if token == "+":
                        result = operand_1 + operand_2
                    elif token == "-":
                        result = operand_1 - operand_2
                    elif token == "*":
                        result = operand_1 * operand_2
                    else:
                        # Leetcode think 12/-7 = -1, 12/-13 = 0
                        result = abs(operand_1) / abs(operand_2) * negative

                    value_stack.append(result)
                else:
                    value_stack.append(int(token))
            return value_stack[-1]

    """
    ["18"]
    ["12", "-7", "/"]
    ["2", "1", "+", "3", "*"]
    ["4", "13", "5", "/", "+"]
    """



155. Min Stack
--------------------

.. code-block:: python


    Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.


    push(x) -- Push element x onto stack.


    pop() -- Removes the element on top of the stack.


    top() -- Get the top element.


    getMin() -- Retrieve the minimum element in the stack.




    Example:

    MinStack minStack = new MinStack();
    minStack.push(-2);
    minStack.push(0);
    minStack.push(-3);
    minStack.getMin();   --> Returns -3.
    minStack.pop();
    minStack.top();      --> Returns 0.
    minStack.getMin();   --> Returns -2.



    =================================================================
    class MinStack(object):

      def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

      def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if not self.stack:
          self.stack.append((x, x))
        else:
          self.stack.append((x, min(x, self.stack[-1][-1])))

      def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()

      def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]

      def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]

    # Your MinStack object will be instantiated and called as such:
    # obj = MinStack()
    # obj.push(x)
    # obj.pop()
    # param_3 = obj.top()
    # param_4 = obj.getMin()


    =================================================================

    class MinStack(object):
        # According to:
        # https://leetcode.com/discuss/45373/c-using-two-stacks-quite-short-and-easy-to-understand
        def __init__(self):
            self.stack_d = []
            self.stack_m = []

        def push(self, x):
            self.stack_d.append(x)
            if not self.stack_m or x <= self.getMin():
                self.stack_m.append(x)

        def pop(self):
            if self.top() == self.getMin():
                self.stack_m.pop()
            self.stack_d.pop()

        def top(self):
            return self.stack_d[-1]

        def getMin(self):
            return self.stack_m[-1]

    '''
    if __name__ == '__main__':
        one_stack = MinStack()
        one_stack.push(3)
        one_stack.push(4)
        one_stack.push(2)
        one_stack.push(1)

        print one_stack.getMin()
        one_stack.pop()
        print one_stack.getMin()
        one_stack.pop()
        print one_stack.getMin()
        one_stack.push(0)
        print one_stack.getMin()
    '''



1224. Basic Caculator
------------------------------

.. code-block:: python

    Implement a basic calculator to evaluate a simple expression string.

    The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces  .

    You may assume that the given expression is always valid.

    Some examples:

    "1 + 1" = 2
    " 2-1 + 2 " = 3
    "(1+(4+5+2)-3)+(6+8)" = 23




    Note: Do not use the eval built-in library function.


    =================================================================
    class Solution(object):
      def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = "(" + s + ")"
        stack = []
        _stack = []
        i = 0
        while i < len(s):
          if s[i] == " ":
            i += 1
          elif s[i] == "(":
            _stack.append(len(stack))
            i += 1
          elif s[i] == ")":
            start = _stack.pop()
            j = start
            a = stack[j]
            while j + 2 < len(stack):
              ops = stack[j + 1]
              if ops == "+":
                a = a + stack[j + 2]
              elif ops == "-":
                a = a - stack[j + 2]
              else:
                return "invalid"
              j += 2
            k = len(stack) - start
            while k > 0:
              stack.pop()
              k -= 1
            stack.append(a)
            i += 1
          elif s[i] in "+-":
            stack.append(s[i])
            i += 1
          else:
            start = i
            while i < len(s) and s[i] not in "-+() ":
              i += 1
            num = int(s[start:i])
            stack.append(num)
        return stack[0]


    =================================================================
    class Solution(object):
        """
        1. Start from + sign and scan s from left to right;
        2. if c == digit: This number = Last digit * 10 + This digit;
        3. if c == '+': Add num to result before this sign;
            This sign = Last context sign * 1; clear num;
        4. if c == '-': Add num to result before this sign;
            This sign = Last context sign * -1; clear num;
        5. if c == '(': Push this context sign to stack;
        6. if c == ')': Pop this context and we come back to last context;
        7. Add the last num. This is because we only add number after '+' / '-'.
        """
        def calculate(self, s):
            # s = "".join(s.split(" "))     # Erase the redundant space
            sign = 1
            sign_stack = [1]
            num, result = 0, 0
            for ch in s:
                if ch == " ":
                    pass
                elif ch == "(":
                    sign_stack.append(sign)
                elif ch in "+-":
                    result += num * sign
                    sign = sign_stack[-1] * (1 if ch == "+" else -1)
                    num = 0
                elif ch == ")":
                    sign_stack.pop()
                else:
                    num = num * 10 + int(ch)
            result += num * sign
            return result

    """
    " 1234 "
    "1 + 1"
    " 2-1 + 2 "
    " 2  - (1+2-(1+2))"
    "(1+(4+5+2)-3)+(6+8)"
    """



225. Implement stack using queues
-------------------------------------

.. code-block:: python

    Implement the following operations of a stack using queues.


    push(x) -- Push element x onto stack.


    pop() -- Removes the element on top of the stack.


    top() -- Get the top element.


    empty() -- Return whether the stack is empty.


    Notes:

    You must use only standard operations of a queue -- which means only push to back, peek/pop from front, size, and is empty operations are valid.
    Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or deque (double-ended queue), as long as you use only standard operations of a queue.
    You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).



    Credits:Special thanks to @jianchao.li.fighter for adding this problem and all test cases.

    =================================================================
    from collections import deque


    class Stack(object):
      def __init__(self):
        """
        initialize your data structure here.
        """
        self.queue = deque([])

      def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.queue.append(x)
        for _ in range(0, len(self.queue) - 1):
          self.queue.append(self.queue.popleft())

      def pop(self):
        """
        :rtype: nothing
        """
        self.queue.popleft()

      def top(self):
        """
        :rtype: int
        """
        return self.queue[0]

      def empty(self):
        """
        :rtype: bool
        """
        return not self.queue


    =================================================================
    from collections import deque


    class Stack(object):
        def __init__(self):
            self._queue = deque()

        def push(self, x):
            # Pushing to back and
            # then rotating the queue until the new element is at the front
            q = self._queue
            q.append(x)
            for i in xrange(len(q) - 1):
                q.append(q.popleft())

        def pop(self):
            self._queue.popleft()

        def top(self):
            return self._queue[0]

        def empty(self):
            return not len(self._queue)

    """Test
    if __name__ == '__main__':
        s = Stack()
        s.push(1)
        s.push(2)
        print s.top()
        s.pop()
        print s.empty()
        print s.top()
        s.pop()
        print s.empty()
    """



227. Basic Calculator 2
------------------------------

.. code-block:: python

    Implement a basic calculator to evaluate a simple expression string.

    The expression string contains only non-negative integers, +, -, *, / operators and empty spaces  . The integer division should truncate toward zero.

    You may assume that the given expression is always valid.

    Some examples:

    "3+2*2" = 7
    " 3/2 " = 1
    " 3+5 / 2 " = 5




    Note: Do not use the eval built-in library function.


    Credits:Special thanks to @ts for adding this problem and creating all test cases.


    =================================================================
    from collections import deque


    class Solution(object):
      def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        i = 0
        queue = deque([])
        while i < len(s):
          if s[i] == " ":
            i += 1
          elif s[i] in "-+*/":
            queue.append(s[i])
            i += 1
          else:
            start = i
            while i < len(s) and s[i] not in "+-*/ ":
              i += 1
            num = int(s[start:i])
            queue.append(num)
            while len(queue) > 2 and queue[-2] in "*/":
              b = queue.pop()
              ops = queue.pop()
              a = queue.pop()
              if ops == "*":
                queue.append(a * b)
              elif ops == "/":
                queue.append(int(float(a) / b))
              else:
                return "invalid"
        if queue:
          a = queue.popleft()
          while len(queue) >= 2:
            ops = queue.popleft()
            b = queue.popleft()
            if ops == "+":
              a = a + b
            elif ops == "-":
              a = a - b
          return a
        return 0


    =================================================================
    class Solution(object):
        # Stack
        def calculate(self, s):
            num_stack = []
            num = 0
            sign = "+"
            for i in xrange(len(s)):
                ch = s[i]
                if ch.isdigit():
                    num = num * 10 + ord(ch) - ord('0')
                if not ch.isdigit() and ch != " " or i == len(s) - 1:
                    if sign == "+":
                        num_stack.append(num)
                    elif sign == "-":
                        num_stack.append(-num)
                    elif sign == "*":
                        num_stack.append(num * num_stack.pop())
                    else:
                        tmp = num_stack.pop()
                        divid = tmp / num
                        if tmp / num < 0 and tmp % num != 0:
                            divid += 1
                        num_stack.append(divid)
                    sign = ch
                    num = 0
            return sum(num_stack)

    """
    # No Stack
    # Refer to a very smart solution
    # https://leetcode.com/discuss/41641/17-lines-c-easy-20-ms
    class Solution(object):
        def calculate(self, s):
            s = "".join(s.split(" "))
            s = "+" + s + "+"
            len_s = len(s)
            result, term = 0, 0
            i = 0
            while i < len_s:
                ch = s[i]
                if ch in "+-":
                    result += term
                    term = 0
                    i += 1
                    while i < len_s and s[i] not in "+-*/":
                        term = term * 10 + int(s[i])
                        i += 1
                    term *= 1 if ch == "+" else -1
                else:
                    new_term = 0
                    i += 1
                    while i < len_s and s[i] not in "+-*/":
                        new_term = new_term * 10 + int(s[i])
                        i += 1
                    # For python: -3/2 = -2, which isn't suitable here.
                    # We need -3/2 = -1, because 14-3/2 = 13 not 12.
                    oper = 1
                    if term < 0:
                        oper *= -1
                        term *= -1
                    if ch == "/":
                        term /= new_term
                    else:
                        term *= new_term
                    term *= oper
            return result
    """

    """
    if __name__ == '__main__':
        sol = Solution()
        print sol.calculate("3112 ")
        print sol.calculate("3+ 2 * 2/1 *2")
        print sol.calculate(" 14- 3/  2")
        print sol.calculate("3 + 50 / 2")
    """



232. Implement queue using stacks
----------------------------------------

.. code-block:: python

    Implement the following operations of a queue using stacks.


    push(x) -- Push element x to the back of queue.


    pop() -- Removes the element from in front of queue.


    peek() -- Get the front element.


    empty() -- Return whether the queue is empty.


    Notes:

    You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
    Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
    You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).



    =================================================================
    from collections import deque


    class Queue(object):
      def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1 = deque([])
        self.stack2 = deque([])

      def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.stack1.append(x)

      def pop(self):
        """
        :rtype: nothing
        """
        self.peek()
        self.stack2.pop()

      def peek(self):
        """
        :rtype: int
        """
        if not self.stack2:
          while self.stack1:
            self.stack2.append(self.stack1.pop())
          return self.stack2[-1]
        else:
          return self.stack2[-1]

      def empty(self):
        """
        :rtype: bool
        """
        return not self.stack1 and not self.stack2


    =================================================================
    class Queue(object):
        """
        Use python list as the underlying data structure for stack.
        Add a "move()" method to simplify code: it moves all elements
        of the "inStack" to the "outStack" when the "outStack" is empty.
        """
        def __init__(self):
            self.in_stack, self.out_stack = [], []

        def push(self, x):
            self.in_stack.append(x)

        def pop(self):
            self.move()
            self.out_stack.pop()

        def peek(self):
            self.move()
            return self.out_stack[-1]

        def empty(self):
            return (not self.in_stack) and (not self.out_stack)

        def move(self):
            if not self.out_stack:
                while self.in_stack:
                    self.out_stack.append(self.in_stack.pop())

    '''
    if __name__ == '__main__':
        q = Queue()
        q.push(2)
        q.push(3)
        q.push(4)
        print q.peek()
        q.pop()
        print q.peek()
        q.pop()
        q.pop()
        print q.empty()
    '''


316. Remove duplicate letters
----------------------------------

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



341. Flatten nested list iterator
--------------------------------------

.. code-block:: python

    Given a nested list of integers, implement an iterator to flatten it.

    Each element is either an integer, or a list -- whose elements may also be integers or other lists.

    Example 1:
    Given the list [[1,1],2,[1,1]],

    By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].



    Example 2:
    Given the list [1,[4,[6]]],

    By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].

    =================================================================
    from collections import deque


    class NestedIterator(object):

      def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = deque(nestedList[::-1])
        self.value = None

      def next(self):
        """
        :rtype: int
        """
        self.hasNext()
        ret = self.value
        self.value = None
        return ret

      def hasNext(self):
        """
        :rtype: bool
        """
        if self.value is not None:
          return True

        stack = self.stack
        while stack:
          top = stack.pop()
          if top.isInteger():
            self.value = top.getInteger()
            return True
          else:
            stack.extend(top.getList()[::-1])
        return False

    # Your NestedIterator object will be instantiated and called as such:
    # i, v = NestedIterator(nestedList), []
    # while i.hasNext(): v.append(i.next())


    =================================================================
    class NestedIterator(object):
        """
        According to:
        https://discuss.leetcode.com/topic/42042/simple-java-solution-using-a-stack-with-explanation

        In the constructor, we push all the nestedList into the stack from back to front,
        so when we pop the stack, it returns the very first element.
        Second, in the hasNext() function, we peek the first element in stack currently,
        and if it is an Integer, we will return true and pop the element.

        If it is a list, we will further flatten it.
        This is iterative version of flatting the nested list.
        Again, we need to iterate from the back to front of the list.
        """
        def __init__(self, nestedList):
            """Initialize your data structure here.

            :type nestedList: List[NestedInteger]
            """
            self.stack = []
            for item in nestedList[::-1]:
                self.stack.append(item)

        def next(self):
            val = self.stack[-1].getInteger()
            self.stack.pop()
            return val

        def hasNext(self):
            while self.stack:
                curr = self.stack[-1]
                if curr.isInteger():
                    return True
                self.stack.pop()
                if curr.getList():
                    for i in curr.getList()[::-1]:
                        self.stack.append(i)
            return False


    class NestedIterator_2(object):
        """ Python Generators solution

        According to:
        https://discuss.leetcode.com/topic/45844/python-generators-solution
        """
        def __init__(self, nestedList):
            """Initialize your data structure here.
            """
            def gen(nestedList):
                for item in nestedList:
                    if item.isInteger():
                        yield item.getInteger()
                    else:
                        for list in gen(item.getList()):
                            yield list

            self.gen = gen(nestedList)

        def next(self):
            return self.value

        def hasNext(self):
            try:
                self.value = next(self.gen)
                return True
            except StopIteration:
                return False

    # Your NestedIterator object will be instantiated and called as such:
    # i, v = NestedIterator(nestedList), []
    # while i.hasNext(): v.append(i.next())

    '''
    []
    [1,[2,[3]]]
    [[1,2],3,[4,5]]
    [[[1,2,3], [4,5], 7], [8,9], 10]
    '''


