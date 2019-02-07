DFA - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

65. Valid Number
--------------------

.. code-block:: python

    Validate if a given string is numeric.


    Some examples:
    "0" => true
    "   0.1  " => true
    "abc" => false
    "1 a" => false
    "2e10" => true


    Note: It is intended for the problem statement to be ambiguous. You should gather all requirements up front before implementing one.



    Update (2015-02-10):
    The signature of the C++ function had been updated. If you still see your function signature accepts a const char * argument, please click the reload button  to reset your code definition.

    =================================================================
    class States(object):
      def __init__(self):
        self.init = 0
        self.decimal = 1
        self.decpoint = 2
        self.afterdp = 3
        self.e = 4
        self.aftere = 5
        self.sign = 6
        self.nullpoint = 7
        self.esign = 8
        self.afteresign = 9


    class Solution(object):
      def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.strip()
        states = States()
        state = states.init
        decimals = "01234567890"

        for c in s:
          if state == states.init:
            if c == ".":
              state = states.nullpoint
            elif c in decimals:
              state = states.decimal
            elif c in ["+", "-"]:
              state = states.sign
            else:
              return False
          elif state == states.sign:
            if c in decimals:
              state = states.decimal
            elif c == ".":
              state = states.nullpoint
            else:
              return False
          elif state == states.esign:
            if c not in decimals:
              return False
            state = states.afteresign
          elif state == states.afteresign:
            if c not in decimals:
              return False
          elif state == states.nullpoint:
            if c not in decimals:
              return False
            state = states.decpoint
          elif state == states.decimal:
            if c in decimals:
              continue
            elif c == "e":
              state = states.e
            elif c == ".":
              state = states.decpoint
            else:
              return False
          elif state == states.decpoint:
            if c in decimals:
              state = states.afterdp
            elif c == "e":
              state = states.e
            else:
              return False
          elif state == states.afterdp:
            if c in decimals:
              continue
            elif c == "e":
              state = states.e
            else:
              return False
          elif state == states.e:
            if c in decimals:
              state = states.aftere
            elif c in ["+", "-"]:
              state = states.esign
            else:
              return False
          elif state == states.aftere:
            if c not in decimals:
              return False
          else:
            return False
        return state not in [states.init, states.e, states.nullpoint, states.sign, states.esign]


    =================================================================
    class Solution(object):
        def isNumber(self, s):
            """DFA

            Details can be found here:
            https://github.com/xuelangZF/LeetCode/blob/master/Images/65_ValidNumber.png
            https://github.com/xuelangZF/LeetCode/blob/master/Images/65_StateConvert.png
            """
            s = s.strip()
            if not s:
                return False

            # DFA states change table
            DFA_states_change = {
                0: {1: 2, 2: 1, 3: 8, 4: -1},
                1: {1: 2, 2: -1, 3: 8, 4: -1},
                2: {1: 2, 2: -1, 3: 3, 4: 5},
                3: {1: 4, 2: -1, 3: -1, 4: 5},
                4: {1: 4, 2: -1, 3: -1, 4: 5},
                5: {1: 7, 2: 6, 3: -1, 4: -1},
                6: {1: 7, 2: -1, 3: -1, 4: -1},
                7: {1: 7, 2: -1, 3: -1, 4: -1},
                8: {1: 4, 2: -1, 3: -1, 4: -1}
            }
            current_state = 0
            for char in s:
                input_num = self.input_num(char)
                if not input_num:
                    return False
                next_state = DFA_states_change[current_state][input_num]
                if next_state == -1:
                    return False
                current_state = next_state

            if (current_state == 2 or current_state == 3 or
               current_state == 4 or current_state == 7):
                return True
            else:
                return False

        def input_num(self, char):
            if char in "0123456789":
                return 1
            elif char in "+-":
                return 2
            elif char == ".":
                return 3
            elif char == "e":
                return 4
            else:
                return 0

    # True
    """
    " .1"
    "012"
    "+12"
    "-12"
    "12e1"
    "12e-1"
    "12e+1"
    "12e0"
    "0e1"
    "-1e1"
    "1.2"
    ".2"
    ".1e1"
    "+.2"
    "1."
    "      .1 "
    "46.e3"
    """






