Linked List - Easy 2
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

2. Add Two Number
--------------------

.. code-block:: python

    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

    You may assume the two numbers do not contain any leading zero, except the number 0 itself.


    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output: 7 -> 0 -> 8

    =================================================================
    class Solution(object):
      # maybe standard version
      def _addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        p = dummy = ListNode(-1)
        carry = 0
        while l1 and l2:
          p.next = ListNode(l1.val + l2.val + carry)
          carry = p.next.val / 10
          p.next.val %= 10
          p = p.next
          l1 = l1.next
          l2 = l2.next

        res = l1 or l2
        while res:
          p.next = ListNode(res.val + carry)
          carry = p.next.val / 10
          p.next.val %= 10
          p = p.next
          res = res.next
        if carry:
          p.next = ListNode(1)
        return dummy.next

      # shorter version
      def addTwoNumbers(self, l1, l2):
        p = dummy = ListNode(-1)
        carry = 0
        while l1 or l2 or carry:
          val = (l1 and l1.val or 0) + (l2 and l2.val or 0) + carry
          carry = val / 10
          p.next = ListNode(val % 10)
          l1 = l1 and l1.next
          l2 = l2 and l2.next
          p = p.next
        return dummy.next



    =================================================================
    # Definition for singly-linked list.
    class ListNode(object):
        def __init__(self, x):
            self.val = x
            self.next = None


    class Solution(object):
        def addTwoNumbers(self, l1, l2):
            """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
            carry_in = 0
            head = ListNode(0)
            l_sum = head

            while l1 and l2:
                l_sum.next = ListNode((l1.val + l2.val + carry_in) % 10)
                carry_in = (l1.val + l2.val + carry_in) / 10
                l1 = l1.next
                l2 = l2.next
                l_sum = l_sum.next

            if l1:
                while l1:
                    l_sum.next = ListNode((l1.val + carry_in) % 10)
                    carry_in = (l1.val + carry_in) / 10
                    l1 = l1.next
                    l_sum = l_sum.next

            if l2:
                while l2:
                    l_sum.next = ListNode((l2.val + carry_in) % 10)
                    carry_in = (l2.val + carry_in) / 10
                    l2 = l2.next
                    l_sum = l_sum.next

            if carry_in != 0:
                l_sum.next = ListNode(carry_in)

            return head.next



61. Rotate LIst
--------------------

.. code-block:: python


    Given a list, rotate the list to the right by k places, where k is non-negative.

    For example:
    Given 1->2->3->4->5->NULL and k = 2,
    return 4->5->1->2->3->NULL.

    =================================================================
    class Solution(object):
      def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
          return head
        l = 1
        p = head
        while p.next:
          l += 1
          p = p.next
        k = k % l
        if k == 0:
          return head
        k = l - k % l - 1
        pp = head
        print
        k
        while k > 0:
          pp = pp.next
          k -= 1
        newHead = pp.next
        pp.next = None
        p.next = head
        return newHead



    =================================================================
    class Solution(object):
        def rotateRight(self, head, k):
            """No benefit by using slow/fast pointers to find the tail node.

            So just find the total length, and then do the rotate.
            """
            # Get the length of ListNode
            if not head or not head.next:
                return head

            len_scan, length = head, 0
            while len_scan:
                length += 1
                len_scan = len_scan.next

            # Get the new head after rotated
            k = k % length
            if not k:
                return head
            scan_count = 0
            new_tail = head
            while scan_count < length - k - 1:
                new_tail = new_tail.next
                scan_count += 1

            new_head = new_tail.next
            # Set the rotated right part point to none.
            new_tail.next = None

            # Get the last node in the original list
            original_tail = new_head
            while original_tail and original_tail.next:
                original_tail = original_tail.next

            # Merge the two list
            original_tail.next = head

            return new_head

    """
    []
    0
    [1,2,3,4,5]
    0
    [1,2,3,4,5]
    3
    [1,2,3,4,5]
    10
    []
    2
    """



76. Minimum window substring
------------------------------------

.. code-block:: python


    Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).



    For example,
    S = "ADOBECODEBANC"
    T = "ABC"


    Minimum window is "BANC".



    Note:
    If there is no such window in S that covers all characters in T, return the empty string "".


    If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.


    =================================================================
    class Solution(object):
      def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        score = 0
        wanted = collections.Counter(t)
        start, end = len(s), 3 * len(s)
        d = {}
        deq = collections.deque([])
        for i, c in enumerate(s):
          if c in wanted:
            deq.append(i)
            d[c] = d.get(c, 0) + 1
            if d[c] <= wanted[c]:
              score += 1
            while deq and d[s[deq[0]]] > wanted[s[deq[0]]]:
              d[s[deq.popleft()]] -= 1
            if score == len(t) and deq[-1] - deq[0] < end - start:
              start, end = deq[0], deq[-1]
        return s[start:end + 1]



    =================================================================
    class Solution(object):
        def minWindow(self, s, t):
            s_size = len(s)
            if not t or not s:
                return ""

            # Keep the present tims of all characters in T.
            t_dict = {}
            for char in t:
                if char not in t_dict:
                    t_dict[char] = 1
                else:
                    t_dict[char] += 1

            count = 0
            t_size = len(t)
            start = end = 0
            # min_window(start, end): the suitable window's left and right board
            min_window = (0, s_size)
            # Keep the present tims of the window's characters that appear in T.
            win_dict = {}
            while end < s_size:
                cur_char = s[end]
                if cur_char in t_dict:
                    if cur_char not in win_dict:
                        win_dict[cur_char] = 1
                    else:
                        win_dict[cur_char] += 1
                    if win_dict[cur_char] <= t_dict[cur_char]:
                        count += 1

                if count == t_size:
                    # Move start toward to cut the window's size.
                    is_suitable_window = True
                    while start <= end and is_suitable_window:
                        start_char = s[start]
                        if start_char not in t_dict:
                            start += 1
                        else:
                            if win_dict[start_char] > t_dict[start_char]:
                                win_dict[start_char] -= 1
                                start += 1
                            else:
                                is_suitable_window = False

                    # Update the minimum window
                    window_size = end - start + 1
                    min_size = min_window[1] - min_window[0] + 1
                    if window_size < min_size:
                        min_window = (start, end)

                    # Move start toward to find another suitable window
                    win_dict[s[start]] -= 1
                    start += 1
                    count -= 1

                end += 1
            # No suitable window
            if min_window[1] == s_size:
                return ""
            return s[min_window[0]: min_window[1] + 1]

    """
    "a"
    ""
    "ADOBECODEBANC"
    "ABCC"
    """


82. Remove duplicates from sorted list 2
----------------------------------------------

.. code-block:: python

    Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.


    For example,
    Given 1->2->3->3->4->4->5, return 1->2->5.
    Given 1->1->1->2->3, return 2->3.


    =================================================================
    class Solution(object):
      def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        dummy.next = head
        p = dummy
        while p.next:
          if p.next.next and p.next.val == p.next.next.val:
            z = p.next
            while z and z.next and z.val == z.next.val:
              z = z.next
            p.next = z.next
          else:
            p = p.next
        return dummy.next



    =================================================================
    # Recursively
    class Solution(object):
        def deleteDuplicates(self, head):
            if not head or not head.next:
                return head
            if head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                return self.deleteDuplicates(head.next)
            else:
                head.next = self.deleteDuplicates(head.next)
                return head


    # Iteraively
    class Solution_2(object):
        def deleteDuplicates(self, head):
            cur = pre_head = ListNode(0)
            while head:
                if head.next and head.val == head.next.val:
                    # Skip the duplicated nodes.
                    while head.next and head.val == head.next.val:
                        head = head.next
                    head = head.next
                # we can make sure head is one single node here.
                else:
                    cur.next = head
                    cur = cur.next
                    head = head.next
            cur.next = None     # Make sure the cur here is the tail: [1,2,2]
            return pre_head.next

    """
    []
    [1]
    [1,2,2]
    [3,3,3,3,3]
    [1,1,1,2,3,4,4,4,4,5]
    """



86. Partition List
--------------------

.. code-block:: python

    Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.


    You should preserve the original relative order of the nodes in each of the two partitions.


    For example,
    Given 1->4->3->2->5->2 and x = 3,
    return 1->2->2->4->3->5.


    =================================================================
    class Solution(object):
      def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        if head is None:
          return None
        dummy = ListNode(-1)
        dummy.next = head
        sHead = sDummy = ListNode(-1)
        p = dummy
        while p and p.next:
          if p.next.val < x:
            sDummy.next = p.next
            p.next = p.next.next
            sDummy = sDummy.next
          else:
            p = p.next
          # if you change p.next then make sure you wouldn't change p in next run
        sDummy.next = dummy.next
        return sHead.next



    =================================================================
    class Solution(object):
        def partition(self, head, x):
            """
            :type head: ListNode
            :type x: int
            :rtype: ListNode
            """
            keep_node = min_last = ListNode(x-1)
            tail_node = min_last

            while head:
                # Insert the node less than x after the last-min node.
                if head.val < x:
                    first_greater_node = min_last.next
                    next_node = ListNode(head.val)
                    min_last.next = next_node
                    min_last = next_node
                    min_last.next = first_greater_node

                    # There are no nodes greater than or equal to x.
                    if tail_node.val < x:
                        tail_node = min_last

                # Move the tail forward when meet a node >= x.
                else:
                    next_node = ListNode(head.val)
                    tail_node.next = next_node
                    tail_node = tail_node.next

                head = head.next

            return keep_node.next

    """
    []
    1
    [2, 4, 3, 2, 5, 2]
    3
    [3, 7, 8, -5, 2, 6]
    -2
    """



92. Reverse Linked List 2
-------------------------------

.. code-block:: python

    Reverse a linked list from position m to n. Do it in-place and in one-pass.



    For example:
    Given 1->2->3->4->5->NULL, m = 2 and n = 4,


    return 1->4->3->2->5->NULL.


    Note:
    Given m, n satisfy the following condition:
    1 &le; m &le; n &le; length of list.


    =================================================================
    class Solution(object):
      def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """

        def reverse(root, prep, k):
          cur = root
          pre = None
          next = None
          while cur and k > 0:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
            k -= 1
          root.next = next
          prep.next = pre
          return pre

        dummy = ListNode(-1)
        dummy.next = head
        k = 1
        p = dummy
        start = None
        while p:
          if k == m:
            start = p
          if k == n + 1:
            reverse(start.next, start, n - m + 1)
            return dummy.next
          k += 1
          p = p.next



    =================================================================
    class Solution(object):
        def reverseBetween(self, head, m, n):
            """
            :type head: ListNode
            :type m: int
            :type n: int
            :rtype: ListNode
            """
            reverse_count = 1
            reverse_start_node = ListNode(0)
            reverse_start_node.next = head
            keep_node = reverse_start_node
            while reverse_count < n:
                # Get the node(reverse_start_node) before the reversed position.
                if reverse_count < m:
                    reverse_start_node = head
                    head = head.next
                    reverse_count += 1
                # Insert the node after current head to the reversed position.
                else:
                    assert(head.next)
                    be_reversed_node = head.next

                    # Build the connection in the reversed list's tail.
                    tail_next_node = be_reversed_node.next
                    head.next = tail_next_node

                    # Build the connection in the reversed list's head.
                    head_next_node = reverse_start_node.next
                    reverse_start_node.next = be_reversed_node
                    be_reversed_node.next = head_next_node

                    reverse_count += 1

            return keep_node.next

    """
    [1]
    1
    1
    [1,2,3,4,5,6,7]
    1
    7
    """



138. Copy list with random pointer
--------------------------------------

.. code-block:: python

    A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.



    Return a deep copy of the list.


    =================================================================
    class Solution(object):
      def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        p = head
        while p:
          copy = RandomListNode(p.label)
          copy.next = p.next
          p.next = copy
          p = copy.next

        p = head
        while p:
          p.next.random = p.random and p.random.next
          p = p.next.next

        p = head
        copy = chead = head and head.next
        while p:
          p.next = p = copy.next
          copy.next = copy = p and p.next
        return chead



    =================================================================
    class Solution(object):
        # Hash table way, easy to understand. O(n) space costed.
        def copyRandomList(self, head):
            if not head:
                return None

            node_hash = {}
            cur_node = head
            while cur_node:
                cur_copy = RandomListNode(cur_node.label)
                node_hash[cur_node] = cur_copy
                cur_node = cur_node.next

            keep_head = node_hash[head]
            while head:
                head_copy = node_hash[head]
                head_copy.next = node_hash.get(head.next, None)
                head_copy.random = node_hash.get(head.random, None)
                head = head.next
            return keep_head

        # Solution 2, beats 85% of python submisssions.  Refer to:
        # https://leetcode.com/discuss/22421/solution-constant-space-complexity-linear-time-complexity
        def copyRandomList_1(self, head):
            if not head:
                return None

            # First round: make copy of each node,
            # and link them together side-by-side in a single list.
            cur_node = head
            while cur_node:
                next_node = cur_node.next
                copy_node = RandomListNode(cur_node.label)
                cur_node.next = copy_node
                copy_node.next = next_node
                cur_node = next_node

            # Second round: assign random pointers for the copy nodes.
            cur_node = head
            while cur_node:
                random_node = cur_node.random
                if random_node:
                    cur_node.next.random = random_node.next
                cur_node = cur_node.next.next

            # Third round: restore the original list, and extract the copy list.
            cur_node = head
            dummy_node = cur_copy_node = RandomListNode(0)
            while cur_node:
                next_node = cur_node.next.next
                # extract the copy list
                copy_node = cur_node.next
                cur_copy_node.next = copy_node
                cur_copy_node = copy_node
                # restore the original list
                cur_node.next = next_node
                cur_node = next_node

            return dummy_node.next



New
--------------------

.. code-block:: python


=================================================================



=================================================================



New
--------------------

.. code-block:: python


=================================================================



=================================================================



New
--------------------

.. code-block:: python


=================================================================



=================================================================



New
--------------------

.. code-block:: python


=================================================================



=================================================================



New
--------------------

.. code-block:: python


=================================================================



=================================================================



New
--------------------

.. code-block:: python


=================================================================



=================================================================
