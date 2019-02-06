Linked List - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

21. Merge Two Sorted Lists
--------------------------------

.. code-block:: python

    """

    Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def mergeTwoLists(self, l1, l2):
            """
            :type l1: ListNode
            :type l2: ListNode
            :rtype: ListNode
            """
            head = cur = ListNode(0)
            while l1 and l2:
                if l1.val > l2.val:
                    cur.next = l2
                    l2 = l2.next
                else:
                    cur.next = l1
                    l1 = l1.next
                cur = cur.next
            cur.next = l1 or l2
            return head.next

83. Remove Duplicates from Sorted List
------------------------------------------

.. code-block:: python

    """

    Given a sorted linked list, delete all duplicates such that each element appear only once.

    For example,
    Given 1->1->2, return 1->2.
    Given 1->1->2->3->3, return 1->2->3.

    """

    class Solution(object):
        def deleteDuplicates(self, head):
            """
            :type head: ListNode
            :rtype: ListNode
            """
            if not head:
                return head
            p = head
            q = head.next
            while q:
                if q.val == p.val:
                    p.next = q.next
                    q = q.next
                else:
                    p = p.next
                    q = q.next
            return head



    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    class Solution(object):
        def deleteDuplicates(self, head):
            """
            :type head: ListNode
            :rtype: ListNode
            """
            cur = head
            while cur:
                while cur.next and cur.val == cur.next.val:
                    cur.next = cur.next.next
                cur = cur.next
            return head




141. Linked List Cycle
--------------------------------

.. code-block:: python

    """

    Given a linked list, determine if it has a cycle in it.

    Follow up:
    Can you solve it without using extra space?

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def hasCycle(self, head):
            """
            :type head: ListNode
            :rtype: bool
            """
            if not head:
                return False
            walker = head
            runner = head.next
            try:
                while walker!=runner:
                    walker = walker.next
                    runner = runner.next.next
                return True
            except:
                return False

160. Intersection of Two Linked Lists
-------------------------------------------

.. code-block:: python

    """

    Write a program to find the node at which the intersection of two singly linked lists begins.


    For example, the following two linked lists:

    A:          a1 �넂 a2
                       �넊
                         c1 �넂 c2 �넂 c3
                       �넇
    B:     b1 �넂 b2 �넂 b3
    begin to intersect at node c1.


    Notes:

    If the two linked lists have no intersection at all, return null.
    The linked lists must retain their original structure after the function returns.
    You may assume there are no cycles anywhere in the entire linked structure.
    Your code should preferably run in O(n) time and use only O(1) memory.
    Credits:
    Special thanks to @stellari for adding this problem and creating all test cases.

    """
    class Solution(object):
        def getIntersectionNode(self, headA, headB):
            """
            :type head1, head1: ListNode
            :rtype: ListNode
            """
            if not headA or not headB:
                return None
            pa = headA
            pb = headB
            while pa is not pb:
                pa = headB if pa == None else pa.next
                pb = headA if pb == None else pb.next
            return pa






203. Remove Linked List Elements
--------------------------------

.. code-block:: python

    """

    Remove all elements from a linked list of integers that have value val.

    Example
    Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
    Return: 1 --> 2 --> 3 --> 4 --> 5

    """
    class Solution(object):
        def removeElements(self, head, val):
            """
            :type head: ListNode
            :type val: int
            :rtype: ListNode
            """
            dummy = ListNode(-1)
            dummy.next = head
            cur = dummy
            while cur:
                while cur.next and cur.next.val == val:
                    cur.next = cur.next.next
                cur=cur.next
            return dummy.next



206. Reverse Linked List
--------------------------------

.. code-block:: python

    """

    Reverse a singly linked list.

    """

    class Solution(object):
        def reverseList(self, head):
            """
            :type head: ListNode
            :rtype: ListNode
            """
            if not head:
                return None
            p = head
            q = head.next
            while q:
                head.next = q.next
                q.next = p
                p = q
                q = head.next
            return p

234. Palindrome Linked List
--------------------------------

.. code-block:: python

    """

    Given a singly linked list, determine if it is a palindrome.

    Follow up:
    Could you do it in O(n) time and O(1) space?

    """

    class Solution(object):
        def isPalindrome(self, head):
            """
            :type head: ListNode
            :rtype: bool
            """
            slow = fast = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next

            node = None
            while slow:
                nxt = slow.next
                slow.next = node
                node = slow
                slow = nxt

            while node and head:
                if node.val != head.val:
                    return False
                node = node.next
                head = head.next
            return True


237. Delete Node in a Linked List
--------------------------------------

.. code-block:: python


    """

    Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

    Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.

    """

    class Solution(object):
        def deleteNode(self, node):
            """
            :type node: ListNode
            :rtype: void Do not return anything, modify node in-place instead.
            """
            node.val = node.next.val
            node.next = node.next.next

