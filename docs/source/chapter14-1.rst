Heap - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

23. Merge K Sorted Lists
------------------------------

.. code-block:: python

    Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

    =================================================================
    import heapq


    class Solution(object):
      def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        heap = []
        p = dummy = ListNode(-1)
        for i in range(0, len(lists)):
          node = lists[i]
          if not node:
            continue
          heapq.heappush(heap, (node.val, node))

        while heap:
          value, node = heapq.heappop(heap)
          p.next = node
          p = p.next
          if node.next:
            node = node.next
            heapq.heappush(heap, (node.val, node))
        return dummy.next


    =================================================================
    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    import heapq


    class Solution(object):
        def mergeKLists(self, lists):
            """
            :type lists: List[ListNode]
            :rtype: ListNode
            """
            if not lists:
                return []

            head = merged_list = ListNode(0)
            heap_record = []
            # push head of all the linked list to heap
            for node in lists:
                if node:
                    heap_record.append((node.val, node))
            heapq.heapify(heap_record)

            # get the min val and push the node into heap
            while heap_record:
                min_node = heapq.heappop(heap_record)
                merged_list.next = min_node[1]
                merged_list = merged_list.next
                if min_node[1].next:
                    next_node = min_node[1].next
                    heapq.heappush(heap_record, (next_node.val, next_node))

            return head.next

    """
    []
    [[1,4,5,6,9], [2,3,4,5,6,8], [0,1,2,3,4], [2,2,2,2]]
    """


295. Find Median From Data Stream
-------------------------------------

.. code-block:: python

    Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.
    Examples:
    [2,3,4] , the median is 3
    [2,3], the median is (2 + 3) / 2 = 2.5


    Design a data structure that supports the following two operations:


    void addNum(int num) - Add a integer number from the data stream to the data structure.
    double findMedian() - Return the median of all elements so far.



    For example:

    addNum(1)
    addNum(2)
    findMedian() -> 1.5
    addNum(3)
    findMedian() -> 2


    Credits:Special thanks to @Louis1992 for adding this problem and creating all test cases.

    =================================================================
    import heapq


    class MedianFinder:
      def __init__(self):
        """
        Initialize your data structure here.
        """
        self.left = []
        self.right = []
        self.median = None

      def addNum(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        left = self.left
        right = self.right
        if self.median is None:
          self.median = num
          return

        if num <= self.median:
          heapq.heappush(left, -num)
        else:
          heapq.heappush(right, num)

        if len(left) > len(right) + 1:
          top = -heapq.heappop(left)
          heapq.heappush(right, self.median)
          self.median = top
        if len(right) > len(left) + 1:
          top = heapq.heappop(right)
          heapq.heappush(left, -self.median)
          self.median = top

      def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        left, right = self.left, self.right
        if len(left) == len(right):
          return self.median
        elif len(left) > len(right):
          return (self.median - left[0]) / 2.0
        if len(right) > len(left):
          return (self.median + right[0]) / 2.0

    # Your MedianFinder object will be instantiated and called as such:
    # mf = MedianFinder()
    # mf.addNum(1)
    # mf.findMedian()


    =================================================================
    from heapq import *


    class MedianFinder:
        """According to
        https://leetcode.com/discuss/64850/short-simple-java-c-python-o-log-n-o-1

        keep two heaps (or priority queues):
            1. Max-heap small has the smaller half of the numbers.
            2. Min-heap large has the larger half of the numbers.

        This gives me direct access to the one
        or two middle values (they're the tops of the heaps)
        """
        def __init__(self):
            self.small, self.large = [], []
            self.count = 0

        def addNum(self, num):
            self.count += 1
            # Python has no max-heap, so we do some trick here by keep the -num in
            # smaller half, then the max num will be at the top of the heap.
            heappush(self.small, -heappushpop(self.large, num))
            if self.count & 1:
                heappush(self.large, -heappop(self.small))

        def findMedian(self):
            if self.count & 1:
                return float(self.large[0])
            else:
                return (self.large[0] - self.small[0]) / 2.0

    """
    if __name__ == '__main__':
        mf = MedianFinder()
        mf.addNum(6)
        print mf.findMedian()
        mf.addNum(10)
        print mf.findMedian()
        mf.addNum(2)
        mf.addNum(6)
        mf.addNum(5)
        print mf.findMedian()
    """




347. Top K Frequent Elements
-----------------------------------

.. code-block:: python

    Given a non-empty array of integers, return the k most frequent elements.

    For example,
    Given [1,1,1,2,2,3] and k = 2, return [1,2].


    Note:

    You may assume k is always valid, 1 &le; k &le; number of unique elements.
    Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

    =================================================================
    class Solution(object):
      def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        d = {}
        res = []
        ans = []
        buckets = [[] for _ in range(len(nums) + 1)]

        for num in nums:
          d[num] = d.get(num, 0) + 1

        for key in d:
          res.append((d[key], key))

        for t in res:
          freq, key = t
          buckets[freq].append(key)

        buckets.reverse()

        for item in buckets:
          if item and k > 0:
            while item and k > 0:
              ans.append(item.pop())
              k -= 1
            if k == 0:
              return ans

        return ans


    =================================================================
    class Solution(object):
        def topKFrequent(self, nums, k):
            """ Given a non-empty array of integers, return the k most frequent elements.

            heapq.nlargest(n, iterable[, key])
            Return a list with the n largest elements from the dataset defined by iterable.
            """
            num_count = collections.Counter(nums)
            return heapq.nlargest(k, num_count, key=lambda x: num_count[x])


    class Solution_2(object):
        def topKFrequent(self, nums, k):
            ''' Use Counter to extract the top k frequent elements

            most_common(k) return a list of tuples,
            where the first item of the tuple is the element,
            and the second item of the tuple is the count
            Thus, the built-in zip function could be used to extract
            the first item from the tuples
            '''
            return zip(*collections.Counter(nums).most_common(k))[0]

    """
    [1,1,1,2,2,3]
    2
    [1,1,2,3,3,3,4,4,4,4,1,1,1]
    3
    """
