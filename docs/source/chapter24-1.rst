TwoPointers - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

11. Container with most water
-----------------------------------

.. code-block:: python

    Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

    Note: You may not slant the container and n is at least 2.


    =================================================================
    class Solution(object):
      def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        ans = left = 0
        right = len(height) - 1
        while left < right:
          ans = max(ans, (right - left) * min(height[left], height[right]))
          if height[left] <= height[right]:
            left += 1
          else:
            right -= 1
        return ans


    =================================================================
    class Solution(object):
        def maxArea(self, height):
            """
            :type height: List[int]
            :rtype: int
            """
            length = len(height)
            left = 0
            right = length - 1
            max_area = 0

            # To find the biggest container, we recursively find a container
            # which is much bigger than what we have find before.
            while left < right:
                area = (right - left) * min(height[left], height[right])
                max_area = max(max_area, area)

                # To get a bigger container, we move point(lower height) to right
                if height[left] < height[right]:
                    min_height = height[left]
                    left += 1
                    while height[left] < min_height:
                        left += 1

                # To get a bigger container, we move point(lower height) to left
                else:
                    min_height = height[right]
                    right -= 1
                    while height[right] < min_height:
                        right -= 1

            return max_area

    """
    [1,1]
    [1,2,3,4,8,7,6,5]
    []
    """


15. 3 Sum
--------------------

.. code-block:: python

    Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

    Note: The solution set must not contain duplicate triplets.


    For example, given array S = [-1, 0, 1, 2, -1, -4],

    A solution set is:
    [
      [-1, 0, 1],
      [-1, -1, 2]
    ]

    =================================================================
    class Solution(object):
      def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        for i in range(0, len(nums)):
          if i > 0 and nums[i] == nums[i - 1]:
            continue
          target = 0 - nums[i]
          start, end = i + 1, len(nums) - 1
          while start < end:
            if nums[start] + nums[end] > target:
              end -= 1
            elif nums[start] + nums[end] < target:
              start += 1
            else:
              res.append((nums[i], nums[start], nums[end]))
              end -= 1
              start += 1
              while start < end and nums[end] == nums[end + 1]:
                end -= 1
              while start < end and nums[start] == nums[start - 1]:
                start += 1
        return res


    =================================================================
    class Solution(object):
        def threeSum(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
            solution = []
            nums.sort()
            length = len(nums)
            for i in range(length-2):
                # avoid duplicate triplets.
                if i == 0 or nums[i] > nums[i-1]:
                    cur_num = nums[i]

                    # Keep two points to scan double direction.
                    left = i + 1
                    right = length - 1
                    while left < right:
                        if nums[left] + nums[right] + cur_num < 0:
                            left += 1
                        elif nums[left] + nums[right] + cur_num > 0:
                            right -= 1
                        else:
                            triplet = [cur_num, nums[left], nums[right]]
                            solution.append(triplet)
                            left += 1
                            right -= 1
                            # avoid duplicate triplets.
                            while left < right and nums[left] == nums[left-1]:
                                left += 1
                            while left < right and nums[right] == nums[right+1]:
                                right -= 1

            return solution

    """
    []
    [-1,1,2,-1,-1,0,-2,1,1,3]
    """



16. 3sum closest
--------------------

.. code-block:: python

    Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.


        For example, given array S = {-1 2 1 -4}, and target = 1.

        The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

    =================================================================
    class Solution(object):
      def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        ans = 0
        diff = float("inf")
        for i in range(0, len(nums)):
          start, end = i + 1, len(nums) - 1
          while start < end:
            sum = nums[i] + nums[start] + nums[end]
            if sum > target:
              if abs(target - sum) < diff:
                diff = abs(target - sum)
                ans = sum
              end -= 1
            else:
              if abs(target - sum) < diff:
                diff = abs(target - sum)
                ans = sum
              start += 1
        return ans


    =================================================================
    class Solution(object):
        def threeSumClosest(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: int
            """
            nums.sort()
            min_distance = 2 ** 31 - 1
            length = len(nums)
            # keep the sum of three nums
            solution = 0
            for i in range(length-2):
                cur_num = nums[i]
                left = i + 1
                right = length - 1
                while left < right:
                    left_num = nums[left]
                    right_num = nums[right]
                    three_sum = cur_num + left_num + right_num

                    # the right point go back
                    if three_sum > target:
                        right -= 1
                        if min_distance > three_sum - target:
                            solution = three_sum
                            min_distance = three_sum - target
                    # the left point go forward
                    elif three_sum < target:
                        if min_distance > target - three_sum:
                            solution = three_sum
                            min_distance = target - three_sum
                        left += 1
                    else:
                        return three_sum

            return solution

    """
    [0,0,0]
    1
    [-1,-1,-1,-2,-3,1,2]
    5
    """



18. 4 sum
--------------------

.. code-block:: python

    Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

    Note: The solution set must not contain duplicate quadruplets.



    For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.

    A solution set is:
    [
      [-1,  0, 0, 1],
      [-2, -1, 1, 2],
      [-2,  0, 0, 2]
    ]

    =================================================================
    class Solution(object):
      def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        for i in range(0, len(nums)):
          if i > 0 and nums[i] == nums[i - 1]:
            continue
          for j in range(i + 1, len(nums)):
            if j > i + 1 and nums[j] == nums[j - 1]:
              continue
            start = j + 1
            end = len(nums) - 1
            while start < end:
              sum = nums[i] + nums[j] + nums[start] + nums[end]
              if sum < target:
                start += 1
              elif sum > target:
                end -= 1
              else:
                res.append((nums[i], nums[j], nums[start], nums[end]))
                start += 1
                end -= 1
                while start < end and nums[start] == nums[start - 1]:
                  start += 1
                while start < end and nums[end] == nums[end + 1]:
                  end -= 1
        return res


    =================================================================
    class Solution(object):
        def fourSum(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[List[int]]
            """
            nums.sort()
            solution = []
            length = len(nums)

            for i in range(length - 3):
                # avoid duplicate triplets.
                if i > 0 and nums[i] == nums[i-1]:
                    continue

                a = nums[i]
                for j in range(i + 1, length - 2):
                    # avoid duplicate triplets.
                    if j > i+1 and nums[j] == nums[j-1]:
                        continue

                    # Two points which are form head and bottom move toward
                    # to make the a + b + c + d == target
                    b = nums[j]
                    left = j + 1
                    right = length - 1
                    while left < right:
                        c = nums[left]
                        d = nums[right]
                        if a + b + c + d < target:
                            left += 1
                        elif a + b + c + d > target:
                            right -= 1
                        else:
                            solution.append([a, b, c, d])
                            # avoid duplicate triplets.
                            left += 1
                            while left < right and nums[left] == nums[left-1]:
                                left += 1
                            right -= 1
                            while right > left and nums[right] == nums[right+1]:
                                right -= 1

            return solution

    """
    []
    0
    [1, 0, -1, 0, -2, 2]
    0
    [-2,-2,-2,-2,-1,-1,-1,-1,1,1,1,1,2,2,2,2,0,0,0]
    0
    """



19. remove nth node from end of list
--------------------------------------

.. code-block:: python

    Given a linked list, remove the nth node from the end of list and return its head.


    For example,


       Given linked list: 1->2->3->4->5, and n = 2.

       After removing the second node from the end, the linked list becomes 1->2->3->5.



    Note:
    Given n will always be valid.
    Try to do this in one pass.


    =================================================================
    class Solution(object):
      def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        dummy.next = head
        fast = slow = dummy

        while n and fast:
          fast = fast.next
          n -= 1

        while fast.next and slow.next:
          fast = fast.next
          slow = slow.next

        slow.next = slow.next.next
        return dummy.next


    =================================================================
    class Solution(object):
        def removeNthFromEnd(self, head, n):
            steps = 0
            first = head
            # Let the first pointer goto n+1'th node.
            while first:
                first = first.next
                steps += 1
                if steps == n + 1:
                    break

            # the node to be removed is the head node.
            if steps < n + 1:
                return head.next

            # Let second move with first one by one. When first meet the NULL
            # Second will meet the (N+1)th Node from end of list.
            second = head
            while first:
                first = first.next
                second = second.next

            # Next node of the second will be removed.
            second.next = second.next.next
            return head

    """
    [1]
    1
    [1,2,3,4,5,6,7,8]
    5
    [1,2,3,4,5,6,7,8]
    8
    """



75. Sort Colors
--------------------

.. code-block:: python

    Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.



    Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.



    Note:
    You are not suppose to use the library's sort function for this problem.


    click to show follow up.


    Follow up:
    A rather straight forward solution is a two-pass algorithm using counting sort.
    First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
    Could you come up with an one-pass algorithm using only constant space?



    =================================================================
    class Solution(object):
      def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        x = y = z = -1
        for i in range(0, len(nums)):
          if nums[i] == 0:
            x += 1
            y += 1
            z += 1
            if z != -1:
              nums[z] = 2
            if y != -1:
              nums[y] = 1
            nums[x] = 0
          elif nums[i] == 1:
            y += 1
            z += 1
            nums[z] = 2
            if x != -1:
              nums[x] = 0
            if y != -1:
              nums[y] = 1
          elif nums[i] == 2:
            z += 1
            if y != -1:
              nums[y] = 1
            if x != -1:
              nums[x] = 0
            nums[z] = 2


    =================================================================
    class Solution(object):
        def sortColors(self, nums):
            len_n = len(nums)
            # pos_put_0: next position to put 0
            # pos_put_2: next position to put 2
            pos_put_0 = 0
            pos_put_2 = len_n - 1
            index = 0
            while index <= pos_put_2:
                if nums[index] == 0:
                    nums[index], nums[pos_put_0] = nums[pos_put_0], nums[index]
                    pos_put_0 += 1
                    index += 1

                elif nums[index] == 2:
                    nums[index], nums[pos_put_2] = nums[pos_put_2], nums[index]
                    pos_put_2 -= 1

                else:
                    index += 1

    """
    [0]
    [1,0]
    [0,1,2]
    [1,1,1,2,0,0,0,0,2,2,1,1,2]
    """



80. Remove duplicates from sorted array 2
--------------------------------------------

.. code-block:: python

    Follow up for "Remove Duplicates":
    What if duplicates are allowed at most twice?


    For example,
    Given sorted array nums = [1,1,1,2,2,3],


    Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length.


    =================================================================
    class Solution(object):
      def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 2:
          return len(nums)
        cnt = 0
        j = 1
        for i in range(1, len(nums)):
          if nums[i] == nums[i - 1]:
            cnt += 1
            if cnt < 2:
              nums[j] = nums[i]
              j += 1
          else:
            nums[j] = nums[i]
            j += 1
            cnt = 0
        return j

    =================================================================
    class Solution(object):
        def removeDuplicates(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            first_occ = 0
            nums_l = len(nums)
            count = 0
            while first_occ < nums_l:
                # The last single number occurence only once.
                if first_occ == nums_l - 1:
                    nums[count] = nums[first_occ]
                    count += 1
                    break

                # Always keep the first occurence of a number
                first_num = nums[first_occ]
                second_num = nums[first_occ+1]
                # Move the number occurence only once to it's position
                if first_num != second_num:
                    nums[count] = first_num
                    count += 1
                    first_occ += 1
                    continue

                # Move the number occur twice to their positions
                if first_num == second_num:
                    nums[count] = first_num
                    nums[count+1] = second_num
                    next_occ = first_occ+2
                    while next_occ < nums_l and nums[next_occ] == second_num:
                        next_occ += 1
                    count += 2
                    first_occ = next_occ
            return count


    """
    []
    [1,1,1,1,2,2,2,3,3,3,4,5,6,6,6,7]
    """



88. Merge sorted array
-------------------------------

.. code-block:: python

    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.


    Note:
    You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

    =================================================================
    class Solution(object):
      def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        end = m + n - 1
        m -= 1
        n -= 1
        while end >= 0 and m >= 0 and n >= 0:
          if nums1[m] > nums2[n]:
            nums1[end] = nums1[m]
            m -= 1
          else:
            nums1[end] = nums2[n]
            n -= 1
          end -= 1

        while n >= 0:
          nums1[end] = nums2[n]
          end -= 1
          n -= 1


    =================================================================
    class Solution(object):
        def merge(self, nums1, m, nums2, n):
            nums1_left = 0
            nums2_left = 0

            # Set 0 to the redundant space
            for i in range(m + n, len(nums1)):
                nums1[i] = 0

            while nums1_left < m + n and nums2_left < n:
                # All the number in nums1 is in the suitable position
                if not m or nums1_left == m + nums2_left:
                    nums1[nums1_left] = nums2[nums2_left]
                    nums1_left += 1
                    nums2_left += 1

                # nums1 don't need to change, just move toward
                elif nums2[nums2_left] > nums1[nums1_left]:
                    nums1_left += 1

                # add the number in nums2 into nums1
                else:
                    val_2 = nums2[nums2_left]
                    val_1 = nums1[nums1_left]
                    nums1[nums1_left] = val_2
                    nums1_left += 1
                    for i in range(m + nums2_left, nums1_left, -1):
                        nums1[i] = nums1[i - 1]
                    nums1[nums1_left] = val_1
                    nums2_left += 1
    """
    [1,2,3,0,0,0]
    3
    [2,5,6]
    3
    [3,5,7,8,9,10,0,0,0,0]
    5
    [2,4,6,7]
    4
    """



125. Valid palindrome
-------------------------

.. code-block:: python

    Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.



    For example,
    "A man, a plan, a canal: Panama" is a palindrome.
    "race a car" is not a palindrome.



    Note:
    Have you consider that the string might be empty? This is a good question to ask during an interview.

    For the purpose of this problem, we define empty string as valid palindrome.



    =================================================================
    class Solution(object):
      def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        start, end = 0, len(s) - 1
        while start < end:
          if not s[start].isalnum():
            start += 1
            continue
          if not s[end].isalnum():
            end -= 1
            continue
          if s[start].lower() != s[end].lower():
            return False
          start += 1
          end -= 1
        return True


    =================================================================
    class Solution(object):
        def isPalindrome(self, s):
            """
            :type s: str
            :rtype: bool
            """
            alpha_num_str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            s = s.upper()
            s_l = len(s)
            pre = 0
            post = s_l - 1
            while pre < post and pre < s_l and post >= 0:
                # Remember the situation ",,..".
                # Make sure pre and post don't
                while pre < s_l and s[pre] not in alpha_num_str:
                    pre += 1
                while post >= 0 and s[post] not in alpha_num_str:
                    post -= 1
                if pre >= post:
                    break
                if s[pre] != s[post]:
                    return False
                pre += 1
                post -= 1

            return True

    """
    ""
    "1a2"
    ",,,,...."
    "A man, a plan, a canal: Panama"
    "race a car"
    """



141. Linked List Cycle
---------------------------

.. code-block:: python

    Given a linked list, determine if it has a cycle in it.



    Follow up:
    Can you solve it without using extra space?


    =================================================================
    class Solution(object):
      def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        while fast and fast.next:
          fast = fast.next.next
          slow = slow.next
          if slow == fast:
            return True
        return False

    =================================================================
    class Solution(object):
        """
        Two pointers: one go 1 step, another one go 2 steps every time.
        Then if the list has a cycle, fast one will meet the slow one absolutely.
        Prove as follows:
        1. If has a circle
            Assume there are m nodes that not in cycle, and then k nodes in cycle.
            And slow one now go m+i nodes, fast one go 2m + 2i nodes whitout doubt.
            So, slow one in the i's node of the circle, and fast one m+2i
            That's say, fast one goes m+i steps more than slow one.
            As the nodes keep going,
            i grows so (m+i) mode k == 0, then fast and slow meet here.
        2. If not:
            fast one will meet None node.
        """
        def hasCycle(self, head):
            one_step = head
            two_steps = head
            while two_steps and two_steps.next:
                one_step = one_step.next
                two_steps = two_steps.next.next
                if one_step == two_steps:
                    return True
            return False



142. Linked list cycle 2
------------------------------

.. code-block:: python

    Given a linked list, return the node where the cycle begins. If there is no cycle, return null.



    Note: Do not modify the linked list.


    Follow up:
    Can you solve it without using extra space?


    =================================================================
    class Solution(object):
      def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = finder = head
        while fast and fast.next:
          slow = slow.next
          fast = fast.next.next
          if slow == fast:
            while finder != slow:
              finder = finder.next
              slow = slow.next
            return finder
        return None

    =================================================================
    class Solution(object):
        """
        Two pointers: one go 1 step, another one go 2 steps every time.
        Then if the list has a cycle, fast one will meet the slow one absolutely.
        Prove as follows:
        1. If has a circle
            Assume there are m nodes that not in cycle, and then k nodes in cycle.
            And slow one now go m+i nodes, fast one go 2m + 2i nodes whitout doubt.
            So, slow one in the i's node of the circle, and fast one m+2i
            That's say, fast one goes m+i steps more than slow one.
            As the nodes keep going,
            i grows so (m + i) mode k == 0, then fast and slow meet here.
        2. If not:
            fast one will meet None node.

        And once fast and slow meet at node i, then let slow continue going.
        One node from head go at the same time.  We can prove these two nodes
        will meet at the begin node of cycle, prove as follows:

        Assume before the node from head and the slow node meet, they go x steps.
        Then (x-m) mode k = (x+i) mod k, the minest x will be m clearly.
        Just remember we have proved:  (m + i) mode k == 0 before.
        """
        def detectCycle(self, head):
            has_cycle = False
            one_step = head
            two_steps = head
            while two_steps and two_steps.next:
                one_step = one_step.next
                two_steps = two_steps.next.next
                if one_step == two_steps:
                    has_cycle = True
                    break
            if not has_cycle:
                return None
            two_steps = head
            while two_steps != one_step:
                one_step = one_step.next
                two_steps = two_steps.next
            return one_step



209. Minimum size subarray sum
-----------------------------------

.. code-block:: python

    Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum &ge; s. If there isn't one, return 0 instead.


    For example, given the array [2,3,1,2,4,3] and s = 7,
    the subarray [4,3] has the minimal length under the problem constraint.


    click to show more practice.

    More practice:

    If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n).


    Credits:Special thanks to @Freezen for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        sum = 0
        j = 0
        ans = float("inf")
        for i in range(0, len(nums)):
          while j < len(nums) and sum < target:
            sum += nums[j]
            j += 1
          if sum >= target:
            ans = min(ans, j - i)
          sum -= nums[i]
        return ans if ans != float("inf") else 0

    =================================================================
    class Solution(object):
        # Maintain a minimum window with  two indices.
        def minSubArrayLen(self, s, nums):
            if not nums:
                return 0
            start, end, sums, min_len = 0, 0, nums[0], 0
            len_nums = len(nums)
            while end < len_nums:
                if sums < s and end + 1 < len_nums:
                    end += 1
                    sums += nums[end]
                if sums >= s:
                    if min_len == 0:
                        min_len = end - start + 1
                    else:
                        min_len = min(min_len, end - start + 1)
                    sums -= nums[start]
                    if start < end:
                        start += 1
                if end == len_nums - 1 and sums < s:
                    break
            return min_len

    """
    100
    []
    20
    [1,3,12,8,3,4,21]
    0
    [1,1,2]
    4
    [1,4,4]
    """



283. Move Zeroes
--------------------

.. code-block:: python

    Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.



    For example, given nums  = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].



    Note:

    You must do this in-place without making a copy of the array.
    Minimize the total number of operations.



    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = j = 0
        for i in range(0, len(nums)):
          if nums[i] != 0:
            nums[j], nums[i] = nums[i], nums[j]
            j += 1

    =================================================================
    class Solution(object):
        def moveZeroes(self, nums):
            # Get sum of zeros
            count = 0
            for num in nums:
                if not num:
                    count += 1

            # Move the no-zero number to the right position.
            pos, i = 0, 0
            while pos < len(nums):
                if nums[pos]:
                    nums[i] = nums[pos]
                    i += 1
                pos += 1

            # Append the zeros
            if count:
                nums[-count:] = [0] * count
            return

    """
    []
    [1]
    [0]
    [0,0,0]
    [0,1,0,3,12]
    [7,6,5,4,0,4,0,5,6,0,7,0,0]
    """



287. Find the duplicate number
--------------------------------

.. code-block:: python

    Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.



    Note:

    You must not modify the array (assume the array is read only).
    You must use only constant, O(1) extra space.
    Your runtime complexity should be less than O(n2).
    There is only one duplicate number in the array, but it could be repeated more than once.



    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums) - 1
        start, end = 1, n
        while start + 1 < end:
          mid = start + (end - start) / 2
          count = 0
          for num in nums:
            if num < mid:
              count += 1
          if count >= mid:
            end = mid
          else:
            start = mid
        if nums.count(start) > nums.count(end):
          return start
        return end


    =================================================================
    class Solution(object):
        """
        Use two pointers the fast and the slow. The fast one goes forward two steps
        each time, while the slow one goes only step each time.
        In fact, they meet in a circle, the duplicate number
        must be the entry point of the circle when visiting the array from nums[0].
        """
        def findDuplicate(self, nums):
            # assert(len(nums) > 1)
            slow = nums[0]
            fast = nums[nums[0]]
            while slow != fast:
                slow = nums[slow]
                fast = nums[nums[fast]]

            target = 0
            while target != slow:
                target = nums[target]
                slow = nums[slow]
            return target

    """
    [1]
    [1,1]
    [1,3,4,5,1,2]
    [1,3,4,1,1,2]
    """



345. Reverse vowels of a string
----------------------------------

.. code-block:: python


    Write a function that takes a string as input and reverse only the vowels of a string.


    Example 1:
    Given s = "hello", return "holle".



    Example 2:
    Given s = "leetcode", return "leotcede".



    Note:
    The vowels does not include the letter "y".


    =================================================================
    import string


    class Solution(object):
      def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = set(["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"])
        s = list(s)
        start, end = 0, len(s) - 1
        while start < end:
          if s[start] not in vowels:
            start += 1
          elif s[end] not in vowels:
            end -= 1
          else:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
        return "".join(s)


    =================================================================
    class Solution(object):
        def reverseVowels(self, s):
            # Scan while incrementing start and decrementing end.
            all_vowels = set(['a', 'e', 'i', 'o', 'u',
                              'A', 'E', 'I', 'O', 'U'])
            s = list(s)
            left, right = 0, len(s) - 1
            while left < right:
                if s[left] in all_vowels and s[right] in all_vowels:
                    s[left], s[right] = s[right], s[left]
                    left += 1
                    right -= 1
                elif s[left] in all_vowels:
                    right -= 1
                elif s[right] in all_vowels:
                    left += 1
                else:
                    left += 1
                    right -= 1
            return "".join(s)

    """
    ""
    "hello"
    "leetcode"
    "Administrator"
    """
