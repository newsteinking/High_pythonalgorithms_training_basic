Hash table - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

1. Two Sum
--------------------

.. code-block:: python

    """

    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Example:
    Given nums = [2, 7, 11, 15], target = 9,

    Because nums[0] + nums[1] = 2 + 7 = 9,
    return [0, 1].

    """

    class Solution(object):
        def twoSum(self, nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            dic = dict()
            for index,value in enumerate(nums):
                sub = target - value
                if sub in dic:
                    return [dic[sub],index]
                else:
                    dic[value] = index

136. Single Number
--------------------

.. code-block:: python

    """

    Given an array of integers, every element appears twice except for one. Find that single one.

    Note:
    Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

    """

    class Solution(object):
        def singleNumber(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            res = 0
            for i in nums:
                res = res ^ i
            return res



202. Happy Number
--------------------

.. code-block:: python


    """

    Write an algorithm to determine if a number is "happy".

    A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

    Example: 19 is a happy number

    12 + 92 = 82
    82 + 22 = 68
    62 + 82 = 100
    12 + 02 + 02 = 1
    Credits:
    Special thanks to @mithmatt and @ts for adding this problem and creating all test cases.

    """

    class Solution(object):
        def isHappy(self, n):
            """
            :type n: int
            :rtype: bool
            """
            c = set()
            while not n in c:
                c.add(n)
                n = sum([int(x) ** 2 for x in str(n)])
            return n==1


    class Solution(object):
        def isHappy(self, n):
            """
            :type n: int
            :rtype: bool
            """
            slow = n
            quick = sum([int(x) ** 2 for x in str(n)])
            while quick != slow:
                quick = sum([int(x) ** 2 for x in str(quick)])
                quick = sum([int(x) ** 2 for x in str(quick)])
                slow = sum([int(x) ** 2 for x in str(slow)])
            return slow == 1





204. Count Primes
--------------------

.. code-block:: python

    """

    Description:

    Count the number of prime numbers less than a non-negative number, n.

    """

    class Solution(object):
        def countPrimes(self, n):
            """
            :type n: int
            :rtype: int
            """
            if n < 3:
                return 0
            res = [True] * n
            res[0] = res[1] = False
            for i in range(2,int(math.sqrt(n)) + 1):
                res[i*i:n:i] = [False] * len(res[i*i:n:i])
            return sum(res)




205. Isomorphic Strings
------------------------------

.. code-block:: python

    """

    Given two strings s and t, determine if they are isomorphic.

    Two strings are isomorphic if the characters in s can be replaced to get t.

    All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.

    For example,
    Given "egg", "add", return true.

    Given "foo", "bar", return false.

    Given "paper", "title", return true.

    """

    class Solution(object):
        def isIsomorphic(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: bool
            """
            if len(s) != len(t):
                return False
            dic = dict()
            for i in range(len(s)):
                if s[i] in dic and dic[s[i]] != t[i]:
                    return False
                else:
                    dic[s[i]] = t[i]
            return True

    class Solution(object):
        def isIsomorphic(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: bool
            """
            if len(s) != len(t):
                return False
            dic1 = dict()
            dic2 = dict()
            for i in range(len(s)):
                if (s[i] in dic1 and dic1[s[i]] != t[i]) or (t[i] in dic2 and dic2[t[i]] != s[i]):
                    return False
                else:
                    dic1[s[i]] = t[i]
                    dic2[t[i]] = s[i]
            return True


    class Solution(object):
        def isIsomorphic(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: bool
            """
            return len(set(zip(s,t))) == len(set(s)) == len(set(t))


217. Contains Duplicate
----------------------------

.. code-block:: python

    """

    Given an array of integers, find if the array contains any duplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

    Subscribe to see which companies asked this question.

    """

    class Solution(object):
        def containsDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            if not nums:
                return False
            dic = dict()
            for num in nums:
                if num in dic:
                    return True
                dic[num] = 1
            return False


    class Solution(object):
        def containsDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            return len(nums) != len(set(nums))


219. Cotains Duplicate 2
-----------------------------

.. code-block:: python

    """

    Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

    """

    class Solution(object):
        def containsNearbyDuplicate(self, nums, k):
            """
            :type nums: List[int]
            :type k: int
            :rtype: bool
            """
            dic = dict()
            for index,value in enumerate(nums):
                if value in dic and index - dic[value] <= k:
                    return True
                dic[value] = index
            return False


242. Valid Anagram
--------------------

.. code-block:: python


    """
    Given two strings s and t, write a function to determine if t is an anagram of s.

    For example,
    s = "anagram", t = "nagaram", return true.
    s = "rat", t = "car", return false.

    Note:
    You may assume the string contains only lowercase alphabets.

    """

    class Solution(object):
        def isAnagram(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: bool
            """
            dic1 = {}
            dic2 = {}
            for i in s:
                dic1[i] = dic1.get(i,0)+1
            for i in t:
                dic2[i] = dic2.get(i,0)+1
            return dic1 == dic2


290. Word Pattern
--------------------

.. code-block:: python


    """

    Total Accepted: 76577
    Total Submissions: 233596
    Difficulty: Easy
    Contributor: LeetCode
    Given a pattern and a string str, find if str follows the same pattern.

    Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.

    Examples:
    pattern = "abba", str = "dog cat cat dog" should return true.
    pattern = "abba", str = "dog cat cat fish" should return false.
    pattern = "aaaa", str = "dog cat cat dog" should return false.
    pattern = "abba", str = "dog dog dog dog" should return false.
    Notes:
    You may assume pattern contains only lowercase letters, and str contains lowercase letters separated by a single space.

    """

    class Solution(object):
        def wordPattern(self, pattern, str):
            """
            :type pattern: str
            :type str: str
            :rtype: bool
            """

            return len(pattern) == len(str.split(' ')) and len(set(zip(pattern, str.split(' ')))) == len(
                set(pattern)) == len(set(str.split(' ')))



349. Intersection of Two Arrays
-----------------------------------

.. code-block:: python

    """

    Given two arrays, write a function to compute their intersection.

    Example:
    Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

    """

    class Solution(object):
        def intersection(self, nums1, nums2):
            """
            :type nums1: List[int]
            :type nums2: List[int]
            :rtype: List[int]
            """

            nums1 = set(nums1)
            return [x for x in set(nums2) if x in nums1]


350. Intersection of Two Arrays 2
------------------------------------

.. code-block:: python

    """

    Given two arrays, write a function to compute their intersection.

    Example:
    Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].

    Note:
    Each element in the result should appear as many times as it shows in both arrays.
    The result can be in any order.
    Follow up:
    What if the given array is already sorted? How would you optimize your algorithm?
    What if nums1's size is small compared to nums2's size? Which algorithm is better?
    What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

    """

    class Solution(object):
        def intersect(self, nums1, nums2):
            """
            :type nums1: List[int]
            :type nums2: List[int]
            :rtype: List[int]
            """
            dic1,dic2 = dict(),dict()
            for num in nums1:
                dic1[num] = dic1.get(num,0) + 1
            for num in nums2:
                dic2[num] = dic2.get(num,0) + 1
            return [x for x in dic2 for j in range(min(dic1.get(x,0),dic2.get(x,0)))]


    import collections

    class Solution(object):
        def intersect(self, nums1, nums2):
            """
            :type nums1: List[int]
            :type nums2: List[int]
            :rtype: List[int]
            """
            c1,c2 = collections.Counter(nums1),collections.Counter(nums2)
            return [i for i in c1.keys() for j in range(min([c1[i], c2[i]]))]

389. Find the Difference
--------------------------------

.. code-block:: python


    """

    Given two strings s and t which consist of only lowercase letters.

    String t is generated by random shuffling string s and then add one more letter at a random position.

    Find the letter that was added in t.

    Example:

    Input:
    s = "abcd"
    t = "abcde"

    Output:
    e

    Explanation:
    'e' is the letter that was added.


    """
    class Solution(object):
        def findTheDifference(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: str
            """
            dic = dict()
            for single in s:
                dic[single] = dic.get(single, 0) + 1
            for single in t:
                if single in dic:
                    dic[single] = dic[single] - 1
                    if dic[single] == 0:
                        del dic[single]
                else:
                    return single





    class Solution(object):
        def findTheDifference(self, s, t):
            """
            :type s: str
            :type t: str
            :rtype: str
            """
            return chr(reduce(operator.xor, map(ord, (s + t))))




409. Longest Palindrome
---------------------------------

.. code-block:: python

    """

    Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.

    This is case sensitive, for example "Aa" is not considered a palindrome here.

    Note:
    Assume the length of given string will not exceed 1,010.

    Example:

    Input:
    "abccccdd"

    Output:
    7

    Explanation:
    One longest palindrome that can be built is "dccaccd", whose length is 7.

    """

    import collections
    class Solution(object):
        def longestPalindrome(self, s):
            """
            :type s: str
            :rtype: int
            """
            t = collections.Counter(s)
            return sum([t[x] for x in t if t[x] %2==0]) + sum([t[x]-1 for x in t if t[x] > 1 and t[x]%2==1])+max([1 for x in t if t[x]%2==1] or [0])



438. Find All Anagrams in a String
----------------------------------------

.. code-block:: python

    """

    Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

    Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

    The order of output does not matter.

    Example 1:

    Input:
    s: "cbaebabacd" p: "abc"

    Output:
    [0, 6]

    Explanation:
    The substring with start index = 0 is "cba", which is an anagram of "abc".
    The substring with start index = 6 is "bac", which is an anagram of "abc".
    Example 2:

    Input:
    s: "abab" p: "ab"

    Output:
    [0, 1, 2]

    Explanation:
    The substring with start index = 0 is "ab", which is an anagram of "ab".
    The substring with start index = 1 is "ba", which is an anagram of "ab".
    The substring with start index = 2 is "ab", which is an anagram of "ab".

    """

    class Solution(object):
        def findAnagrams(self, s, p):
            """
            :type s: str
            :type p: str
            :rtype: List[int]
            """
            res = []
            left = right = 0
            count = len(p)
            dic = dict()
            for i in p:
                dic[i] = dic.get(i,0)+1
            while right < len(s):
                if s[right] in dic.keys():
                    if dic[s[right]]>=1:
                        count = count - 1
                    dic[s[right]] = dic[s[right]]-1
                right = right+1
                if count == 0 :
                    res.append(left)
                if right - left == len(p):
                    if s[left] in dic.keys():
                        if dic[s[left]]>=0:
                            count = count + 1
                        dic[s[left]]+=1
                    left = left+1
            return res




447. Number of Boomerangs
----------------------------------

.. code-block:: python

    """

    Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points (i, j, k) such that the distance between i and j equals the distance between i and k (the order of the tuple matters).

    Find the number of boomerangs. You may assume that n will be at most 500 and coordinates of points are all in the range [-10000, 10000] (inclusive).

    Example:
    Input:
    [[0,0],[1,0],[2,0]]

    Output:
    2

    Explanation:
    The two boomerangs are [[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]]


    """

    class Solution(object):
        def numberOfBoomerangs(self, points):
            """
            :type points: List[List[int]]
            :rtype: int
            """
            res = 0
            for p in points:
                cmap = {}
                for q in points:
                    dis = (p[0]-q[0]) ** 2 + (p[1]-q[1])**2
                    cmap[dis] = cmap.get(dis,0)+1
                for key in cmap:
                    res += (cmap[key]) * (cmap[key]-1)
            return res


463. Island Perimeter
--------------------

.. code-block:: python

    """

    You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

    Example:

    [[0,1,0,0],
     [1,1,1,0],
     [0,1,0,0],
     [1,1,0,0]]

    Answer: 16
    Explanation: The perimeter is the 16 yellow stripes in the image below:


    """

    class Solution(object):
        def islandPerimeter(self, grid):
            """
            :type grid: List[List[int]]
            :rtype: int
            """
            num = 0
            neighbor = 0
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 1:
                        num = num + 1
                        if i > 0 and grid[i-1][j] == 1:
                            neighbor += 1
                        if j>0 and grid[i][j-1] == 1:
                            neighbor += 1
            return num * 4 - neighbor *2



575. Distribute Candies
---------------------------

.. code-block:: python

    """

    Given an integer array with even length, where different numbers in this array represent different kinds of candies. Each number means one candy of the corresponding kind. You need to distribute these candies equally in number to brother and sister. Return the maximum number of kinds of candies the sister could gain.

    Example 1:
    Input: candies = [1,1,2,2,3,3]
    Output: 3
    Explanation:
    There are three different kinds of candies (1, 2 and 3), and two candies for each kind.
    Optimal distribution: The sister has candies [1,2,3] and the brother has candies [1,2,3], too.
    The sister has three different kinds of candies.
    Example 2:
    Input: candies = [1,1,2,3]
    Output: 2
    Explanation: For example, the sister has candies [2,3] and the brother has candies [1,1].
    The sister has two different kinds of candies, the brother has only one kind of candies.

    """

    class Solution(object):
        def distributeCandies(self, candies):
            """
            :type candies: List[int]
            :rtype: int
            """
            return len(set(candies)) if len(set(candies)) < len(candies)/2 else len(candies)/2


594. Longest Harmonious Subsequence
----------------------------------------

.. code-block:: python

    """

    We define a harmonious array is an array where the difference between its maximum value and its minimum value is exactly 1.

    Now, given an integer array, you need to find the length of its longest harmonious subsequence among all its possible subsequences.

    Example 1:
    Input: [1,3,2,2,5,2,3,7]
    Output: 5
    Explanation: The longest harmonious subsequence is [3,2,2,2,3].
    Note: The length of the input array will not exceed 20,000.

    """

    import collections
    class Solution(object):
        def findLHS(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            dic = dict(collections.Counter(nums))
            max = 0
            for i in dic:
                if dic.get(i,0) > 0 and dic.get(i+1,0) > 0 and dic.get(i,0)+dic.get(i+1,0) > max:
                    max = dic.get(i,0) + dic.get(i+1,0)
            return max


599. Minimum Index Sum of Two Lists
----------------------------------------

.. code-block:: python

    """

    Suppose Andy and Doris want to choose a restaurant for dinner, and they both have a list of favorite restaurants represented by strings.

    You need to help them find out their common interest with the least list index sum. If there is a choice tie between answers, output all of them with no order requirement. You could assume there always exists an answer.

    Example 1:
    Input:
    ["Shogun", "Tapioca Express", "Burger King", "KFC"]
    ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
    Output: ["Shogun"]
    Explanation: The only restaurant they both like is "Shogun".
    Example 2:
    Input:
    ["Shogun", "Tapioca Express", "Burger King", "KFC"]
    ["KFC", "Shogun", "Burger King"]
    Output: ["Shogun"]
    Explanation: The restaurant they both like and have the least index sum is "Shogun" with index sum 1 (0+1).

    """

    class Solution(object):
        def findRestaurant(self, list1, list2):
            """
            :type list1: List[str]
            :type list2: List[str]
            :rtype: List[str]
            """
            dic1 = {v:i for i,v in enumerate(list1)}
            best,ans = 1e9,[]
            for i,v in enumerate(list2):
                if v in dic1:
                    if i+dic1[v] < best:
                        best = i+dic1[v]
                        ans = [v]
                    elif i+dic1[v] == best:
                        ans.append(v)
            return ans
