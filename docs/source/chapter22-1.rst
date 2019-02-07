ToBeOptimized - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

18. 4sum
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
            length = len(nums)

            # Get all the two sums and their two addend's index.
            two_sums_dict = {}
            for i in range(length):
                for j in range(i+1, length):
                    two_sums = nums[i] + nums[j]
                    if two_sums not in two_sums_dict:
                        two_sums_dict[two_sums] = []
                    two_sums_dict[two_sums].append([i, j])

            sums_list = two_sums_dict.keys
            sums_list.sort()
            solution = []







    """
    []
    0
    [1, 0, -1, 0, -2, 2]
    0
    [1,1,1,1,0,0,0,0,-1,-1,-1,-1]
    0
    """


125. Valid Palindrome
----------------------------

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




127. Word Ladder
--------------------

.. code-block:: python

    Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:


    Only one letter can be changed at a time.
    Each transformed word must exist in the word list. Note that beginWord is not a transformed word.



    For example,


    Given:
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]


    As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
    return its length 5.



    Note:

    Return 0 if there is no such transformation sequence.
    All words have the same length.
    All words contain only lowercase alphabetic characters.
    You may assume no duplicates in the word list.
    You may assume beginWord and endWord are non-empty and are not the same.




    UPDATE (2017/1/20):
    The wordList parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.


    =================================================================
    import string
    from collections import deque


    class Solution(object):
      def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: Set[str]
        :rtype: int
        """

        def getNbrs(src, dest, wordList):
          res = []
          for c in string.ascii_lowercase:
            for i in range(0, len(src)):
              newWord = src[:i] + c + src[i + 1:]
              if newWord == src:
                continue
              if newWord in wordList or newWord == dest:
                yield newWord

        queue = deque([beginWord])
        length = 0
        while queue:
          length += 1
          for k in range(0, len(queue)):
            top = queue.popleft()
            for nbr in getNbrs(top, endWord, wordList):
              wordList.remove(nbr)
              if nbr == endWord:
                return length + 1
              queue.append(nbr)
        return 0


    =================================================================
    class Solution(object):

        def ladderLength(self, beginWord, endWord, wordList):
            """
            Breadth First Search
            When build the adjacency tree, skip the visited word
            """
            if beginWord == endWord:
                return 1
            cur_level = [beginWord]
            next_level = []
            visited_word = {}
            visited_word[beginWord] = 1
            length = 0
            while cur_level:
                length += 1
                for cur_word in cur_level:
                    cur_len = len(cur_word)
                    one_distance = 0
                    for i in range(cur_len):
                        if cur_word[i] != endWord[i]:
                            if one_distance == 1:
                                one_distance = 0
                                break
                            one_distance += 1
                    if one_distance == 1:
                        return length + 1

                    # Get the next level
                    # When I put "abc...xyz" in the out loop, it just exceeded.
                    for i in range(cur_len):
                        pre_word = cur_word[:i]
                        post_word = cur_word[i+1:]
                        for j in "abcdefghijklmnopqrstuvwxyz":
                            next_word = pre_word + j + post_word
                            if (next_word not in visited_word and
                                    next_word in wordList):
                                next_level.append(next_word)
                                visited_word[next_word] = 1
                # Scan the next level then
                cur_level = next_level
                next_level = []
            return 0

        """ disapproved, when wordList growth bigger, it may be called too many times
        def is_one_distance(self, word_1, word_2):
            # alert(len(word_1) == len(word_2))
            word_l = len(word_1)
            one_distance = False
            for i in range(word_l):
                if word_1[i] != word_2[i]:
                    if not one_distance:
                        one_distance = True
                    else:
                        return False

            return one_distance
        """
    """ Test Case
    if __name__ == '__main__':
        sol = Solution()
        print sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"])
        print sol.ladderLength("hit", "cog", ["hot", "dot", "doh", "lot", "loh"])
        print sol.ladderLength(
            "hit", "cog",
            ["hot", "dot", "dog", "lot", "log", "hig", "hog"])
        print sol.ladderLength(
            "cet",
            "ism",
            ["kid", "tag", "pup", "ail", "tun", "woo", "erg", "luz", "brr", "gay",
             "sip", "kay", "per", "val", "mes", "ohs", "now", "boa", "cet", "pal",
             "bar", "die", "war", "hay", "eco", "pub", "lob", "rue", "fry", "lit",
             "rex", "jan", "cot", "bid", "ali", "pay", "col", "gum", "ger", "row",
             "won", "dan", "rum", "fad", "tut", "sag", "yip", "sui", "ark", "has",
             "zip", "fez", "own", "ump", "dis", "ads", "max", "jaw", "out", "btu",
             "ana", "gap", "cry", "led", "abe", "box", "ore", "pig", "fie", "toy",
             "fat", "cal", "lie", "noh", "sew", "ono", "tam", "flu", "mgm", "ply",
             "awe", "pry", "tit", "tie", "yet", "too", "tax", "jim", "san", "pan",
             "map", "ski", "ova", "wed", "non", "wac", "nut", "why", "bye", "lye",
             "oct", "old", "fin", "feb", "chi", "sap", "owl", "log", "tod", "dot",
             "bow", "fob", "for", "joe", "ivy", "fan", "age", "fax", "hip", "jib",
             "mel", "hus", "sob", "ifs", "tab", "ara", "dab", "jag", "jar", "arm",
             "lot", "tom", "sax", "tex", "yum", "pei", "wen", "wry", "ire", "irk",
             "far", "mew", "wit", "doe", "gas", "rte", "ian", "pot", "ask", "wag",
             "hag", "amy", "nag", "ron", "soy", "gin", "don", "tug", "fay", "vic",
             "boo", "nam", "ave", "buy", "sop", "but", "orb", "fen", "paw", "his",
             "sub", "bob", "yea", "oft", "inn", "rod", "yam", "pew", "web", "hod",
             "hun", "gyp", "wei", "wis", "rob", "gad", "pie", "mon", "dog", "bib",
             "rub", "ere", "dig", "era", "cat", "fox", "bee", "mod", "day", "apr",
             "vie", "nev", "jam", "pam", "new", "aye", "ani", "and", "ibm", "yap",
             "can", "pyx", "tar", "kin", "fog", "hum", "pip", "cup", "dye", "lyx",
             "jog", "nun", "par", "wan", "fey", "bus", "oak", "bad", "ats", "set",
             "qom", "vat", "eat", "pus", "rev", "axe", "ion", "six", "ila", "lao",
             "mom", "mas", "pro", "few", "opt", "poe", "art", "ash", "oar", "cap",
             "lop", "may", "shy", "rid", "bat", "sum", "rim", "fee", "bmw", "sky",
             "maj", "hue", "thy", "ava", "rap", "den", "fla", "auk", "cox", "ibo",
             "hey", "saw", "vim", "sec", "ltd", "you", "its", "tat", "dew", "eva",
             "tog", "ram", "let", "see", "zit", "maw", "nix", "ate", "gig", "rep",
             "owe", "ind", "hog", "eve", "sam", "zoo", "any", "dow", "cod", "bed",
             "vet", "ham", "sis", "hex", "via", "fir", "nod", "mao", "aug", "mum",
             "hoe", "bah", "hal", "keg", "hew", "zed", "tow", "gog", "ass", "dem",
             "who", "bet", "gos", "son", "ear", "spy", "kit", "boy", "due", "sen",
             "oaf", "mix", "hep", "fur", "ada", "bin", "nil", "mia", "ewe", "hit",
             "fix", "sad", "rib", "eye", "hop", "haw", "wax", "mid", "tad", "ken",
             "wad", "rye", "pap", "bog", "gut", "ito", "woe", "our", "ado", "sin",
             "mad", "ray", "hon", "roy", "dip", "hen", "iva", "lug", "asp", "hui",
             "yak", "bay", "poi", "yep", "bun", "try", "lad", "elm", "nat", "wyo",
             "gym", "dug", "toe", "dee", "wig", "sly", "rip", "geo", "cog", "pas",
             "zen", "odd", "nan", "lay", "pod", "fit", "hem", "joy", "bum", "rio",
             "yon", "dec", "leg", "put", "sue", "dim", "pet", "yaw", "nub", "bit",
             "bur", "sid", "sun", "oil", "red", "doc", "moe", "caw", "eel", "dix",
             "cub", "end", "gem", "off", "yew", "hug", "pop", "tub", "sgt", "lid",
             "pun", "ton", "sol", "din", "yup", "jab", "pea", "bug", "gag", "mil",
             "jig", "hub", "low", "did", "tin", "get", "gte", "sox", "lei", "mig",
             "fig", "lon", "use", "ban", "flo", "nov", "jut", "bag", "mir", "sty",
             "lap", "two", "ins", "con", "ant", "net", "tux", "ode", "stu", "mug",
             "cad", "nap", "gun", "fop", "tot", "sow", "sal", "sic", "ted", "wot",
             "del", "imp", "cob", "way", "ann", "tan", "mci", "job", "wet", "ism",
             "err", "him", "all", "pad", "hah", "hie", "aim", "ike", "jed", "ego",
             "mac", "baa", "min", "com", "ill", "was", "cab", "ago", "ina", "big",
             "ilk", "gal", "tap", "duh", "ola", "ran", "lab", "top", "gob", "hot",
             "ora", "tia", "kip", "han", "met", "hut", "she", "sac", "fed", "goo",
             "tee", "ell", "not", "act", "gil", "rut", "ala", "ape", "rig", "cid",
             "god", "duo", "lin", "aid", "gel", "awl", "lag", "elf", "liz", "ref",
             "aha", "fib", "oho", "tho", "her", "nor", "ace", "adz", "fun", "ned",
             "coo", "win", "tao", "coy", "van", "man", "pit", "guy", "foe", "hid",
             "mai", "sup", "jay", "hob", "mow", "jot", "are", "pol", "arc", "lax",
             "aft", "alb", "len", "air", "pug", "pox", "vow", "got", "meg", "zoe",
             "amp", "ale", "bud", "gee", "pin", "dun", "pat", "ten", "mob"]
        )
    """


131. palindrome partitioning
------------------------------------

.. code-block:: python

    Given a string s, partition s such that every substring of the partition is a palindrome.


    Return all possible palindrome partitioning of s.


    For example, given s = "aab",

    Return

    [
      ["aa","b"],
      ["a","a","b"]
    ]


    =================================================================
    class Solution(object):
      def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        pal = [[False for i in range(0, len(s))] for j in range(0, len(s))]
        ans = [[[]]] + [[] for _ in range(len(s))]

        for i in range(0, len(s)):
          for j in range(0, i + 1):
            if (s[j] == s[i]) and ((j + 1 > i - 1) or (pal[j + 1][i - 1])):
              pal[j][i] = True
              for res in ans[j]:
                a = res + [s[j:i + 1]]
                ans[i + 1].append(a)
        return ans[-1]


    =================================================================
    class Solution(object):
        def partition(self, s):
            if not s:
                return []
            self.result = []
            self.end = len(s)
            self.str = s

            self.is_palindrome = [[False for i in range(self.end)]
                                  for j in range(self.end)]

            for i in range(self.end-1, -1, -1):
                for j in range(self.end):
                    if i > j:
                        pass
                    elif j-i < 2 and s[i] == s[j]:
                        self.is_palindrome[i][j] = True
                    elif self.is_palindrome[i+1][j-1] and s[i] == s[j]:
                        self.is_palindrome[i][j] = True
                    else:
                        self.is_palindrome[i][j] = False

            self.palindrome_partition(0, [])
            return self.result

        def palindrome_partition(self, start, sub_strs):
            if start == self.end:
                # It's confused the following sentence doesn't work.
                # self.result.append(sub_strs)
                self.result.append(sub_strs[:])
                return

            for i in range(start, self.end):
                if self.is_palindrome[start][i]:
                    sub_strs.append(self.str[start:i+1])
                    self.palindrome_partition(i+1, sub_strs)
                    sub_strs.pop()      # Backtracking here


    if __name__ == "__main__":
        sol = Solution()
        print sol.partition("aab")
        print sol.partition("aabb")
        print sol.partition("aabaa")
        print sol.partition("acbca")
        print sol.partition("acbbca")



132. palindrome partitioning 2
----------------------------------

.. code-block:: python

Given a string s, partition s such that every substring of the partition is a palindrome.


Return the minimum cuts needed for a palindrome partitioning of s.


For example, given s = "aab",
Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.

=================================================================
class Solution(object):
  def minCut(self, s):
    """
    :type s: str
    :rtype: int
    """
    pal = [[False for j in range(0, len(s))] for i in range(0, len(s))]
    dp = [len(s) for _ in range(0, len(s) + 1)]
    for i in range(0, len(s)):
      for j in range(0, i + 1):
        if (s[i] == s[j]) and ((j + 1 > i - 1) or (pal[i - 1][j + 1])):
          pal[i][j] = True
          dp[i + 1] = min(dp[i + 1], dp[j] + 1) if j != 0 else 0
    return dp[-1]


=================================================================
class Solution(object):
    """
    Dynamic Programming:
    cuts[i]: minimum cuts needed for a palindrome partitioning of s[i:]
    is_palindrome[i][j]: whether s[i:i+1] is palindrome
    """
    def minCut(self, s):
        if not s:
            return 0
        s_len = len(s)

        is_palindrome = [[False for i in range(s_len)]
                         for j in range(s_len)]

        cuts = [s_len-1-i for i in range(s_len)]
        for i in range(s_len-1, -1, -1):
            for j in range(i, s_len):
                # if self.is_palindrome(i, j):
                if ((j-i < 2 and s[i] == s[j]) or
                        (s[i] == s[j] and is_palindrome[i+1][j-1])):
                    is_palindrome[i][j] = True
                    if j == s_len - 1:
                        cuts[i] = 0
                    else:
                        cuts[i] = min(cuts[i], 1+cuts[j+1])
                else:
                    pass

        return cuts[0]

"""
if __name__ == "__main__":
    sol = Solution()
    print sol.minCut("aab")
    print sol.minCut("aabb")
    print sol.minCut("aabaa")
    print sol.minCut("acbca")
    print sol.minCut("acbbca")
"""



=================================================================
optimized
class Solution(object):
    """
    Dynamic Programming:
    """
    def minCut(self, s):
        s_len = len(s)
        # number of minnum cuts for the pre i characters
        min_cuts = [i-1 for i in range(s_len+1)]

        for i in range(s_len):
            # odd length palindrome
            j = 0
            while i-j >= 0 and i+j < s_len:
                if s[i-j] == s[i+j]:
                    min_cuts[i+j+1] = min(min_cuts[i+j+1], min_cuts[i-j]+1)
                    j += 1
                else:
                    break
            # even length palindrome
            j = 1
            while i-j+1 >= 0 and i+j < s_len:
                if s[i-j+1] == s[i+j]:
                    min_cuts[i+j+1] = min(min_cuts[i+j+1], min_cuts[i-j+1]+1)
                    j += 1
                else:
                    break

        return min_cuts[s_len]

"""
if __name__ == "__main__":
    sol = Solution()
    print sol.minCut("aab")
    print sol.minCut("aabb")
    print sol.minCut("aabaa")
    print sol.minCut("acbca")
    print sol.minCut("acbbca")
"""



140. Word break 2
--------------------

.. code-block:: python

    Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. You may assume the dictionary does not contain duplicate words.



    Return all such possible sentences.



    For example, given
    s = "catsanddog",
    dict = ["cat", "cats", "and", "sand", "dog"].



    A solution is ["cats and dog", "cat sand dog"].



    UPDATE (2017/1/4):
    The wordDict parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.


    =================================================================
    class Solution(object):
      def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        res = []
        if not self.checkWordBreak(s, wordDict):
          return res
        queue = [(0, "")]
        slen = len(s)
        lenList = [l for l in set(map(len, wordDict))]
        while queue:
          tmpqueue = []
          for q in queue:
            start, path = q
            for l in lenList:
              if start + l <= slen and s[start:start + l] in wordDict:
                newnode = (start + l, path + " " + s[start:start + l] if path else s[start:start + l])
                tmpqueue.append(newnode)
                if start + l == slen:
                  res.append(newnode[1])
          queue, tmpqueue = tmpqueue, []
        return res

      def checkWordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        queue = [0]
        slen = len(s)
        lenList = [l for l in set(map(len, wordDict))]
        visited = [0 for _ in range(0, slen + 1)]
        while queue:
          tmpqueue = []
          for start in queue:
            for l in lenList:
              if s[start:start + l] in wordDict:
                if start + l == slen:
                  return True
                if visited[start + l] == 0:
                  tmpqueue.append(start + l)
                  visited[start + l] = 1
          queue, tmpqueue = tmpqueue, []
        return False


    =================================================================
    class Solution(object):
        """
        Dynamic Programming
        dp[i]: if s[i:] can be broken to wordDict. then:
        dp[i-1] = s[i:i+k] in wordDict and dp[i+k+1], for all the possible k.
        """
        def wordBreak(self, s, wordDict):
            if not s:
                return [""]

            self.s_len = len(s)
            self.result = []
            self.str = s
            self.words = wordDict

            dp = [False for i in range(self.s_len + 1)]
            dp[-1] = True

            for i in range(self.s_len - 1, -1, -1):
                k = 0
                while k + i < self.s_len:
                    cur_fisrt_word = self.str[i:i+k+1]
                    if cur_fisrt_word in self.words and dp[i + k + 1]:
                        dp[i] = True
                        break

                    k += 1

            self.word_break(0, [], dp)
            return self.result

        # Depth First Search
        def word_break(self, start, word_list, dp):
            if start == self.s_len:
                self.result.append(" ".join(word_list))
                return

            k = 0
            while start+k < self.s_len:
                cur_word = self.str[start:start+k+1]
                if cur_word in self.words and dp[start+k+1]:
                    word_list.append(cur_word)
                    self.word_break(start+k+1, word_list, dp)
                    word_list.pop()
                k += 1
    """
    "a"
    []
    ""
    []
    "catsanddog"
    ["cat","cats","and","sand","dog"]
    "leetcode"
    ["leet", "code", "lee", "t"]
    """


279. Perfect squares
----------------------

.. code-block:: python

    Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.



    For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        squares = []
        j = 1
        while j * j <= n:
          squares.append(j * j)
          j += 1
        level = 0
        queue = [n]
        visited = [False] * (n + 1)
        while queue:
          level += 1
          temp = []
          for q in queue:
            for factor in squares:
              if q - factor == 0:
                return level
              if q - factor < 0:
                break
              if visited[q - factor]:
                continue
              temp.append(q - factor)
              visited[q - factor] = True
          queue = temp
        return level


    =================================================================
    # Dynamic Programming with static variable
    class Solution(object):
        # Since dp is a static vector, we have already calculated the result
        # during previous function calls and we can just return the result now.
        _dp = [0]

        def numSquares(self, n):
            dp = self._dp
            while len(dp) <= n:
                i = len(dp)
                min_count = 2 ** 31 - 1
                for j in range(1, int(i**0.5) + 1):
                    min_count = min(min_count, dp[i-j*j]+1)
                dp.append(min_count)
            return dp[n]


    # Dynamic Programming
    # Easy to undersrtand but unfortually "Time Limit Exceeded"
    class Solution_2(object):
        def numSquares(self, n):
            dp = [0] * (n+1)
            for i in range(1, n+1):
                min_count = 2 ** 31 - 1
                for j in range(1, int(i**0.5) + 1):
                    min_count = min(min_count, dp[i-j*j]+1)
                dp[i] = min_count
            return dp[n]

    """
    1
    12
    13
    156
    """


322. Coin Change
--------------------

.. code-block:: python

    You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.



    Example 1:
    coins = [1, 2, 5], amount = 11
    return 3 (11 = 5 + 5 + 1)



    Example 2:
    coins = [2], amount = 3
    return -1.



    Note:
    You may assume that you have an infinite number of each kind of coin.


    Credits:Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.

    =================================================================
    class Solution(object):
      def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
          for coin in coins:
            if i - coin >= 0:
              dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float("inf") else -1

    =================================================================
    class Solution(object):
        def coinChange(self, coins, amount):
            """
            Very classic dynamic programming problem, like 0-1 Knapsack problem.
            dp[i] is the fewest number of coins making up amount i,
            then for every coin in coins, dp[i] = min(dp[i - coin] + 1).
            """
            dp = [amount + 1] * (amount+1)
            dp[0] = 0
            for i in xrange(amount+1):
                for coin in coins:
                    if coin <= i:
                        dp[i] = min(dp[i], dp[i-coin]+1)
            return -1 if dp[amount] > amount else dp[amount]


    class Solution_2(object):
        def coinChange(self, coins, amount):
            # BFS Way.  Scan the possible tree level by level. More Faster!
            if amount == 0:
                return 0
            amounts = [False] * (amount + 1)
            coins_sum = [0]
            count = 0

            # upper bound on number of coins (+1 to represent the impossible case)
            coins.sort(reverse = True)
            upperBound = amount / coins[-1] + 1

            # Use upperBound to pruning.
            while coins_sum and count < upperBound:
                new_coins_sum = []
                count += 1
                for s in coins_sum:
                    for coin in coins:
                        new_sum = s + coin
                        if new_sum == amount:
                            return count
                        elif new_sum > amount:
                            continue
                        elif not amounts[new_sum]:
                            amounts[new_sum] = True
                            new_coins_sum.append(new_sum)
                        else:
                            pass
                coins_sum = new_coins_sum
            return -1

    """
    [1, 2, 5]
    11
    [1]
    0
    [2]
    3
    """




324. Wiggle Sort 2
--------------------

.. code-block:: python

    Given an unsorted array nums, reorder it such that
    nums[0] < nums[1] > nums[2] < nums[3]....



    Example:
    (1) Given nums = [1, 5, 1, 1, 6, 4], one possible answer is [1, 4, 1, 5, 1, 6].
    (2) Given nums = [1, 3, 2, 2, 3, 1], one possible answer is [2, 3, 1, 3, 1, 2].



    Note:
    You may assume all input has valid answer.



    Follow Up:
    Can you do it in O(n) time and/or in-place with O(1) extra space?


    Credits:Special thanks to @dietpepsi for adding this problem and creating all test cases.

    =================================================================
    import random


    class Solution(object):
      def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 2:
          nums.sort()
          return
        numscopy = nums + []
        mid = self.quickselect(0, len(nums) - 1, nums, len(nums) / 2 - 1)
        ans = [mid] * len(nums)
        if len(nums) % 2 == 0:
          l = len(nums) - 2
          r = 1
          for i in range(0, len(nums)):
            if nums[i] < mid:
              ans[l] = nums[i]
              l -= 2
            elif nums[i] > mid:
              ans[r] = nums[i]
              r += 2
        else:
          l = 0
          r = len(nums) - 2
          for i in range(0, len(nums)):
            if nums[i] < mid:
              ans[l] = nums[i]
              l += 2
            elif nums[i] > mid:
              ans[r] = nums[i]
              r -= 2
        for i in range(0, len(nums)):
          nums[i] = ans[i]

      def quickselect(self, start, end, A, k):
        if start == end:
          return A[start]

        mid = self.partition(start, end, A)

        if mid == k:
          return A[k]
        elif mid > k:
          return self.quickselect(start, mid - 1, A, k)
        else:
          return self.quickselect(mid + 1, end, A, k)

      def partition(self, start, end, A):
        left, right = start, end
        pivot = A[left]
        while left < right:
          while left < right and A[right] <= pivot:
            right -= 1
          A[left] = A[right]
          while left < right and A[left] >= pivot:
            left += 1
          A[right] = A[left]
        A[left] = pivot
        return left


    =================================================================
    class Solution(object):
        def wiggleSort(self, nums):
            """
            Clear solutionm, explanation and proof can be found here:
            https://leetcode.com/discuss/76965/3-lines-python-with-explanation-proof
            """
            nums.sort()
            half = (len(nums[::2])) - 1
            # Consider [4,5,5,6]
            # half = (len(nums)+1) // 2
            # nums[::2], nums[1::2] = nums[:half], nums[half:]
            nums[::2], nums[1::2] = nums[half::-1], nums[:half:-1]


    class Solution_2(object):
        def wiggleSort(self, nums):
            """
            O(n)-time O(1)-space solution, no sort here.
            According to:
            https://leetcode.com/discuss/77133/o-n-o-1-after-median-virtual-indexing
            """


    """
    [4, 5, 5, 6]
    [1, 5, 1, 1, 6, 4]
    [1, 3, 2, 2, 3, 1]
    """

