Graph - Easy
=======================================


`Github <https://github.com/newsteinking/leetcode>`_ | https://github.com/newsteinking/leetcode

133. Clone Graph
--------------------

.. code-block:: python

    Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.




    OJ's undirected graph serialization:


    Nodes are labeled uniquely.


    We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.




    As an example, consider the serialized graph {0,1,2#1,2#2,2}.



    The graph has a total of three nodes, and therefore contains three parts as separated by #.

    First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
    Second node is labeled as 1. Connect node 1 to node 2.
    Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.




    Visually, the graph looks like the following:

           1
          / \
         /   \
        0 --- 2
             / \
             \_/



    =================================================================
    class Solution:
      # @param node, a undirected graph node
      # @return a undirected graph node
      def cloneGraph(self, node):
        graph = {}
        visited = set()

        def dfs(node, visited, graph):
          if not node or node.label in visited:
            return
          visited |= {node.label}
          if node.label not in graph:
            graph[node.label] = UndirectedGraphNode(node.label)
          newNode = graph[node.label]

          for nbr in node.neighbors:
            if nbr.label not in graph:
              graph[nbr.label] = UndirectedGraphNode(nbr.label)
            newNode.neighbors.append(graph[nbr.label])
            dfs(nbr, visited, graph)
          return newNode

        return dfs(node, visited, graph)


    =================================================================
    class Solution(object):
        def cloneGraph(self, node):
            if not node:
                return None

            copyed_node_pair = {}
            copy_head = UndirectedGraphNode(node.label)
            copy_head.neighbors = []
            copyed_node_pair[node] = copy_head

            nodes_stack = []
            nodes_stack.append(node)
            while nodes_stack:
                one_node = nodes_stack.pop()

                for neighbor in one_node.neighbors:
                    if neighbor not in copyed_node_pair:
                        copy_node = UndirectedGraphNode(neighbor.label)
                        copy_node.neighbors = []
                        copyed_node_pair[neighbor] = copy_node
                        nodes_stack.append(neighbor)

                    copyed_node_pair[one_node].neighbors.append(
                        copyed_node_pair[neighbor])

            return copy_head

    """
    {0,0,0}
    {0,1,2#1,2#2,2}
    """



207. Course Schedule
-------------------------

.. code-block:: python

    There are a total of n courses you have to take, labeled from 0 to n - 1.

    Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]


    Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?


    For example:
    2, [[1,0]]
    There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.

    2, [[1,0],[0,1]]
    There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.

    Note:

    The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
    You may assume that there are no duplicate edges in the input prerequisites.



    click to show more hints.

    Hints:

    This problem is equivalent to finding if a cycle exists in a directed graph. If a cycle exists, no topological ordering exists and therefore it will be impossible to take all courses.
    Topological Sort via DFS - A great video tutorial (21 minutes) on Coursera explaining the basic concepts of Topological Sort.
    Topological sort could also be done via BFS.

    =================================================================
    class Solution(object):
      def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        def dfs(start, parent, visited, graph):
          visited[start] = 1
          for nbr in graph[start]:
            if visited[nbr] == 1:
              return False
            if dfs(nbr, start, visited, graph) == False:
              return False
          visited[start] = 2
          return True

        graph = [[] for _ in range(0, numCourses)]
        for pre in prerequisites:
          start, end = pre
          graph[start].append(end)

        visited = [0 for _ in range(0, numCourses)]

        for pre in prerequisites:
          start, end = pre
          if visited[start] == 0:
            if dfs(start, None, visited, graph) == False:
              return False
        return True


    =================================================================
    class Solution(object):
        """
        Topological Sort:
        1. Find a "start nodes" which have no incoming edges;
        2. delete the node, update the graph. Then goto 1
        If all the nodes can be deleted, then can finish the course.
        """
        def canFinish(self, numCourses, prerequisites):
            course_req_dict = {}
            # pre_count: the num of one node's incoming edges.
            pre_count = [0] * numCourses
            for edge in prerequisites:
                if edge[1] not in course_req_dict:
                    course_req_dict[edge[1]] = [edge[0]]
                else:
                    course_req_dict[edge[1]].append(edge[0])
                pre_count[edge[0]] += 1

            # Keep nodes which have no incoming edges.
            available = [i for i, v in enumerate(pre_count) if v == 0]
            while available:
                course = available[0]
                del available[0]

                for post_course in course_req_dict.get(course, []):
                    pre_count[post_course] -= 1
                    if pre_count[post_course] == 0:
                        available.append(post_course)
            return sum(pre_count) == 0

    """
    1
    []
    10
    [[1,2],[3,4],[4,5],[5,6],[5,8],[5,9]]
    10
    [[1,2],[3,4],[4,5],[5,6],[5,8],[6,4]]
    """



210. Course Schedule 2
--------------------------

.. code-block:: python

    There are a total of n courses you have to take, labeled from 0 to n - 1.

    Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]


    Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

    There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.


    For example:
    2, [[1,0]]
    There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]

    4, [[1,0],[2,0],[3,1],[3,2]]
    There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].

    Note:

    The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
    You may assume that there are no duplicate edges in the input prerequisites.



    click to show more hints.

    Hints:

    This problem is equivalent to finding the topological order in a directed graph. If a cycle exists, no topological ordering exists and therefore it will be impossible to take all courses.
    Topological Sort via DFS - A great video tutorial (21 minutes) on Coursera explaining the basic concepts of Topological Sort.
    Topological sort could also be done via BFS.


    =================================================================
    class Solution(object):
      def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        def dfs(start, visited, graph, ans):
          visited[start] = 1
          for nbr in graph[start]:
            if visited[nbr] == 1:
              return False
            if visited[nbr] != 0:
              continue
            if dfs(nbr, visited, graph, ans) == False:
              return False
          ans.append(start)
          visited[start] = 2
          return True

        graph = [[] for _ in range(0, numCourses)]
        ans = []

        for pre in prerequisites:
          start, end = pre
          graph[start].append(end)

        visited = [0 for _ in range(0, numCourses)]

        for pre in prerequisites:
          start, end = pre
          if visited[start] != 0:
            continue
          if dfs(start, visited, graph, ans) == False:
            return []
        for i in range(0, numCourses):
          if visited[i] == 0:
            ans.append(i)
        return ans


    =================================================================
    class Solution(object):
        """
        Topological Sort:
        1. Find a "start nodes" which have no incoming edges;
        2. delete the node, update the graph. Then goto 1
        If all the nodes can be deleted, then can finish the course.
        """
        def findOrder(self, numCourses, prerequisites):
            edges_hash = {i: [] for i in range(numCourses)}
            in_degree = [0] * numCourses
            for edge in prerequisites:
                edges_hash[edge[1]].append(edge[0])
                in_degree[edge[0]] += 1

            correct_orders = []
            availables = [i for i, v in enumerate(in_degree) if v == 0]
            while availables:
                course = availables[0]
                correct_orders.append(course)
                del availables[0]
                for co in edges_hash[course]:
                    in_degree[co] -= 1
                    if in_degree[co] == 0:
                        availables.append(co)
            if not sum(in_degree):
                return correct_orders
            else:
                return []

    if __name__ == '__main__':
        sol = Solution()
        print sol.findOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
        print sol.findOrder(4, [[1, 0], [2, 0], [0, 1], [3, 2]])

