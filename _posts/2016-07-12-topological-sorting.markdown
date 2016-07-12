---
layout: post
title:  "Lintcode: Topological Sorting"
date:   2016-07-12 01:00:00 -0700
categories: leetcode
---

This is a problem on Lintcode, I found a quite different solution to it.

The question:

Given an directed graph, a topological order of the graph nodes is defined as follow:

* For each directed edge A -> B in graph, A must before B in the order list.
* The first node in the order can be any node in the graph with no nodes direct to it.

Find any topological order for the given graph.

Try solve it yourself: [link](http://www.lintcode.com/en/problem/topological-sorting/)

Solution:

{% highlight python %}
# Definition for a Directed graph node
# class DirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: A list of graph nodes in topological order.
    """
    def topSort(self, graph):
        # write your code here
        sln = []
        graph = set(graph)
        while len(graph):
            hs = set()
            for node in graph:
                hs.add(node)
            for node in graph:
                for nb in node.neighbors:
                    if nb in hs:
                        hs.remove(nb)
            for node in hs:
                graph.remove(node)
                sln.append(node)
        return sln
{% endhighlight %}

The idea, as covered in Kleinberg's book, is to iteratively find nodes that has no incoming edges, remove them and also add them to the topological ordering, until the graph is empty.

To find all nodes without incoming edges, create a hash set, first push all nodes into the hash set, then scan the neighbors list of each node, and remove nodes from the hash set that appears in the neighbor lists. What's left in the hash set are the nodes that has no incoming edges in this iteration.

As the general idea states, we'd like to remove processed nodes from the graph. Normally this could be done by keeping track of a visited array, and keep visiting it in each iteration or passing it as an argument recursively. 

That's a good solution, but not elegent enough. Here I simply convert the graph into a hash set, then actually remove all visied nodes on the fly. There's a little bit overhead, but time complexity is still the same since we get constant time removal, and because the graph shinks on the fly, the algorithm might be slightly fast for large n.

The resulting code couldn't be any shorter, and every part is like plain English, god I love Python.



