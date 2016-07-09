---
layout: post
title:  "Google Foo.bar: Breeding Like Rabbits"
date:   2016-07-08 18:00:00 -0700
categories: foo.bar
---

This is a quite fun question in Google's secret foo.bar challange.

{% highlight html %}
Breeding like rabbits  
=====================  
As usual, the zombie rabbits (zombits) are breeding... like rabbits! But instead of following the Fibonacci sequence like all good rabbits do, the zombit population changes according to this bizarre formula, where R(n) is the number of zombits at time n:  
 
R(0) = 1  
R(1) = 1  
R(2) = 2  
R(2n) = R(n) + R(n + 1) + n (for n > 1)  
R(2n + 1) = R(n - 1) + R(n) + 1 (for n >= 1)  
 
(At time 2, we realized the difficulty of a breeding program with only one zombit and so added an additional zombit.)  
 
Being bored with the day-to-day duties of a henchman, a bunch of Professor Boolean/'s minions passed the time by playing a guessing game: when will the zombit population be equal to a certain amount? Then, some clever minion objected that this was too easy, and proposed a slightly different game: when is the last time that the zombit population will be equal to a certain amount? And thus, much fun was had, and much merry was made.  
 
(Not in this story: Professor Boolean later downsizes his operation, and you can guess what happens to these minions.)  
 
Write a function answer(str_S) which, given the base-10 string representation of an integer S, returns the largest n such that R(n) = S. Return the answer as a string in base-10 representation. If there is no such n, return "None". S will be a positive integer no greater than 10^25.  
 
Languages 
 =========  
To provide a Python solution, edit solution.py  
To provide a Java solution, edit solution.java 
 
Test cases 
 ==========  
Inputs:  
(string) str_S = "7"  
Output:  
(string) "4"  
 
Inputs: (string) str_S = "100"  
Output: (string) "None"  
 
Use verify [file] to test your solution and see how it does. When you are finished editing your code, use submit [file] to submit your answer. If your solution passes the test cases, it will be removed from your home folder. 

{% endhighlight %}

Omitting all the rabbit or professor related stuff, the core problem is that we have a sequence of numbers defined by the set of recursive equations, then for a given number **S**, the goal is to find an index **n** such that **R(n)=S**.

You might get the illusion that we can probably derive some sort of analytical formula to solve it in constant time. It's exact what I did in the beginning, unfortunately it's quite difficult to do (if possible).

So go back to **searching**. Heck, if we just do a linear search and find the last element such that **R(n)<=S**, problem is easily solved. This certainly won't work because **S** can go as large as 10^25.

Binary search doesn't work either because the sequence is not monotonously increasing. However, here comes the key observation: **the odd and even sub-sequences are monotonously increasing, respectively**, which means that we can do binary search on each of them separately.

Now that we reduced the outer loop (searching) to O(logn), we turn to the inner loop. In each iteration of the binary search, we have to call the recursive formulas many times in order to evaluate **R(n)**, and the upper bound of the number of recursive calls is not O(logn). Since each **R(2n)** or **R(2n+1)** depends on two previous terms, the recursion might lead to as many as O(n) calls.

As we state above, O(n) is not acceptable because **S** can be extremely large.

My solution to this is hashing. For each **n** for which **R(n)** is calculated, we stash the result in a hash map ("dict" in the attached code). Note that this only reduces the time complexity of the evaluation of **R(n)** to **approximately** O(logn), but it works fairly well in practice.

The final time complexity is **approximately** O(logn*logn).

{% highlight python %}

def getR(n, dict): 
    if n in dict: 
        return dict[n] 
    if n == 0 or n == 1: 
        return 1 
    if n == 2: 
        return 2 
    if n % 2 == 0: 
        dict[n] = getR(n/2,dict) + getR(n/2+1,dict) + n/2 
        return dict[n] 
    elif n % 2 == 1: 
        dict[n] = getR(n/2-1,dict) + getR(n/2,dict) + 1 
        return dict[n]

def searchOdd(r): 
    p1 = 0 
    p2 = r / 2 + 1 
    while (p1 <= p2): 
        mid = (p1+p2)/2 
        dict = {} 
        Rmid = getR(2*mid+1,dict) 
        if Rmid == r: 
            return str(2*mid+1) 
        elif Rmid < r: 
            p1 = mid+1 
        else: 
            p2 = mid-1 
    return "None"

def searchEven(r): 
    p1 = 0 
    p2 = r / 2 + 1 
    while (p1 <= p2): 
        mid = (p1+p2)/2 
        dict = {} 
        Rmid = getR(2*mid,dict) 
        if Rmid == r: 
            return str(2*mid) 
        elif Rmid < r: 
            p1 = mid+1 
        else: 
            p2 = mid-1 
    return "None"

def answer(str_S): 
    r = int(str_S) 
    retOdd = searchOdd(r) 
    if retOdd != "None": 
        return retOdd 
    else: 
        return searchEven(r) 

{% endhighlight %}


