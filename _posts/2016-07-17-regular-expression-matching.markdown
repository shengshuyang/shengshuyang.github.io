---
layout: post
title:  "Lintcode: Regular Expression Matching"
date:   2016-07-16 18:00:00 -0700
categories: leetcode
---

Very interesting problem. It's a regular 2D dynamic programming problem, but understanding the problem is actually harder than solving it.

It's actually easier to think of `*` as a "counter" that takes values from 0 to infinity. For example `ab` matches the pattern `c*a*b*` because `c*a*b*` can be interpreted as `c0a1b1`. Similarly `aaaaaa` matches `a*` because `a*` can be interpreted as `a6`.

Based on this concept, the pattern `.*` can match any string because it can be `.0` which matches empty string, or `.100` which matches 100 arbitrary characters.

{% highlight python %}
class Solution:
    """
    @param s: A string 
    @param p: A string includes "." and "*"
    @return: A boolean
    """
    def isMatch(self, s, p):
        # write your code here
        opt = [[False]*(len(s)+1) for i in range(len(p)+1)]
        # insert dummy variables to the beginning so s and p are also 1 based
        s = "0" + s
        p = "0" + p
        opt[0][0] = True
        for i in range(1,len(p)):
            for j in range(len(s)):
            
                # if the current pattern is "." or a-z, the situation is quite
                # trivial, we simply check if p[:i-1] matches s[:j-1] and then
                # if p[i] matches s[j]
                if p[i] == "." or p[i] == s[j]:
                    if opt[i-1][j-1] == True:
                        opt[i][j] = True
                
                # if current pattern is "*", it can match 0, 1 or more chars.
                elif p[i] == "*":
                    
                    # matching 1 preceding character. e.g., "ba*" |= "ba"
                    if opt[i-1][j]:
                        opt[i][j] = True
                    
                    # matching 0 preceding character. e.g., "ba*" |= "b", here
                    # the preciding character is "a" and it has 0 count
                    elif i > 1 and opt[i-2][j]:
                        opt[i][j] = True
                    
                    # matching more. e.g., "ba*" |= "baaaaaaa". using the 
                    # other two cases, we know "ba*" |= "ba", so for "baa"
                    # we use the fact that "ba*" |= "ba" and the new character
                    # s[j] = "a" equals to s[j-1], so it is also a match.
                    elif opt[i][j-1] and s[j] == s[j-1]:
                        opt[i][j] = True
                        
                    # a special case here is ".*" matches any string. e.g., 
                    # ".*" |= "abcdef" because it represents zero or more
                    # ".". In this case ".*" = "......" |= "abcdef" 
                    elif opt[i][j-1] and p[i-1] == '.':
                        opt[i][j] = True
        return opt[-1][-1]
{% endhighlight %}




